"""The hosted knowledge service (knowledge_service/) and the client the chat site uses to call it.
Needs the disposable Neo4j for the `kb` fixture; Chroma and Redis are in-memory fakes. All names and cases are FICTIONAL."""
import logging
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from knowledge_service import app as svc
from backend.knowledge import graphrag
from backend.knowledge.remote import KnowledgeUnavailable, RemoteKnowledge
from test_graphrag import load

TOKEN = "t" * 32


def service_client(kb=None, rag=None, mem=None, **kw):
    return TestClient(svc.create_app(TOKEN, kb, rag, mem, **kw), headers={"Authorization": f"Bearer {TOKEN}"})


# ---- the service ------------------------------------------------------------------------------------
def test_service_refuses_to_start_without_a_long_token():
    for bad in ("", None, "short", "x" * 23):
        with pytest.raises(svc.ConfigError):
            svc.create_app(bad)


def test_healthz_needs_no_token_and_touches_no_store():
    assert TestClient(svc.create_app(TOKEN)).get("/healthz").json() == {"status": "ok"}


def test_every_other_path_needs_the_token_and_gives_the_same_401(kb, rag_store, memory):
    app = svc.create_app(TOKEN, kb, rag_store, memory)
    anon = TestClient(app)
    answers = set()
    for headers in ({}, {"Authorization": "Bearer wrong" + "x" * 30}, {"Authorization": "Basic " + TOKEN}, {"Authorization": TOKEN}):
        for method, path, kw in (("POST", "/v1/context", {"json": {"question": "bribe"}}), ("GET", "/readyz", {}),
                                 ("POST", "/v1/remember", {"json": {"session": "s1", "question": "q"}}), ("DELETE", "/v1/memory?session=s1", {})):
            r = anon.request(method, path, headers=headers, **kw)
            assert r.status_code == 401
            answers.add(r.text)
    assert len(answers) == 1                                                            # missing and wrong tokens look identical


def test_unknown_paths_and_methods_are_refused(kb, rag_store, memory):
    c = service_client(kb, rag_store, memory)
    assert c.get("/").status_code == 404 and c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404
    assert c.get("/v1/context").status_code == 405


def test_context_returns_ranked_cited_items_and_whether_data_exists(kb, rag_store, memory):
    load(kb, rag_store)
    c = service_client(kb, rag_store, memory)
    out = c.post("/v1/context", json={"question": "List the corruption cases of Test Person", "session": "abc"}).json()
    assert out["populated"] is True and out["notes"] == []
    assert out["items"][0]["kind"] == "fact" and "Test Housing Scam" in out["items"][0]["text"]
    assert out["u"]["entities"][0]["name"] == "Test Person" and out["items"][0]["source"]["url"] == "https://example.org/order1"
    assert set(out) == {"items", "notes", "u", "populated"}                              # nothing else leaves the service


def test_input_is_validated(kb, rag_store, memory):
    c = service_client(kb, rag_store, memory)
    assert c.post("/v1/context", json={"question": ""}).status_code == 422
    assert c.post("/v1/context", json={"question": "x" * 1001}).status_code == 422
    assert c.post("/v1/context", json={"question": "ok", "session": "bad session!"}).status_code == 422
    assert c.post("/v1/context", json={"question": "ok", "extra": 1}).status_code == 200      # unknown fields are ignored, not trusted
    assert c.delete("/v1/memory").status_code == 422


def test_no_store_at_all_is_a_503_not_a_crash():
    assert service_client().post("/v1/context", json={"question": "a bribe case"}).status_code == 503


def test_a_failing_retrieval_is_a_clean_502(kb, rag_store, memory, monkeypatch):
    monkeypatch.setattr(graphrag, "build_context", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom with a secret inside")))
    r = service_client(kb, rag_store, memory).post("/v1/context", json={"question": "a bribe case"})
    assert r.status_code == 502 and "secret" not in r.text


def test_rate_limit_and_concurrency_limit(kb, rag_store, memory, monkeypatch):
    c = service_client(kb, rag_store, memory, rate_per_minute=2)
    assert [c.get("/readyz").status_code for _ in range(3)] == [200, 200, 429]
    gate, entered = threading.Event(), threading.Event()

    def slow(*a, **k):
        entered.set()
        gate.wait(5)
        return graphrag.Context(items=[], understanding=graphrag.Understanding("q", "q", [], "general", []))
    monkeypatch.setattr(graphrag, "build_context", slow)
    c2 = service_client(kb, rag_store, memory, max_concurrent=1)
    first = threading.Thread(target=lambda: c2.post("/v1/context", json={"question": "one"}))
    first.start()
    assert entered.wait(5)
    assert c2.post("/v1/context", json={"question": "two"}).status_code == 503         # only one at a time
    gate.set()
    first.join(5)


def test_remember_and_forget_manage_one_users_memory(kb, rag_store, memory):
    c = service_client(kb, rag_store, memory)
    c.post("/v1/remember", json={"session": "u1", "question": "q", "answer": "a", "entities": [{"id": "Person:a", "type": "Person", "name": "A"}]})
    c.post("/v1/remember", json={"session": "u2", "question": "other", "answer": "b"})
    assert memory.recall("u1")["entities"][0]["name"] == "A" and memory.recall("u2")["turns"][0]["q"] == "other"
    assert c.delete("/v1/memory", params={"session": "u1"}).json() == {"forgotten": True}
    assert memory.recall("u1") == {"turns": [], "entities": []} and memory.recall("u2")["turns"]


def test_logs_never_contain_the_token_the_question_or_the_session(kb, rag_store, memory, caplog):
    load(kb, rag_store)
    caplog.set_level(logging.DEBUG)
    c = service_client(kb, rag_store, memory)
    c.post("/v1/context", json={"question": "UNIQUE-QUESTION-TEXT about Test Person", "session": "SESSION-XYZ-123"})
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert TOKEN not in logged and "UNIQUE-QUESTION-TEXT" not in logged and "SESSION-XYZ-123" not in logged


# ---- the client ---------------------------------------------------------------------------------------
def handler_for(status=200, json=None, exc=None):
    def handler(request):
        if exc:
            raise exc
        return httpx.Response(status, json=json)
    return httpx.MockTransport(handler)


def test_client_refuses_to_send_the_token_over_plain_http_or_with_a_short_token():
    with pytest.raises(ValueError, match="plain http"):
        RemoteKnowledge("http://knowledge.example.com", TOKEN)
    with pytest.raises(ValueError, match="shorter"):
        RemoteKnowledge("https://knowledge.example.com", "short")
    with pytest.raises(ValueError, match="address"):
        RemoteKnowledge("ftp://x", TOKEN)
    assert RemoteKnowledge("https://knowledge.example.com/", TOKEN).base_url == "https://knowledge.example.com"
    assert RemoteKnowledge("http://localhost:9000", TOKEN).base_url == "http://localhost:9000"          # loopback is fine for local testing


def test_client_from_env_is_none_when_unset(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_URL", raising=False)
    assert RemoteKnowledge.from_env() is None
    monkeypatch.setenv("KNOWLEDGE_URL", "https://k.example.com")
    monkeypatch.setenv("KNOWLEDGE_TOKEN", TOKEN)
    assert RemoteKnowledge.from_env().base_url == "https://k.example.com"


@pytest.mark.parametrize("kwargs, text", [
    ({"exc": httpx.ReadTimeout("slow")}, "in time"), ({"exc": httpx.ConnectError("down")}, "could not be reached"),
    ({"status": 401}, "refused the token"), ({"status": 503, "json": {}}, "503"), ({"status": 200, "json": {"unexpected": 1}}, "unexpected")])
def test_every_client_failure_becomes_knowledge_unavailable(kwargs, text):
    client = RemoteKnowledge("https://k.example.com", TOKEN, transport=handler_for(**kwargs))
    with pytest.raises(KnowledgeUnavailable, match=text):
        client.context("a bribe case")
    client.remember("s1", "q", "a", [])               # memory calls swallow failures
    client.forget("s1")


def test_client_sends_the_token_and_only_what_it_should():
    seen = {}

    def handler(request):
        seen["auth"], seen["body"], seen["url"] = request.headers.get("authorization"), request.content, str(request.url)
        return httpx.Response(200, json={"items": [], "notes": [], "u": {"question": "q", "resolved": "q", "entities": [], "intent": "general", "terms": [], "followup": False}, "populated": True})
    ctx, populated = RemoteKnowledge("https://k.example.com", TOKEN, transport=httpx.MockTransport(handler)).context("What is a bribe?", "sess1")
    assert populated is True and ctx.items == [] and seen["auth"] == f"Bearer {TOKEN}" and seen["url"] == "https://k.example.com/v1/context"
    assert b"What is a bribe?" in seen["body"] and b"sess1" in seen["body"]


# ---- the chat site talking to the service (the real GraphRAG code on both ends) -----------------------
@pytest.fixture()
def site_for(ckpt_dir, tok_file):
    def build(service_app=None, token=TOKEN, transport=None):
        remote = RemoteKnowledge("http://127.0.0.1:9", token, transport=transport)
        if service_app is not None:
            remote.http = TestClient(service_app, base_url="http://127.0.0.1:9", headers={"Authorization": f"Bearer {token}"})
        app = create_app(ckpt_dir, load_on_start=False, max_concurrent=0, remote_service=remote)
        app.state.service.tokenizer_file = tok_file
        app.state.service.load()
        return TestClient(app)
    return build


def ask(client, text, **extra):
    return client.post("/api/chat", json={"text": text, "max_new_tokens": 20, "greedy": True, **extra}).json()


def test_site_uses_the_context_from_the_service_with_citations(kb, rag_store, memory, site_for):
    load(kb, rag_store)
    body = ask(site_for(svc.create_app(TOKEN, kb, rag_store, memory)), "List the corruption cases of Test Person")
    assert body["searched"] is True and body["kb_entities"] == ["Test Person"] and body["kb_facts"]
    assert body["sources"][0]["url"] == "https://example.org/order1" and body["kb_facts"][0]["source"]["url"] == "https://example.org/order1"


def test_followups_use_memory_kept_by_the_service(kb, rag_store, memory, site_for):
    load(kb, rag_store)
    site = site_for(svc.create_app(TOKEN, kb, rag_store, memory))
    ask(site, "List the corruption cases of Test Person", session_id="s1")
    follow = ask(site, "what about his charges?", session_id="s1")
    assert follow["kb_followup"] is True and follow["kb_entities"] == ["Test Person"]
    assert site.delete("/api/memory", params={"session_id": "s1"}).json() == {"forgotten": True}
    assert "kb_entities" not in ask(site, "what about his charges?", session_id="s1") or ask(site, "what about his charges?", session_id="s1")["kb_entities"] == []


def test_unanswerable_corruption_question_is_refused_when_the_service_has_data(kb, rag_store, memory, site_for):
    load(kb, rag_store)
    body = ask(site_for(svc.create_app(TOKEN, kb, rag_store, memory)), "Tell me about the 2G scam arrests in Kerala")
    assert body["text"] == "Not found in my data." and body["model_used"] is False


@pytest.mark.parametrize("transport", [handler_for(exc=httpx.ConnectError("down")), handler_for(exc=httpx.ReadTimeout("asleep")), handler_for(status=401)])
def test_service_down_asleep_or_refusing_gives_an_honest_message_for_corruption_questions_only(transport, site_for):
    site = site_for(transport=transport)
    assert "unavailable" in ask(site, "List the corruption cases of Test Person")["text"]
    assert ask(site, "hii").get("stop_reason") != "knowledge_base"                           # everything else keeps working


def test_a_wrong_token_on_the_site_is_the_same_as_the_service_being_down(kb, rag_store, memory, site_for):
    site = site_for(svc.create_app(TOKEN, kb, rag_store, memory), token="w" * 32)
    # the TestClient carries the site's (wrong) token, so the service answers 401
    assert "unavailable" in ask(site, "List the corruption cases of Test Person")["text"]


def test_a_misconfigured_url_never_crashes_the_site_and_still_refuses_corruption_questions(ckpt_dir, tok_file, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_URL", "http://knowledge.example.com")        # plain http to another host: the client refuses
    monkeypatch.setenv("KNOWLEDGE_TOKEN", TOKEN)
    app = create_app(ckpt_dir, load_on_start=False, max_concurrent=0)
    app.state.service.tokenizer_file = tok_file
    app.state.service.load()
    assert app.state.remote is None
    site = TestClient(app)
    assert "unavailable" in ask(site, "List the corruption cases of Test Person")["text"]
    assert ask(site, "hii").get("stop_reason") != "knowledge_base"
