"""Chat integration of GraphRAG: Neo4j facts + Chroma passages + Redis memory, through the model.
Needs the disposable Neo4j (see the `kb` fixture); Chroma and Redis are in-memory fakes. All names and cases are FICTIONAL."""
import json

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from test_graphrag import load
from usermemory.store import Memory


@pytest.fixture()
def client_for(ckpt_dir, tok_file):
    def build(kb=None, rag=None, mem=None):
        app = create_app(ckpt_dir, load_on_start=False, max_concurrent=0, kb_service=kb, rag_service=rag, memory_service=mem)
        app.state.service.tokenizer_file = tok_file
        app.state.service.load()
        return TestClient(app)
    return build


class BrokenKB:
    """A knowledge graph that is configured but unreachable."""
    def __getattr__(self, name):
        def fail(*a, **k):
            raise ConnectionError("neo4j down")
        return fail


class BrokenRedis:
    def __getattr__(self, name):
        def fail(*a, **k):
            raise ConnectionError("redis down")
        return fail


def ask(client, text, **extra):
    return client.post("/api/chat", json={"text": text, **extra}).json()


def test_graph_facts_and_passages_reach_the_model_with_citations(kb, rag_store, memory, client_for):
    load(kb, rag_store)
    body = ask(client_for(kb, rag_store, memory), "List the corruption cases of Test Person")
    assert body.get("stop_reason") != "knowledge_base"                                   # the model writes the answer, from the notes
    assert body["searched"] is True and body["search_status"] == "used"
    assert body["kb_entities"] == ["Test Person"] and body["kb_notes"] == []
    facts = body["kb_facts"]
    assert facts and "Test Housing Scam" in facts[0]["text"] and facts[0]["level"] in ("alleged", "reported", "court_finding")
    assert facts[0]["source"]["url"] == "https://example.org/order1"
    assert body["sources"][0]["url"] == "https://example.org/order1"                      # cited, with its document type
    assert "verified knowledge-graph facts" in body["search_message"] or "memory window" in body["search_message"]


def test_question_about_no_known_entity_uses_passages_only(kb, rag_store, memory, client_for):
    load(kb, rag_store)
    body = ask(client_for(kb, rag_store, memory), "village officer trapped and arrested for demanding a bribe")
    assert body["kb_facts"] == [] and body["kb_entities"] == []
    assert body["sources"][0]["url"] == "https://example.org/pr12" and body["sources"][0]["domain"] == "press release"


def test_corruption_question_nothing_answers_is_refused_not_guessed(kb, rag_store, memory, client_for):
    load(kb, rag_store)
    body = ask(client_for(kb, rag_store, memory), "Tell me about the 2G scam arrests in Kerala")
    assert body["text"] == "Not found in my data." and body["model_used"] is False


def test_small_talk_still_goes_to_the_model_without_notes(kb, rag_store, memory, client_for):
    load(kb, rag_store)
    body = ask(client_for(kb, rag_store, memory), "hii")
    assert body.get("stop_reason") != "knowledge_base" and not body.get("searched") and "kb_facts" not in body


def test_followup_is_resolved_from_the_users_latest_memory(kb, rag_store, memory, client_for):
    load(kb, rag_store)
    c = client_for(kb, rag_store, memory)
    ask(c, "List the corruption cases of Test Person", session_id="s1")
    follow = ask(c, "what about his charges?", session_id="s1")
    assert follow["kb_followup"] is True and follow["kb_entities"] == ["Test Person"] and follow["kb_facts"]
    other = ask(c, "what about his charges?", session_id="someone-else")
    assert "kb_entities" not in other or other["kb_entities"] == []                       # another user has no such context


def test_forget_endpoint_deletes_the_users_memory(kb, rag_store, memory, client_for):
    load(kb, rag_store)
    c = client_for(kb, rag_store, memory)
    ask(c, "List the corruption cases of Test Person", session_id="s1")
    assert c.delete("/api/memory", params={"session_id": "s1"}).json() == {"forgotten": True}
    after = ask(c, "what about his charges?", session_id="s1")
    assert "kb_entities" not in after or after["kb_entities"] == []


def test_stream_sends_sources_with_facts_before_the_answer(kb, rag_store, memory, client_for):
    load(kb, rag_store)
    r = client_for(kb, rag_store, memory).post("/api/chat/stream", json={"text": "cases of Test Person"})
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    kinds = [e["event"] for e in events]
    first = events[kinds.index("sources")]
    assert kinds.index("sources") < kinds.index("done") and first["kb_facts"] and first["sources"][0]["url"] == "https://example.org/order1"
    assert events[-1]["kb_entities"] == ["Test Person"]


def test_vector_branch_down_still_answers_from_the_graph(kb, rag_store, memory, client_for, monkeypatch):
    load(kb, rag_store)
    monkeypatch.setattr(rag_store.__class__, "query", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("chroma down")))
    body = ask(client_for(kb, rag_store, memory), "List the corruption cases of Test Person")
    assert body["kb_facts"] and body["kb_notes"] == ["vector search unavailable"]


def test_all_stores_unreachable_refuses_corruption_questions_only(ckpt_dir, tok_file):
    app = create_app(ckpt_dir, load_on_start=False, max_concurrent=0, kb_service=BrokenKB())
    app.state.service.tokenizer_file = tok_file
    app.state.service.load()
    c = TestClient(app)
    down = ask(c, "List the corruption cases of Test Person")
    assert "unavailable" in down["text"] and down["model_used"] is False
    assert ask(c, "hii").get("stop_reason") != "knowledge_base"


def test_redis_down_does_not_break_the_chat(kb, rag_store, client_for):
    load(kb, rag_store)
    body = ask(client_for(kb, rag_store, Memory(BrokenRedis())), "List the corruption cases of Test Person", session_id="s1")
    assert body["kb_facts"] and body["sources"]


def test_without_any_store_the_chat_is_unchanged(client_for):
    body = ask(client_for(), "List the corruption cases of Test Person")
    assert body.get("stop_reason") != "knowledge_base" and not body.get("searched")
