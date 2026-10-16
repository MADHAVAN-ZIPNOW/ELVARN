"""Why a live answer said 'answered by the model alone' although stored records and the web exist, and what the site does about it now:
 * a wiki heading is not a note an answer can be written from;
 * a question about the PRESENT is also sent to the web (stored records can be months old), the stored records stay as the fallback;
 * a sleeping knowledge service is said so, instead of a silent model-alone answer.
Offline: the knowledge service and the search engine are fakes. Names and cases are FICTIONAL."""
import httpx
import pytest
from fastapi.testclient import TestClient

from api.server import FRESH, create_app
from backend.knowledge import groundcheck
from backend.knowledge.remote import RemoteKnowledge
from knowledge_service import app as svc
from search.providers.base import RawResult
from search.service import SearchService
from test_knowledge_service import TOKEN, ask
from test_graphrag import load
from test_web_search import FakeProvider


def test_wiki_headings_and_markdown_headings_are_not_usable_notes():
    assert not groundcheck.usable_note("=== Chief Minister of Tamil Nadu (2026-present) ===")
    assert not groundcheck.usable_note("== Early life and political career in the state ==")
    assert not groundcheck.usable_note("## Early life and political career in the state")
    assert groundcheck.usable_note("Test Person became Chief Minister of Test State in May 2026 after the election.")
    assert groundcheck.usable_note("He was born in Test Town; his father was a clerk.")


def test_questions_about_the_present_are_recognised():
    for q in ("Who is the current Chief Minister of Tamil Nadu?", "latest news on the case", "what is happening now", "who won in 2026",
              "Who is the incumbent speaker", "as of today"):
        assert FRESH.search(q), q
    for q in ("What is the capital of France?", "Who was Chief Minister in 1967?", "hi", "Explain the Prevention of Corruption Act"):
        assert not FRESH.search(q), q


@pytest.fixture()
def site(ckpt_dir, tok_file):
    def build(service_app=None, transport=None, provider=None):
        remote = RemoteKnowledge("http://127.0.0.1:9", TOKEN, transport=transport)
        if service_app is not None:
            remote.http = TestClient(service_app, base_url="http://127.0.0.1:9", headers={"Authorization": f"Bearer {TOKEN}"})
        search = SearchService(provider) if provider is not None else None
        app = create_app(ckpt_dir, load_on_start=False, max_concurrent=0, remote_service=remote, **({"search_service": search} if search else {}))
        app.state.service.tokenizer_file = tok_file
        app.state.service.load()
        app.state.service.prompt_budget = lambda: ((lambda text: len(text.split())), 512)     # the live model has a 512-token window; the test model is far smaller
        return TestClient(app)
    return build


WEB = [RawResult("Test State chief minister", "https://news.example.com/cm", "Test Person is the current Chief Minister of Test State, sworn in in May 2026."),
       RawResult("Test State government", "https://www.example.com/gov", "The current Chief Minister of Test State is Test Person, who leads the Test Party government.")]


def test_a_present_day_question_uses_the_web_even_when_stored_records_exist(kb, rag_store, memory, site):
    load(kb, rag_store)
    client = site(svc.create_app(TOKEN, kb, rag_store, memory), provider=FakeProvider(WEB))
    body = ask(client, "Who is the current Chief Minister of Test State?", web_search="auto")
    assert body["search_status"] == "used" and body["evidence_in_prompt"] > 0
    assert {s["url"] for s in body["sources"]} <= {"https://news.example.com/cm", "https://www.example.com/gov"}
    assert "kb_facts" not in body                                                          # answered from the web, not from the records


def test_the_stored_records_stay_when_the_web_gives_nothing_for_a_present_day_question(kb, rag_store, memory, site):
    load(kb, rag_store)
    client = site(svc.create_app(TOKEN, kb, rag_store, memory), provider=FakeProvider([]))
    body = ask(client, "Which is the longest river in Tamil Nadu now?", web_search="auto")
    assert body["searched"] is True and body["sources"] and all(s["url"] != "https://news.example.com/cm" for s in body["sources"])
    assert "kb_facts" in body                                                              # the records answered


def test_a_question_about_the_past_keeps_using_the_stored_records_only(kb, rag_store, memory, site):
    load(kb, rag_store)
    provider = FakeProvider(WEB)
    body = ask(site(svc.create_app(TOKEN, kb, rag_store, memory), provider=provider), "List the corruption cases of Test Person", web_search="auto")
    assert provider.calls == [] and body["kb_facts"] and body["searched"] is True


def test_web_search_off_never_calls_the_web(kb, rag_store, memory, site):
    load(kb, rag_store)
    provider = FakeProvider(WEB)
    ask(site(svc.create_app(TOKEN, kb, rag_store, memory), provider=provider), "Who is the current Chief Minister of Test State?", web_search="off")
    assert provider.calls == []


@pytest.mark.parametrize("transport", [httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("asleep"))),
                                       httpx.MockTransport(lambda r: httpx.Response(503))])
def test_a_sleeping_knowledge_service_is_said_so_and_the_web_still_answers(transport, site):
    client = site(transport=transport, provider=FakeProvider(WEB))
    web = ask(client, "Who is the current Chief Minister of Test State?", web_search="auto")
    assert web["search_status"] == "used" and web["sources"]                                # the web was used even though the records were not reachable
    plain = ask(client, "What do you think about tea?", web_search="auto")
    assert plain["search_status"] == "unavailable" and "stored records" in plain["search_message"] and "waking up" in plain["search_message"]
    assert ask(site(transport=transport), "hii")["search_status"] in ("unavailable", "skipped")


def test_corruption_questions_still_get_the_unavailable_notice_when_the_service_sleeps(site):
    client = site(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("asleep"))))
    body = ask(client, "List the corruption cases of Test Person")
    assert "unavailable" in body["text"] and body["model_used"] is False
