import httpx
import pytest

from tests.conftest import FakeExtractor

P = "/api/jarvis/memory"
UPDATE = ("Today I fixed the ELVARN reranker bug and tested the Render deployment with Ravi. "
          "I still need to write tests for the source cards. The deployment felt great.")


def good_answer(text):
    return {
        "title": "Fixed ELVARN reranker bug",
        "summary": "Madhavan fixed the ELVARN reranker bug and tested the Render deployment with Ravi. He still needs to write tests for the source cards.",
        "tags": ["ELVARN", "Render", "testing"], "projects": ["ELVARN"], "people": ["Ravi"],
        "tasks": [{"task": "Write tests for the source cards", "status": "pending"}],
        "next_steps": ["Write tests for the source cards"], "sentiment": "positive",
    }


# ---------------------------------------------------------------------------------- manual memories
def test_manual_memory_is_saved_and_enriched(make_client, settings):
    c = make_client(FakeExtractor(settings, answer=good_answer))
    r = c.post(P, json={"content": UPDATE})
    assert r.status_code == 201
    m = r.json()
    assert m["source"] == "manual" and m["content"] == UPDATE
    assert m["title"] == "Fixed ELVARN reranker bug"
    assert m["tags"] == ["elvarn", "render", "testing"] and m["projects"] == ["ELVARN"] and m["people"] == ["Ravi"]
    assert m["tasks"] == [{"task": "Write tests for the source cards", "status": "pending", "due_at": None}]
    assert m["sentiment"] == "positive" and m["raw_transcript"] is None
    assert m["metadata_json"]["extraction"]["status"] == "ok"
    assert c.get(f"{P}/{m['id']}").json()["content"] == UPDATE


def test_original_text_is_preserved_exactly(make_client, settings):
    weird = "  Line one\n\n\tIndented ✓ ‘quotes’ தமிழ் <b>bold</b> & more  \n"
    c = make_client(FakeExtractor(settings, answer=good_answer))
    m = c.post(P, json={"content": weird}).json()
    assert m["content"] == weird  # no trimming, no escaping, no rewriting


def test_ai_summary_is_separate_from_the_original(make_client, settings):
    c = make_client(FakeExtractor(settings, answer=good_answer))
    m = c.post(P, json={"content": UPDATE}).json()
    assert m["summary"] and m["summary"] != m["content"]


def test_missing_ai_fields_stay_empty(make_client, settings):
    c = make_client(FakeExtractor(settings, answer={"title": "", "summary": "", "tags": [], "people": []}))
    m = c.post(P, json={"content": "Went for a walk."}).json()
    assert m["tags"] == [] and m["projects"] == [] and m["people"] == [] and m["tasks"] == [] and m["next_steps"] == []
    assert m["sentiment"] is None
    assert m["title"] == "Went for a walk."          # safe fallback derived from the user's own words
    assert m["summary"] == "Went for a walk."        # extractive fallback, nothing invented


def test_invented_facts_never_reach_the_database(make_client, settings):
    invented = {"title": "Meeting with the CEO of Google", "summary": "Madhavan met Sundar Pichai and closed a million dollar deal.",
                "tags": ["google", "deal"], "projects": ["Project Atlas"], "people": ["Sundar Pichai"],
                "tasks": [{"task": "Sign the acquisition contract", "status": "done"}], "next_steps": ["Fly to California"],
                "sentiment": "positive"}
    c = make_client(FakeExtractor(settings, answer=invented))
    m = c.post(P, json={"content": "Had a quiet day reading about transformers."}).json()
    assert m["people"] == [] and m["projects"] == [] and m["tags"] == [] and m["tasks"] == [] and m["next_steps"] == []
    assert "Pichai" not in m["summary"] and "Google" not in m["title"]
    assert m["metadata_json"]["extraction"]["grounding_dropped"]  # and the removal is recorded


# ---------------------------------------------------------------------------------- voice memories
def test_voice_memory_keeps_the_raw_transcript_and_the_edited_text(make_client, settings):
    c = make_client(FakeExtractor(settings, answer=good_answer))
    raw = "today i fix the elvarn ranker bug and test render with ravi"
    edited = "Today I fixed the ELVARN reranker bug and tested Render with Ravi."
    m = c.post(P, json={"content": edited, "source": "voice", "raw_transcript": raw}).json()
    assert m["source"] == "voice" and m["raw_transcript"] == raw and m["content"] == edited


def test_raw_transcript_is_ignored_for_typed_memories(make_client, settings):
    c = make_client(FakeExtractor(settings, answer=good_answer))
    m = c.post(P, json={"content": "typed text", "source": "manual", "raw_transcript": "sneaky"}).json()
    assert m["raw_transcript"] is None


# ---------------------------------------------------------------------------------- resilience
def test_ai_failure_still_saves_the_raw_memory(make_client, settings):
    c = make_client(FakeExtractor(settings, fail=httpx.ConnectError("groq down")))
    r = c.post(P, json={"content": UPDATE})
    assert r.status_code == 201
    m = r.json()
    assert m["content"] == UPDATE and m["metadata_json"]["extraction"]["status"] == "failed"
    assert m["title"] and m["summary"]  # fallbacks, so the memory is still readable and searchable


def test_failed_extraction_can_be_retried(make_client, settings):
    ex = FakeExtractor(settings, fail=httpx.ConnectError("down"))
    c = make_client(ex)
    mid = c.post(P, json={"content": UPDATE}).json()["id"]
    ex.fail, ex.answer = None, good_answer
    m = c.post(f"{P}/{mid}/extract").json()
    assert m["metadata_json"]["extraction"]["status"] == "ok" and m["people"] == ["Ravi"] and m["title"] == "Fixed ELVARN reranker bug"


def test_retry_never_overwrites_what_you_edited(make_client, settings):
    ex = FakeExtractor(settings, answer=good_answer)
    c = make_client(ex)
    mid = c.post(P, json={"content": UPDATE}).json()["id"]
    c.patch(f"{P}/{mid}", json={"title": "My own title", "summary": "My own summary", "tags": ["mine"]})
    m = c.post(f"{P}/{mid}/extract").json()
    assert m["title"] == "My own title" and m["summary"] == "My own summary" and m["tags"] == ["mine"]


def test_without_a_groq_key_the_memory_is_saved_unenriched(make_client):
    from app.extraction import Extractor
    from app.config import Settings

    s = Settings(database_url=make_client().app.state.settings.database_url, owner_key="test-owner-key-0123456789",
                 owner_email="owner@example.com", groq_api_key="", rate_limit_per_minute=1000)
    c = make_client(Extractor(s), groq_api_key="")
    m = c.post(P, json={"content": "No AI today."}).json()
    assert m["content"] == "No AI today." and m["metadata_json"]["extraction"]["status"] == "skipped"


def test_extract_false_skips_the_ai(make_client, settings):
    ex = FakeExtractor(settings, answer=good_answer)
    c = make_client(ex)
    m = c.post(P, json={"content": "plain", "extract": False}).json()
    assert ex.calls == [] and m["metadata_json"]["extraction"]["status"] == "skipped"


# ---------------------------------------------------------------------------------- validation
@pytest.mark.parametrize("body", [
    {}, {"content": ""}, {"content": "   \n "}, {"content": "x" * 20001}, {"content": "ok", "source": "email"},
    {"content": "ok", "source": "daily_review"}, {"content": "ok", "owner_id": "x"}, {"content": "ok", "sentiment": "ecstatic"},
    {"content": "ok", "tasks": [{"task": "t", "status": "someday"}]},
])
def test_invalid_creates_are_rejected(client, body):
    assert client.post(P, json=body).status_code == 422


def test_lists_are_cleaned_and_capped(make_client, settings):
    c = make_client(FakeExtractor(settings, answer={}))
    m = c.post(P, json={"content": "x", "tags": ["  AI ", "ai", "Research", ""] + [f"t{i}" for i in range(40)]}).json()
    assert m["tags"][:2] == ["ai", "research"] and len(m["tags"]) == 20


# ---------------------------------------------------------------------------------- edit / delete / export
def test_editable_fields_and_immutable_original(client):
    mid = client.post(P, json={"content": "original words", "extract": False}).json()["id"]
    r = client.patch(f"{P}/{mid}", json={"title": "T", "summary": "S", "tags": ["a"], "projects": ["P"], "people": ["Bob"],
                                         "next_steps": ["n"], "tasks": [{"task": "do it", "status": "blocked"}], "sentiment": "mixed"})
    m = r.json()
    assert (m["title"], m["summary"], m["tags"], m["projects"], m["people"], m["sentiment"]) == ("T", "S", ["a"], ["P"], ["Bob"], "mixed")
    assert m["tasks"][0]["status"] == "blocked" and m["content"] == "original words"
    assert client.patch(f"{P}/{mid}", json={"content": "rewritten"}).status_code == 422   # originals are read-only
    assert client.patch(f"{P}/{mid}", json={"source": "email"}).status_code == 422
    assert client.patch(f"{P}/{mid}", json={"clear_sentiment": True}).json()["sentiment"] is None


def test_updated_at_moves_forward_on_edit(client):
    m = client.post(P, json={"content": "x", "extract": False}).json()
    n = client.patch(f"{P}/{m['id']}", json={"title": "new"}).json()
    assert n["updated_at"] > m["updated_at"] and n["created_at"] == m["created_at"]


def test_delete_hides_the_memory_everywhere(client):
    mid = client.post(P, json={"content": "secret", "extract": False}).json()["id"]
    assert client.delete(f"{P}/{mid}").status_code == 204
    assert client.get(f"{P}/{mid}").status_code == 404
    assert client.delete(f"{P}/{mid}").status_code == 404
    assert client.get(P).json() == []
    assert client.post(f"{P}/search", json={"q": "secret"}).json()["total"] == 0


def test_purge_removes_old_soft_deleted_rows(client, db):
    mid = client.post(P, json={"content": "old", "extract": False}).json()["id"]
    client.delete(f"{P}/{mid}")
    db.execute("UPDATE memory_entries SET deleted_at = now() - interval '45 days'")
    assert client.app_state.service.purge_deleted(30) == 1
    assert db.execute("SELECT count(*) AS n FROM memory_entries").fetchone()["n"] == 0


def test_delete_all_requires_confirmation(client):
    for i in range(3):
        client.post(P, json={"content": f"m{i}", "extract": False})
    assert client.delete(f"{P}/all").status_code == 428
    assert len(client.get(P).json()) == 3
    assert client.delete(f"{P}/all?confirm=true").json() == {"deleted": 3}
    assert client.get(P).json() == []


def test_export_contains_everything_and_nothing_deleted(client):
    a = client.post(P, json={"content": "keep me", "extract": False}).json()["id"]
    b = client.post(P, json={"content": "drop me", "extract": False}).json()["id"]
    client.delete(f"{P}/{b}")
    data = client.post(f"{P}/export").json()
    assert data["count"] == 1 and data["memories"][0]["id"] == a and data["memories"][0]["content"] == "keep me"


def test_list_is_newest_first_and_paged(client):
    ids = [client.post(P, json={"content": f"m{i}", "extract": False}).json()["id"] for i in range(5)]
    got = [m["id"] for m in client.get(f"{P}?limit=3").json()]
    assert got == ids[::-1][:3]
    assert [m["id"] for m in client.get(f"{P}?limit=3&offset=3").json()] == ids[::-1][3:]
    assert client.get(f"{P}?source=voice").json() == []


def test_one_memory_by_bad_id_is_a_clean_error(client):
    assert client.get(f"{P}/not-a-uuid").status_code == 422
    assert client.get(f"{P}/00000000-0000-0000-0000-000000000000").status_code == 404


def test_other_owners_data_is_invisible(client, add):
    import uuid

    stranger = add("stranger's private note", owner=uuid.uuid4())
    assert client.get(f"{P}/{stranger}").status_code == 404
    assert client.patch(f"{P}/{stranger}", json={"title": "x"}).status_code == 404
    assert client.delete(f"{P}/{stranger}").status_code == 404
    assert client.get(P).json() == []
    assert client.post(f"{P}/search", json={"q": "stranger"}).json()["total"] == 0
    assert client.delete(f"{P}/all?confirm=true").json() == {"deleted": 0}  # and delete-all cannot touch them either


def test_sql_injection_text_is_just_text(client):
    nasty = "'); DROP TABLE memory_entries; --"
    m = client.post(P, json={"content": nasty, "extract": False, "tags": [nasty]}).json()
    assert m["content"] == nasty
    assert client.post(f"{P}/search", json={"q": nasty, "project": nasty, "person": nasty, "tag": nasty}).status_code == 200
    assert len(client.get(P).json()) == 1  # the table is still there
