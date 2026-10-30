from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from app import ask
from app.config import Settings

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 1, 7, 15, 0, tzinfo=IST)     # a Wednesday, 15:00 IST
P = "/api/jarvis/memory"


def test_time_phrases_become_ranges():
    s, e, label = ask.parse_range("what did I do yesterday this time?", NOW)
    assert s == NOW - timedelta(days=1, hours=1, minutes=30) and e == NOW - timedelta(days=1) + timedelta(hours=1, minutes=30)
    s, e, _ = ask.parse_range("what did i do yesterday", NOW)
    assert s == datetime(2026, 1, 6, tzinfo=IST) and e == datetime(2026, 1, 7, tzinfo=IST)
    s, e, _ = ask.parse_range("anything 3 hours ago?", NOW)
    assert s < NOW - timedelta(hours=3) < e
    s, e, _ = ask.parse_range("last week", NOW)
    assert s == datetime(2025, 12, 29, tzinfo=IST) and e == datetime(2026, 1, 5, tzinfo=IST)
    assert ask.parse_range("what is pending for the Ravi project", NOW) is None


def test_written_out_dates_become_that_day():
    want = (datetime(2026, 10, 28, tzinfo=IST), datetime(2026, 10, 29, tzinfo=IST))
    for q in ("what happened on 28/10/2026", "what did I do on 28-10-2026?", "2026-10-28", "what happened on 28th of October 2026", "oct 28 2026"):
        s, e, label = ask.parse_range(q, NOW)
        assert (s, e) == want, q
    assert "28 October 2026" in label
    assert ask.parse_range("what happened on 31/02/2026", NOW) is None          # not a real day
    assert ask.parse_range("what is pending for the Ravi project", NOW) is None


def test_a_time_of_day_narrows_the_day():
    s, e, label = ask.parse_range("what did I do at 3 PM on 28/10/2026", NOW)
    assert (s, e) == (datetime(2026, 10, 28, 14, tzinfo=IST), datetime(2026, 10, 28, 16, tzinfo=IST)) and "15:00" in label
    s, e, _ = ask.parse_range("what happened 28/10/2026 at 09:30", NOW)
    assert (s, e) == (datetime(2026, 10, 28, 8, 30, tzinfo=IST), datetime(2026, 10, 28, 10, 30, tzinfo=IST))
    s, e, _ = ask.parse_range("yesterday afternoon", NOW)
    assert (s, e) == (datetime(2026, 1, 6, 12, tzinfo=IST), datetime(2026, 1, 6, 17, tzinfo=IST))
    s, e, _ = ask.parse_range("what happened on 28/10/2026", NOW)                 # no time given: the whole day
    assert (s, e) == (datetime(2026, 10, 28, tzinfo=IST), datetime(2026, 10, 29, tzinfo=IST))


def test_answers_only_from_the_memories_in_the_time_window(client, add):
    add("Fixed the login bug with Ravi", created_at=datetime(2026, 1, 6, 15, 10, tzinfo=IST))
    add("Wrote the budget report", created_at=datetime(2026, 1, 3, 11, 0, tzinfo=IST))     # too old: must not appear
    svc = client.app_state.service
    out = ask.answer(svc, "what did I do yesterday this time?", now=NOW)           # no Groq key in this call path
    svc.settings = Settings(**{**svc.settings.__dict__, "groq_api_key": ""})
    out = ask.answer(svc, "what did I do yesterday this time?", now=NOW)
    assert "login bug" in out["answer"] and "budget" not in out["answer"] and len(out["sources"]) == 1
    assert ask.answer(svc, "what did i do 2 days ago at midnight?", now=NOW)["sources"] == []


def test_groq_answer_is_used_and_falls_back_when_it_fails(client, add):
    add("Reviewed the invoice with Ravi", created_at=datetime(2026, 1, 7, 9, 0, tzinfo=IST))
    svc = client.app_state.service
    seen = {}

    def ok(request):
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "This morning you reviewed the invoice with Ravi."}}]})

    good = httpx.Client(transport=httpx.MockTransport(ok))
    out = ask.answer(svc, "what did I do today", client=good, now=NOW)
    assert out["answer"].startswith("This morning") and "invoice" in seen["body"] and "ONLY" in seen["body"]
    bad = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert "invoice" in ask.answer(svc, "what did I do today", client=bad, now=NOW)["answer"]


def test_ask_endpoint_requires_the_key_and_a_question(client, make_client):
    assert client.post(f"{P}/ask", json={"question": "x"}).status_code == 422
    assert make_client().post(f"{P}/ask", headers={"X-Owner-Key": ""}, json={"question": "what did I do today"}).status_code == 401
    r = client.post(f"{P}/ask", json={"question": "what did I do today"})
    assert r.status_code == 200 and "answer" in r.json()
