from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app import jobs, review
from tests.test_email_reminders import FakeMailbox, FakePusher, TOKEN, enable

P = "/api/jarvis/memory"
IST = ZoneInfo("Asia/Kolkata")
DAY = date(2026, 1, 5)
EVENING = datetime(2026, 1, 5, 21, 40, tzinfo=IST)       # after the 21:30 review time, before quiet hours


def seed(add, client):
    add("Fixed the login bug with Ravi", summary="Fixed the login bug.", created_at=datetime(2026, 1, 5, 11, 0, tzinfo=IST), source="voice")
    add("Reviewed the budget", summary="Budget reviewed.", created_at=datetime(2026, 1, 5, 15, 0, tzinfo=IST), source="email",
        tasks=[{"task": "Send the invoice", "status": "done"}, {"task": "Call the bank", "status": "pending"}])
    add("Yesterday's note", created_at=datetime(2026, 1, 4, 12, 0, tzinfo=IST))
    client.post(f"{P}/reminders", json={"text": "Dentist at 10", "remind_at": datetime(2026, 1, 6, 10, 0, tzinfo=IST).isoformat()})


def test_the_review_is_built_only_from_that_days_data(client, add):
    seed(add, client)
    data = review.gather(client.app_state.service, DAY)
    assert [m["summary"] for m in data["memories"]] == ["Fixed the login bug.", "Budget reviewed."]
    assert data["tasks_open"] == ["Call the bank"] and data["reminders_tomorrow"][0]["text"] == "Dentist at 10"
    built = review.build(client.app_state.service, DAY)
    assert built["stats"]["memories"] == 2 and "2 updates" in built["summary"] and "Call the bank" in built["summary"]


def test_nightly_tick_builds_pushes_once_and_respects_the_time_window(client, add):
    seed(add, client)
    client.patch(f"{P}/settings", json={"push_enabled": True})
    client.app_state.service.register_device(TOKEN)
    svc, pusher = client.app_state.service, FakePusher()
    early = datetime(2026, 1, 5, 20, 0, tzinfo=IST)
    assert jobs.send_review(svc, pusher, None, early) == {"why": "not_yet"}
    out = jobs.send_review(svc, pusher, None, EVENING)
    assert out["generated"] is True and out["push"] is True
    sent = pusher.sent[-1][1]
    assert sent["type"] == "review" and "2 updates" in sent["title"] and sent["line"]
    assert jobs.send_review(svc, pusher, None, EVENING + timedelta(minutes=5)).get("generated") is None   # once a day
    assert len([p for p in pusher.sent if p[1]["type"] == "review"]) == 1


def test_a_missed_evening_is_not_sent_at_night(client, add):
    seed(add, client)
    late = datetime(2026, 1, 5, 23, 15, tzinfo=IST)
    assert jobs.send_review(client.app_state.service, FakePusher(), None, late) == {"why": "quiet_hours"}


def test_review_email_is_handed_to_the_scheduler_then_confirmed(make_client, add):
    c = make_client(smtp_mode="external")
    enable(c)
    seed(add, c)
    out = jobs.send_review(c.app_state.service, None, None, EVENING)
    deliver = out["email"]["deliver"]
    assert deliver["to"] == "owner@example.com" and "Daily Review" in deliver["subject"] and "<table" in deliver["html"]
    assert "2 updates" in deliver["text"] and "Call the bank" in deliver["text"]
    again = jobs.send_review(c.app_state.service, None, None, EVENING + timedelta(minutes=5))
    assert again["email"]["why"] == "external_delivery"                       # still pending until confirmed
    assert c.post(f"{P}/mail/review-sent", params={"day": "2026-01-05"}).status_code == 200
    assert "email" not in jobs.send_review(c.app_state.service, None, None, EVENING + timedelta(minutes=10))


def test_direct_mode_sends_the_review_email_itself(client, add):
    enable(client)
    seed(add, client)
    box = FakeMailbox()
    out = jobs.send_review(client.app_state.service, None, box, EVENING)
    assert out["email"] == {"sent": True} and box.sent[0][1].startswith("🌙 Your Daily Review")


def test_review_can_be_switched_off(client):
    client.patch(f"{P}/settings", json={"review_enabled": False})
    assert jobs.send_review(client.app_state.service, None, None, EVENING) == {"why": "disabled"}
    assert client.patch(f"{P}/settings", json={"review_hour": 24}).status_code == 422


def test_calendar_and_day_detail_endpoints(client, add):
    seed(add, client)
    client.post(f"{P}/reviews/2026-01-05/generate")
    month = client.get(f"{P}/reviews", params={"month": "2026-01"}).json()
    days = {d["day"]: d for d in month["days"]}
    assert days["2026-01-05"]["memories"] == 2 and "2 updates" in days["2026-01-05"]["summary"]
    assert days["2026-01-04"]["memories"] == 1 and days["2026-01-04"]["summary"] is None
    detail = client.get(f"{P}/reviews/2026-01-05").json()
    assert detail["review"]["stats"]["tasks_open"] == 1 and len(detail["memories"]) == 2
    assert client.get(f"{P}/reviews", params={"month": "2026-13"}).status_code == 422


def test_review_html_escapes_everything():
    fake = {"summary": "<script>x</script> done", "stats": {"memories": 1, "tasks_done": 0, "tasks_open": 1, "reminders_today": 0, "reminders_tomorrow": 0},
            "details": {"tasks_done": [], "tasks_open": ["<img src=x onerror=1>"], "reminders_tomorrow": []}}
    _subject, _text, page = review.render_email(fake, "Madhavan", DAY)
    assert "<script>" not in page and "&lt;script&gt;" in page and "<img src=x" not in page


def test_review_routes_need_the_owner_key(make_client):
    c = make_client()
    assert c.get(f"{P}/reviews", headers={"X-Owner-Key": ""}).status_code == 401
    assert c.post(f"{P}/reviews/2026-01-05/generate", headers={"X-Owner-Key": ""}).status_code == 401
