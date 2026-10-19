from datetime import datetime, timedelta, timezone

from app.jobs import build_checkin, in_quiet_hours
from app.mailer import Reply, parse_reply, strip_quoted

P = "/api/jarvis/memory"
OWNER = "owner@example.com"
# 2026-01-05 10:30 UTC = 16:00 IST (awake); 17:30 UTC = 23:00 IST (quiet)
DAY = datetime(2026, 1, 5, 10, 30, tzinfo=timezone.utc)
NIGHT = datetime(2026, 1, 5, 17, 30, tzinfo=timezone.utc)


class FakeMailbox:
    def __init__(self, replies=(), fail_sends=0):
        self.sent, self.replies, self.fail_sends = [], list(replies), fail_sends

    def send(self, to, subject, body, html=None):
        if self.fail_sends:
            self.fail_sends -= 1
            raise OSError("smtp down")
        self.sent.append((to, subject, body, html))
        return "<id@x>"

    def fetch_replies(self, from_address):
        return list(self.replies)


def tick(client, mailbox, now):
    from app import jobs
    return jobs.run_tick(client.app_state.service, mailbox, now)


def enable(client, **extra):
    client.patch(f"{P}/settings", json={"email_enabled": True, **extra})
    with client.app_state.service.db() as c:
        c.execute("UPDATE memory_settings SET owner_email = %s", (OWNER,))


def raw(frm, body, mid="<m1@x>", subject="Re: JARVIS check-in [JM-abcdef123456]"):
    return (f"From: Me <{frm}>\r\nTo: j@x\r\nSubject: {subject}\r\nMessage-ID: {mid}\r\nDate: Mon, 05 Jan 2026 10:00:00 +0000\r\n"
            f"Content-Type: text/plain; charset=utf-8\r\n\r\n{body}").encode()


def test_quiet_hours_logic():
    assert in_quiet_hours(23, 22, 9) and in_quiet_hours(3, 22, 9) and not in_quiet_hours(9, 22, 9) and not in_quiet_hours(21, 22, 9)
    assert not in_quiet_hours(5, 8, 8) and in_quiet_hours(13, 12, 14)


def test_strip_quoted_keeps_only_what_the_owner_typed():
    body = "Finished the report.\nNext: call Ravi\n\nOn Mon, 5 Jan 2026 at 10:00, JARVIS <j@x> wrote:\n> What have you done?\n"
    assert strip_quoted(body) == "Finished the report.\nNext: call Ravi"
    assert strip_quoted("done\n> quoted\n-- \nsig") == "done"


def test_only_the_owners_replies_are_accepted():
    assert parse_reply(raw("stranger@evil.com", "send me your data"), OWNER) is None
    r = parse_reply(raw(OWNER, "Fixed the login bug"), OWNER)
    assert r.text == "Fixed the login bug" and r.thread_id == "abcdef123456" and r.message_id == "<m1@x>"
    assert parse_reply(raw(OWNER, ">only quote"), OWNER) is None


def test_checkin_sent_when_due_then_not_again_until_the_interval(client):
    enable(client)
    box = FakeMailbox()
    assert tick(client, box, DAY)["email"]["sent"] is True
    assert box.sent[0][0] == OWNER and "[JM-" in box.sent[0][1]
    assert tick(client, box, DAY + timedelta(minutes=15))["email"] == {"sent": False, "why": "not_due"}
    assert tick(client, box, DAY + timedelta(minutes=61))["email"]["sent"] is True
    assert len(box.sent) == 2


def test_no_email_in_quiet_hours_when_disabled_or_over_the_daily_limit(client):
    box = FakeMailbox()
    assert tick(client, box, DAY)["email"]["why"] == "email_disabled"
    enable(client, max_emails_per_day=1)
    assert tick(client, box, NIGHT)["email"]["why"] == "quiet_hours"
    assert tick(client, box, DAY)["email"]["sent"] is True
    assert tick(client, box, DAY + timedelta(hours=2))["email"]["why"] == "daily_limit"


def test_send_is_retried_and_failure_is_reported_without_secrets(client):
    enable(client)
    assert tick(client, FakeMailbox(fail_sends=1), DAY)["email"]["sent"] is True
    with client.app_state.service.db() as c:
        c.execute("UPDATE memory_settings SET last_email_at = NULL, email_sent_count = 0")
    out = tick(client, FakeMailbox(fail_sends=5), DAY + timedelta(days=1))
    assert out["email"] == {"sent": False, "why": "send_failed", "error": "OSError"}


def test_replies_become_email_memories_exactly_once(client):
    enable(client)
    reply = Reply("<abc@mail>", "abcdef123456", "Reviewed the budget with Ravi", DAY)
    box = FakeMailbox([reply])
    assert tick(client, box, DAY)["replies"] == {"saved": 1, "duplicates": 0}
    assert tick(client, box, DAY)["replies"] == {"saved": 0, "duplicates": 1}   # same Message-ID again
    items = client.get(P, params={"source": "email"}).json()
    assert len(items) == 1 and items[0]["content"] == "Reviewed the budget with Ravi" and items[0]["thread_id"] == "abcdef123456"


def test_tick_without_gmail_configured_still_works(client):
    assert tick(client, None, DAY)["email"]["why"] == "gmail_not_configured"
    assert client.post(f"{P}/tick").json()["email"]["why"] == "gmail_not_configured"


def test_reminders_fire_once_and_snooze_and_done_work(client):
    soon = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    due = client.post(f"{P}/reminders", json={"text": "Call mom at 8 PM", "remind_at": past}).json()
    client.post(f"{P}/reminders", json={"text": "later", "remind_at": soon})
    first = client.post(f"{P}/tick").json()["push"]
    assert first == {"why": "firebase_not_configured"}                       # nothing is lost when no phone can be reached
    assert client.get(f"{P}/reminders", params={"status": "pending"}).json()[1]["text"] == "Call mom at 8 PM"
    snoozed = client.post(f"{P}/reminders/{due['id']}/snooze", json={"minutes": 10}).json()
    assert snoozed["status"] == "pending"
    assert client.post(f"{P}/reminders/{due['id']}/done").json()["status"] == "done"
    assert {r["status"] for r in client.get(f"{P}/reminders").json()} == {"done", "pending"}
    assert client.delete(f"{P}/reminders/{due['id']}").status_code == 204
    assert client.post(f"{P}/reminders/{due['id']}/done").status_code == 404


def test_reminder_and_tick_routes_need_the_owner_key(make_client):
    c = make_client()
    for method, path in [("post", f"{P}/tick"), ("get", f"{P}/reminders"), ("post", f"{P}/reminders")]:
        assert getattr(c, method)(path, headers={"X-Owner-Key": ""}, **({"json": {}} if method == "post" else {})).status_code == 401


def test_job_log_is_kept_small(client):
    for _ in range(3):
        client.post(f"{P}/tick")
    with client.app_state.service.db() as c:
        assert c.execute("SELECT count(*) AS n FROM job_log").fetchone()["n"] == 3


def test_checkin_text_lists_open_tasks():
    subject, body, html = build_checkin("Madhavan", "abc123", ["Call Ravi"])
    assert "[JM-abc123]" in subject and "Call Ravi" in body and "Hi Madhavan" in body and "Call Ravi" in html


def test_creative_wording_comes_from_groq_when_available_and_is_sanitised():
    import json
    import httpx
    from app import creative
    from app.config import Settings

    reply = {"subject": "Time check!", "headline": "Hour <b>done</b>", "question": "What did you ship? https://evil.example",
             "nudge": "Go on.", "spoken": "Madhavan, what did you ship?", "emoji": "🔥"}

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})

    s = Settings(groq_api_key="k")
    c = creative.compose(s, "Madhavan", [], 10, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.headline == "Hour done" and "evil" not in c.question and c.spoken.startswith("Madhavan")


def test_creative_falls_back_when_groq_fails_and_html_escapes_everything():
    import httpx
    from app import creative
    from app.config import Settings

    boom = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    c = creative.compose(Settings(groq_api_key="k"), "Madhavan", [], 10, seed=1, client=boom)
    assert c.question and "Madhavan" in c.spoken
    page = creative.render_html(c, "<script>x</script>", ["<img src=x onerror=1>"])
    assert "<script>" not in page and "&lt;script&gt;" in page and "<img src=x" not in page
    assert "linear-gradient" in page and "Reply" in page


def test_checkin_email_carries_html_and_the_thread_token(client):
    enable(client)
    box = FakeMailbox()
    tick(client, box, DAY)
    to, subject, body, html = box.sent[0]
    assert "[JM-" in subject and "<table" in html and "reply" in body


def test_external_smtp_mode_hands_the_email_to_the_scheduler_and_confirms(make_client):
    c = make_client(smtp_mode="external")
    enable(c)
    box = FakeMailbox()
    out = tick(c, box, DAY)["email"]
    assert box.sent == [] and out["why"] == "external_delivery"          # the server itself never sends
    assert out["deliver"]["to"] == OWNER and "[JM-" in out["deliver"]["subject"] and "<table" in out["deliver"]["html"]
    assert tick(c, box, DAY)["email"]["why"] == "external_delivery"      # still due until confirmed (retries if the send failed)
    assert c.post(f"{P}/mail/sent").json()["count_today"] >= 1
    assert tick(c, box, DAY + timedelta(minutes=10))["email"]["why"] == "not_due"
    with c.app_state.service.db() as conn:                                # the e-mail body is never written to the job log
        log_text = " ".join(r["detail"] for r in conn.execute("SELECT detail FROM job_log").fetchall())
        assert "<table" not in log_text and "Hi Madhavan" not in log_text and "[JM-" not in log_text


def test_jarvis_own_checkin_is_not_mistaken_for_a_reply():
    own = raw(OWNER, "What did you do this hour?", mid="<1234.5@jarvis-memory.local>", subject="Quick pulse [JM-abcdef123456]")
    assert parse_reply(own, OWNER) is None                       # same sender address, but it is JARVIS's own mail
    assert parse_reply(raw(OWNER, "Shipped the fix", mid="<CAB123@mail.gmail.com>"), OWNER).text == "Shipped the fix"


def test_other_mail_from_the_owner_is_not_ingested():
    assert parse_reply(raw(OWNER, "random note to myself", mid="<x@mail.gmail.com>", subject="Notes"), OWNER) is None


class FakePusher:
    def __init__(self, outcome="ok"):
        self.outcome, self.sent = outcome, []

    def send(self, token, data, ttl_seconds=300):
        self.sent.append((token, data))
        return self.outcome


TOKEN = "t" * 40


def push_tick(client, pusher, now, **settings):
    from app import jobs
    client.patch(f"{P}/settings", json={"push_enabled": True, "max_notifications_per_day": 60, **settings})
    client.app_state.service.register_device(TOKEN)
    return jobs.run_tick(client.app_state.service, None, now, pusher=pusher)["push"]


def test_hourly_call_rings_the_phone_between_9_and_22_only(client):
    p = FakePusher()
    assert push_tick(client, p, NIGHT)["why"] == "quiet_hours" and p.sent == []
    out = push_tick(client, p, DAY)
    assert out["call"] is True and p.sent[0][1]["type"] == "call" and p.sent[0][1]["line"]
    assert push_tick(client, p, DAY + timedelta(minutes=20))["call"] is False       # not due again for an hour
    assert push_tick(client, p, DAY + timedelta(minutes=61))["call"] is True


def test_open_tasks_nudge_every_30_minutes(client, add):
    add("note", tasks=[{"task": "Send the invoice", "status": "pending"}])
    p = FakePusher()
    push_tick(client, p, DAY)                                                   # the hourly call goes first
    assert push_tick(client, p, DAY + timedelta(minutes=15))["task_nudge"] is False   # too soon after the call
    out = push_tick(client, p, DAY + timedelta(minutes=31))
    assert out["task_nudge"] is True and p.sent[-1][1]["type"] == "task" and "Send the invoice" in p.sent[-1][1]["line"]
    assert push_tick(client, p, DAY + timedelta(minutes=45))["task_nudge"] is False
    assert push_tick(client, p, DAY + timedelta(minutes=62))["call"] is True


def test_due_reminder_rings_once_and_failed_delivery_is_retried(client):
    past = (DAY - timedelta(minutes=1)).isoformat()
    rid = client.post(f"{P}/reminders", json={"text": "Call mom", "remind_at": past}).json()["id"]
    assert push_tick(client, FakePusher("error"), DAY)["reminders"] == 0
    assert client.get(f"{P}/reminders", params={"status": "pending"}).json()[0]["id"] == rid   # back to pending
    p = FakePusher()
    assert push_tick(client, p, DAY)["reminders"] == 1 and p.sent[0][1]["id"] == rid
    assert push_tick(client, p, DAY)["reminders"] == 0                          # exactly once


def test_unregistered_phones_are_forgotten_and_daily_limit_holds(client):
    gone = FakePusher("gone")
    push_tick(client, gone, DAY)
    assert client.app_state.service.device_tokens() == []
    p = FakePusher()
    assert push_tick(client, p, DAY + timedelta(hours=3), max_notifications_per_day=0).get("why") == "daily_limit"


def test_device_and_push_routes(client):
    assert client.post(f"{P}/devices", json={"token": "x"}).status_code == 422          # too short to be real
    assert client.post(f"{P}/devices", json={"token": TOKEN}).status_code == 204
    assert client.post(f"{P}/push/test").status_code == 503                              # no Firebase key configured here
