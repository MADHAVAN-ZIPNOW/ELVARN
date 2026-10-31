"""The daily team update: JARVIS calls, the owner says what they did, and only that is e-mailed to the team after a yes.
No database needed: a tiny in-memory stand-in plays the chat tables, and the mailbox, phone and AI are fakes."""
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx

from app import dayreport, jobs
from app.chat import ChatService
from app.config import Settings

CONTACTS = "Ravi:ravi@example.com, Priya Nair:priya@example.com, broken:not-an-address"
IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 10, 29, 18, 5, tzinfo=IST)
OPENING = "Madhavan, it is time for your team update. What did you get done today?"


class Script:
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages):
        self.seen.append(messages)
        r = self.replies.pop(0)
        return r if isinstance(r, str) else json.dumps(r)


class FakeMailbox:
    def __init__(self, fail_for=()):
        self.sent, self.fail_for = [], set(fail_for)

    def send(self, to, subject, body, html=None):
        if to in self.fail_for:
            raise OSError("refused")
        self.sent.append({"to": to, "subject": subject, "body": body, "html": html})
        return "<id@jarvis-memory.local>"


class FakeDB:
    """Just enough of the chat tables for ChatService: sessions (id, title) and messages (session, role, content)."""

    def __init__(self):
        self.sessions, self.messages = {}, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        sql = " ".join(sql.split())
        rows = []
        if sql.startswith("INSERT INTO chat_sessions"):
            sid = f"s{len(self.sessions) + 1}"
            self.sessions[sid] = params[1]
            rows = [{"id": sid}]
        elif sql.startswith("SELECT 1 FROM chat_sessions"):
            rows = [{"x": 1}] if params[0] in self.sessions else []
        elif sql.startswith("SELECT title FROM chat_sessions"):
            rows = [{"title": self.sessions[params[0]]}]
        elif sql.startswith("INSERT INTO chat_messages"):
            self.messages.append((params[0], params[1], params[2]))
        elif sql.startswith("UPDATE chat_sessions SET title"):
            self.sessions[params[1]] = params[0]
        elif sql.startswith("SELECT role, content FROM chat_messages"):
            rows = [{"role": r, "content": c} for s, r, c in self.messages if s == params[0]][::-1]
        elif sql.startswith("SELECT content FROM chat_messages"):
            rows = [{"content": c} for s, r, c in self.messages if s == params[0] and r == "user"]
        return SimpleNamespace(fetchone=lambda: rows[0] if rows else None, fetchall=lambda: rows)


def make_chat(*replies, mailbox=None, contacts=CONTACTS):
    db = FakeDB()

    def no_personal_memory(*a, **k):
        raise AssertionError("the team update call must never write a personal memory")

    svc = SimpleNamespace(settings=Settings(owner_name="Madhavan", email_contacts=contacts, groq_api_key=""), owner_id="owner", db=lambda: db,
                          create=no_personal_memory)
    script = Script(*replies)
    return ChatService(svc, llm=script, mailbox=mailbox or FakeMailbox()), script, db


def say(text):
    return {"action": "answer", "text": text}


# ---------------------------------------------------------------------------------------------- contacts, times, wording
def test_contacts_and_call_time_parsing():
    contacts = dayreport.parse_contacts(CONTACTS)
    assert set(contacts) == {"ravi", "priya nair"}                       # the entry without a real address is ignored
    assert dayreport.parse_clock("17:30") == (17, 30) and dayreport.parse_clock("garbage") == (18, 0) and dayreport.parse_clock("25:00") == (18, 0)


def test_the_ai_sorts_what_was_said_and_a_bad_answer_falls_back_to_the_plain_text():
    settings = Settings(owner_name="Madhavan", groq_api_key="k")
    good = {"summary": "Madhavan finished the login screen; the payment page is in progress.",
            "items": [{"kind": "done", "text": "Login screen"}, {"kind": "progress", "text": "Payment page"}, {"kind": "weird", "text": "Release Friday"}]}

    def answer(content):
        return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": content}}]})))

    summary, items = dayreport.compose_team(settings, ["I finished the login screen", "payment page is in progress"], answer(json.dumps(good)))
    assert summary.startswith("Madhavan finished") and [i["kind"] for i in items] == ["done", "progress", "update"]     # unknown kind -> update
    summary, items = dayreport.compose_team(settings, ["I finished the login screen"], answer("not json"))
    assert summary == "I finished the login screen" and items == [{"kind": "update", "text": "I finished the login screen"}]
    assert dayreport.compose_team(Settings(groq_api_key=""), ["only text"])[1][0]["kind"] == "update"


def test_email_has_summary_table_chart_and_escapes_html():
    items = [{"kind": "done", "text": "Login <b>screen</b>"}, {"kind": "progress", "text": "Payment page"}, {"kind": "done", "text": "Invoice sent"}]
    subject, plain, page = dayreport.render_team("Madhavan finished two things.", items, "Madhavan", "Ravi", date(2026, 10, 29), "Talk tomorrow")
    assert subject == "Madhavan's team update: Thu 29 Oct" and plain.startswith("Dear Ravi,")
    assert "Done | Login <b>screen</b>" in plain and "from what Madhavan said on today's update call" in plain
    assert "&lt;b&gt;screen" in page and "<b>screen</b>" not in page and "TODAY AT A GLANCE" in page and "UPDATE BREAKDOWN" in page and "Talk tomorrow" in page


# ---------------------------------------------------------------------------------------------- the call
def test_the_call_is_made_once_a_day_after_the_chosen_time(monkeypatch):
    pushed, logged = [], []

    class Pusher:
        def send(self, token, data):
            pushed.append(data)
            return "ok"

    svc = SimpleNamespace(settings=Settings(email_contacts=CONTACTS, team_call_time="18:00"), device_tokens=lambda: ["t1"],
                          get_settings=lambda: {"push_enabled": True, "timezone": "Asia/Kolkata", "quiet_start_hour": 22, "quiet_end_hour": 3, "owner_name": "Madhavan"},
                          record_push=lambda now, call=False: 1, forget_device=lambda t: None)
    monkeypatch.setattr(jobs, "_team_call_done", lambda s, day: bool(logged))
    monkeypatch.setattr(jobs, "_log", lambda s, job, ok, detail: logged.append((job, detail)))
    at = lambda h, m: datetime(2026, 10, 29, h, m, tzinfo=IST)
    assert jobs.send_team_call(svc, Pusher(), at(17, 59))["why"] == "not_yet" and pushed == []
    assert jobs.send_team_call(svc, Pusher(), at(18, 0)) == {"sent": True}
    assert pushed[0]["type"] == "team" and "your team update" in pushed[0]["line"]
    assert logged == [("team_call", "2026-10-29")]
    assert jobs.send_team_call(svc, Pusher(), at(18, 30))["why"] == "done_today" and len(pushed) == 1
    logged.clear()
    assert jobs.send_team_call(svc, Pusher(), at(22, 30))["why"] == "quiet_hours"
    off = SimpleNamespace(**{**svc.__dict__, "settings": Settings(email_contacts="")})
    assert jobs.send_team_call(off, Pusher(), at(18, 0))["why"] == "no_team_contacts"        # no team set up: no call


# ---------------------------------------------------------------------------------------------- the conversation
def test_team_call_keeps_to_the_call_reads_back_and_sends_only_after_a_yes():
    box = FakeMailbox()
    chat, script, db = make_chat(say("Nice. What is in progress or blocked?"), say("Good. What is next?"),
                                 say("Summary: login done, payment page in progress, release Friday. Send it to Ravi and Priya Nair?"),
                                 {"action": "send_team_update", "confirmed": True}, say("Sent. Bye!"), mailbox=box)
    sid = chat.chat("I finished the login screen", context=OPENING, now=NOW, client="phone")["session_id"]
    assert db.sessions[sid] == "Team update 29 Oct"
    assert "TEAM UPDATE" in script.seen[0][0]["content"] and "Ravi and Priya Nair" in script.seen[0][0]["content"]
    chat.chat("The payment page is in progress", sid, now=NOW)                       # later turns have no context: the session says it is a team call
    chat.chat("Next is the release on Friday, and that is all", sid, now=NOW)
    assert box.sent == []                                                            # nothing goes out before the yes
    out = chat.chat("yes, send it", sid, now=NOW)
    assert out["actions"][0]["tool"] == "send_team_update" and [m["to"] for m in box.sent] == ["ravi@example.com", "priya@example.com"]
    body = box.sent[0]["body"]
    assert "Dear Ravi," in body and "Super Intelligence assistant" in body and "I finished the login screen" in body and "release on Friday" in body
    assert "yes, send it" not in body                                                # the confirmation is not part of the update
    assert db.sessions[sid].endswith("(sent)")
    assert "NOT SENT" not in out["actions"][0]["summary"]


def test_a_confirmed_flag_without_the_owners_yes_sends_nothing():
    box = FakeMailbox()
    chat, _, db = make_chat(say("Nice. Anything blocked?"), {"action": "send_team_update", "confirmed": True}, say("Shall I send it?"), mailbox=box)
    sid = chat.chat("I finished the login screen", context=OPENING, now=NOW)["session_id"]
    out = chat.chat("I am still working on the payment page", sid, now=NOW)         # the model jumps ahead: no yes from the owner
    assert box.sent == [] and out["actions"][0]["summary"] == "asked to confirm the team update"


def test_only_work_said_on_the_call_is_sent_and_a_second_yes_does_not_send_twice():
    box = FakeMailbox()
    chat, _, db = make_chat(say("Nice."), {"action": "send_team_update", "confirmed": True}, say("Sent."),
                            {"action": "send_team_update", "confirmed": True}, say("Already sent."), mailbox=box)
    sid = chat.chat("Finished the invoice for the client", context=OPENING, now=NOW)["session_id"]
    chat.chat("yes", sid, now=NOW)
    assert len(box.sent) == 2
    chat.chat("yes", sid, now=NOW)
    assert len(box.sent) == 2                                                        # still two: no duplicate


def test_nothing_but_sending_can_run_on_the_team_call_and_sending_is_only_for_it():
    chat, _, db = make_chat({"action": "search_memory", "question": "my salary"}, say("I can only take your team update."))
    out = chat.chat("what do you remember about my salary", context=OPENING, now=NOW)
    assert out["actions"] == [{"tool": "search_memory", "summary": "not available on this chat"}]       # personal memory stays out of reach
    ordinary, _, _ = make_chat({"action": "send_team_update", "confirmed": True}, say("I cannot do that here."))
    out = ordinary.chat("yes send the team update to everyone", now=NOW)
    assert out["actions"] == [{"tool": "send_team_update", "summary": "not available on this chat"}]


def test_a_failed_send_is_reported_honestly():
    chat, _, db = make_chat(say("Nice."), {"action": "send_team_update", "confirmed": True}, say("One failed."), mailbox=FakeMailbox(fail_for={"priya@example.com"}))
    sid = chat.chat("Finished the invoice for the client", context=OPENING, now=NOW)["session_id"]
    out = chat.chat("yes", sid, now=NOW)
    assert out["actions"][0]["summary"] == "team update sent to 1, failed 1" and not db.sessions[sid].endswith("(sent)")
