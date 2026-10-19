"""Full backup / restore / clean-up: nothing is lost on the way to the PC archive and back."""
import json
from datetime import datetime, timedelta, timezone

from app import backup, review
from tests.test_chat import Script, chat_with, say

P = "/api/jarvis/memory"


def populate(client, add):
    add("Fixed the login bug with Ravi", summary="Login bug fixed.", tags=["work"], tasks=[{"task": "Send invoice", "status": "pending"}])
    add("Reviewed the budget", source="email")
    chat = chat_with(client, Script(say("Hello Madhavan!"), say("You fixed the login bug.")))
    sid = chat.chat("hi jarvis")["session_id"]
    chat.chat("what did I do today", sid)
    client.post(f"{P}/reminders", json={"text": "Call mom", "remind_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()})
    client.app_state.service.claim_due_reminders(datetime.now(timezone.utc))                   # now 'sent'
    client.post(f"{P}/reminders", json={"text": "Later", "remind_at": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()})
    review.build(client.app_state.service, datetime.now(timezone.utc).date())
    return sid


def test_export_everything_includes_chats_reminders_and_reviews(client, add):
    populate(client, add)
    data = client.post(f"{P}/export/all").json()
    assert data["count"] == 2 and len(data["memories"]) == 2
    assert len(data["chat_sessions"]) == 1 and [m["role"] for m in data["chat_sessions"][0]["messages"]] == ["user", "assistant", "user", "assistant"]
    assert {r["text"]: r["status"] for r in data["reminders"]} == {"Call mom": "sent", "Later": "pending"}
    assert len(data["reviews"]) == 1 and data["reviews"][0]["summary"]


def test_restore_puts_everything_back_into_an_empty_cloud_exactly_once(client, add, db):
    populate(client, add)
    snapshot = client.post(f"{P}/export/all").json()
    for table in ("memory_entries", "chat_sessions", "reminders", "daily_reviews"):
        db.execute(f"TRUNCATE {table} CASCADE")                                               # a brand-new empty cloud
    body = {k: snapshot[k] for k in ("memories", "chat_sessions", "reminders", "reviews")}
    first = client.post(f"{P}/archive/restore", content=json.dumps(body), headers={"Content-Type": "application/json"})
    assert first.status_code == 200
    assert first.json()["added"] == {"memories": 2, "chat_sessions": 1, "chat_messages": 4, "reminders": 2, "reviews": 1}
    again = client.post(f"{P}/archive/restore", content=json.dumps(body), headers={"Content-Type": "application/json"}).json()["added"]
    assert set(again.values()) == {0}                                                          # nothing doubled
    after = client.post(f"{P}/export/all").json()
    assert {m["content"] for m in after["memories"]} == {m["content"] for m in snapshot["memories"]}
    assert after["memories"][0]["tasks"] or after["memories"][1]["tasks"]                      # the tasks inside memories came back too
    assert [m["content"] for m in after["chat_sessions"][0]["messages"]] == [m["content"] for m in snapshot["chat_sessions"][0]["messages"]]


def test_restore_never_overwrites_what_the_cloud_already_has(client, add):
    add("Original text")
    snapshot = client.post(f"{P}/export/all").json()
    snapshot["memories"][0]["content"] = "TAMPERED"
    client.post(f"{P}/archive/restore", json={"memories": snapshot["memories"]})
    assert client.get(P).json()[0]["content"] == "Original text"


def test_restore_validates_and_is_owner_scoped(client, add, owner_id, db):
    bad = client.post(f"{P}/archive/restore", json={"memories": [{"id": "not-a-uuid", "content": "x", "created_at": "2026-01-01T00:00:00Z"}]})
    assert bad.status_code == 422
    other = {"id": "11111111-1111-1111-1111-111111111111", "content": "claims another owner", "created_at": "2026-01-01T00:00:00Z",
             "owner_id": "99999999-9999-9999-9999-999999999999"}
    assert client.post(f"{P}/archive/restore", json={"memories": [other]}).json()["added"]["memories"] == 1
    row = db.execute("SELECT owner_id FROM memory_entries WHERE content = 'claims another owner'").fetchone()
    assert str(row["owner_id"]) == str(owner_id)                                               # the owner is always the server's owner


def test_prune_extra_removes_old_chats_finished_reminders_and_reviews_but_never_pending_ones(client, add):
    sid = populate(client, add)
    data = client.post(f"{P}/export/all").json()
    ids = [r["id"] for r in data["reminders"]]                                                 # includes the still-pending one
    out = client.post(f"{P}/archive/prune-extra", json={"chat_session_ids": [sid], "reminder_ids": ids, "review_days": [data["reviews"][0]["day"]]}).json()["deleted"]
    assert out == {"chat_sessions": 1, "reminders": 1, "reviews": 1}                           # the pending reminder stayed
    left = client.post(f"{P}/export/all").json()
    assert left["chat_sessions"] == [] and [r["text"] for r in left["reminders"]] == ["Later"] and left["reviews"] == []
    assert client.post(f"{P}/archive/prune-extra", json={"unknown": 1}).status_code == 422


def test_backup_routes_need_the_owner_key(make_client):
    c = make_client()
    for path in ("export/all", "archive/restore", "archive/prune-extra"):
        assert c.post(f"{P}/{path}", json={}, headers={"X-Owner-Key": ""}).status_code == 401
