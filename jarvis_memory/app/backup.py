"""Full backup, restore and clean-up for the PC archive.

The desktop keeps the long-term copy of EVERYTHING (memories, chats, reminders, daily reviews). The cloud keeps a recent working
set. These functions let the desktop (1) read everything, (2) put a copy back into an empty or new cloud (restore), and (3) remove old
rows from the cloud once they are safely on the PC. Everything is owner-scoped; restore never overwrites anything that exists.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from typing import Any, Optional

from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from .schemas import Source

EXPORT_CHATS = 3000       # most recent chats included in an export (each with all its messages)
EXPORT_REMINDERS = 5000
EXPORT_REVIEWS = 5000
RESTORE_MAX_ITEMS = 2000


def export_everything(svc) -> dict[str, Any]:
    """Memories + chats (with messages) + reminders + daily reviews, as plain JSON-able data."""
    base = svc.export()
    with svc.db() as conn:
        sessions = conn.execute("SELECT id, title, created_at, updated_at FROM chat_sessions WHERE owner_id = %s "
                                "ORDER BY updated_at DESC LIMIT %s", (svc.owner_id, EXPORT_CHATS)).fetchall()
        by_id = {s["id"]: {**s, "id": str(s["id"]), "messages": []} for s in sessions}
        if by_id:
            for m in conn.execute("SELECT session_id, role, content, actions, created_at FROM chat_messages WHERE session_id = ANY(%s) ORDER BY id",
                                  (list(by_id),)).fetchall():
                by_id[m["session_id"]]["messages"].append({"role": m["role"], "content": m["content"], "actions": m["actions"], "created_at": m["created_at"]})
        reminders = conn.execute("SELECT id, text, remind_at, status, created_at, sent_at FROM reminders WHERE owner_id = %s "
                                 "ORDER BY remind_at DESC LIMIT %s", (svc.owner_id, EXPORT_REMINDERS)).fetchall()
        reviews = conn.execute("SELECT day, summary, stats, details, emailed, created_at FROM daily_reviews WHERE owner_id = %s "
                               "ORDER BY day DESC LIMIT %s", (svc.owner_id, EXPORT_REVIEWS)).fetchall()
    base["chat_sessions"] = list(by_id.values())
    base["reminders"] = [{**r, "id": str(r["id"])} for r in reminders]
    base["reviews"] = [{**r, "day": r["day"].isoformat()} for r in reviews]
    return base


# ------------------------------------------------------------------------------------------------ restore
class RMemory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: uuid.UUID
    source: Source = "manual"
    title: str = Field("", max_length=200)
    content: str = Field(max_length=20000)
    raw_transcript: Optional[str] = Field(None, max_length=20000)
    summary: str = Field("", max_length=2000)
    created_at: datetime
    updated_at: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    timezone: str = Field("Asia/Kolkata", max_length=60)
    thread_id: Optional[str] = Field(None, max_length=120)
    external_reference_id: Optional[str] = Field(None, max_length=300)
    sentiment: Optional[str] = Field(None, max_length=20)
    tags: list[str] = Field(default_factory=list, max_length=40)
    projects: list[str] = Field(default_factory=list, max_length=40)
    people: list[str] = Field(default_factory=list, max_length=40)
    next_steps: list[str] = Field(default_factory=list, max_length=40)
    tasks: list[dict[str, Any]] = Field(default_factory=list, max_length=60)
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class RMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=12000)
    actions: list[Any] = Field(default_factory=list, max_length=20)
    created_at: datetime


class RSession(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: uuid.UUID
    title: str = Field("New chat", max_length=120)
    created_at: datetime
    updated_at: datetime
    messages: list[RMessage] = Field(default_factory=list, max_length=2000)


class RReminder(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: uuid.UUID
    text: str = Field(min_length=1, max_length=300)
    remind_at: datetime
    status: str = Field("pending", pattern="^(pending|sent|done)$")
    created_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None


class RReview(BaseModel):
    model_config = ConfigDict(extra="ignore")

    day: date
    summary: str = Field(max_length=4000)
    stats: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    emailed: bool = True


class RestorePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    memories: list[RMemory] = Field(default_factory=list, max_length=RESTORE_MAX_ITEMS)
    chat_sessions: list[RSession] = Field(default_factory=list, max_length=RESTORE_MAX_ITEMS)
    reminders: list[RReminder] = Field(default_factory=list, max_length=RESTORE_MAX_ITEMS)
    reviews: list[RReview] = Field(default_factory=list, max_length=RESTORE_MAX_ITEMS)


def restore(svc, payload: RestorePayload) -> dict[str, int]:
    """Put archived data back into the cloud. Rows that already exist are left exactly as they are. Returns how many were added."""
    owner = svc.owner_id
    added = {"memories": 0, "chat_sessions": 0, "chat_messages": 0, "reminders": 0, "reviews": 0}
    with svc.db() as conn:
        svc._assert_capacity(conn)
        for m in payload.memories:
            added["memories"] += conn.execute(
                "INSERT INTO memory_entries (id, owner_id, source, title, content, raw_transcript, summary, created_at, updated_at, start_time, end_time, "
                "timezone, thread_id, external_reference_id, sentiment, tags, projects, people, next_steps, tasks, metadata_json) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (m.id, owner, m.source, m.title, m.content, m.raw_transcript, m.summary, m.created_at, m.updated_at or m.created_at, m.start_time, m.end_time,
                 m.timezone, m.thread_id, m.external_reference_id, m.sentiment if m.sentiment in ("positive", "neutral", "negative", "mixed") else None,
                 m.tags, m.projects, m.people, m.next_steps, Jsonb(m.tasks), Jsonb(m.metadata_json))).rowcount
        for s in payload.chat_sessions:
            if conn.execute("INSERT INTO chat_sessions (id, owner_id, title, created_at, updated_at) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                            (s.id, owner, s.title, s.created_at, s.updated_at)).rowcount:
                added["chat_sessions"] += 1
                for msg in s.messages:   # messages are only restored together with a session that did not exist (never duplicated)
                    conn.execute("INSERT INTO chat_messages (session_id, role, content, actions, created_at) VALUES (%s,%s,%s,%s,%s)",
                                 (s.id, msg.role, msg.content, Jsonb(msg.actions), msg.created_at))
                    added["chat_messages"] += 1
        for r in payload.reminders:
            added["reminders"] += conn.execute(
                "INSERT INTO reminders (id, owner_id, text, remind_at, status, created_at, sent_at) VALUES (%s,%s,%s,%s,%s,COALESCE(%s, now()),%s) ON CONFLICT DO NOTHING",
                (r.id, owner, r.text, r.remind_at, r.status, r.created_at, r.sent_at)).rowcount
        for v in payload.reviews:
            added["reviews"] += conn.execute(
                "INSERT INTO daily_reviews (owner_id, day, summary, stats, details, emailed) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (owner, v.day, v.summary, Jsonb(v.stats), Jsonb(v.details), v.emailed)).rowcount
    return added


# ------------------------------------------------------------------------------------------------ clean-up of what is already on the PC
class PruneExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chat_session_ids: list[uuid.UUID] = Field(default_factory=list, max_length=500)
    reminder_ids: list[uuid.UUID] = Field(default_factory=list, max_length=500)
    review_days: list[date] = Field(default_factory=list, max_length=500)


def prune_extra(svc, p: PruneExtra) -> dict[str, int]:
    owner = svc.owner_id
    out = {"chat_sessions": 0, "reminders": 0, "reviews": 0}
    with svc.db() as conn:
        if p.chat_session_ids:   # messages go with their session (ON DELETE CASCADE)
            out["chat_sessions"] = conn.execute("DELETE FROM chat_sessions WHERE owner_id = %s AND id = ANY(%s::uuid[])",
                                                (owner, [str(i) for i in p.chat_session_ids])).rowcount
        if p.reminder_ids:
            out["reminders"] = conn.execute("DELETE FROM reminders WHERE owner_id = %s AND id = ANY(%s::uuid[]) AND status <> 'pending'",
                                            (owner, [str(i) for i in p.reminder_ids])).rowcount
        if p.review_days:
            out["reviews"] = conn.execute("DELETE FROM daily_reviews WHERE owner_id = %s AND day = ANY(%s::date[])", (owner, p.review_days)).rowcount
    if any(out.values()):
        import psycopg

        try:
            with psycopg.connect(svc.settings.database_url, autocommit=True, connect_timeout=15) as conn:
                for table in ("chat_sessions", "chat_messages", "reminders", "daily_reviews"):
                    conn.execute(f"VACUUM FULL {table}")
        except Exception:
            pass   # space is still reused; the size figure just drops later
    return out
