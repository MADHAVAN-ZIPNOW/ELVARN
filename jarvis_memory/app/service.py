"""The memory repository: every SQL statement lives here, all of them parameterised and owner-scoped."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from psycopg.types.json import Jsonb

from .config import Settings
from .db import connect, database_size_mb
from .extraction import Extraction, Extractor, extractive_summary, fallback_title
from .schemas import MemoryCreate, MemoryUpdate, SearchRequest, SettingsUpdate

COLUMNS = (
    "id, source, title, content, raw_transcript, summary, created_at, updated_at, start_time, end_time, timezone, "
    "thread_id, external_reference_id, sentiment, tags, projects, people, next_steps, tasks, metadata_json"
)
M_COLUMNS = ", ".join("m." + c.strip() for c in COLUMNS.split(","))
_TS_ARGS = "m.title, m.content, m.raw_transcript, m.summary, m.tags, m.projects, m.people, m.next_steps, m.tasks"
TSV = f"jm_tsv({_TS_ARGS})"        # must stay identical to the expression in the GIN index
SEARCH_TEXT = f"jm_search_text({_TS_ARGS})"
MARK_START, MARK_END = "\u0002", "\u0003"  # highlight markers; the UI renders them as <mark>, never as raw HTML
EXPORT_LIMIT = 50_000


class NotFound(Exception):
    pass


class CapacityError(Exception):
    """The free database is nearly full: new writes are refused so it never hits the provider's hard limit."""


def _memory(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["id"] = str(out["id"])
    out["tasks"] = out.get("tasks") or []
    out["metadata_json"] = out.get("metadata_json") or {}
    out["attachments"] = []
    return out


def _extraction_meta(ex: Extraction) -> dict[str, Any]:
    meta: dict[str, Any] = {"status": ex.status, "at": datetime.now(timezone.utc).isoformat()}
    if ex.model:
        meta["model"] = ex.model
    if ex.error:
        meta["error"] = ex.error
    if ex.dropped:
        meta["grounding_dropped"] = ex.dropped  # what was removed because the text did not support it
    return meta


class MemoryService:
    def __init__(self, settings: Settings, extractor: Optional[Extractor] = None) -> None:
        self.settings = settings
        self.owner_id: uuid.UUID = settings.owner_id
        self.extractor = extractor or Extractor(settings)

    def db(self):
        return connect(self.settings.database_url)

    # ------------------------------------------------------------------------------------------ capacity
    def _soft_limit_mb(self, conn) -> int:
        row = conn.execute("SELECT db_soft_limit_mb FROM memory_settings WHERE owner_id = %s", (self.owner_id,)).fetchone()
        return int(row["db_soft_limit_mb"]) if row else self.settings.db_soft_limit_mb

    def _assert_capacity(self, conn) -> None:
        size, limit = database_size_mb(conn), self._soft_limit_mb(conn)
        if size >= limit:
            raise CapacityError(
                f"The database is {size:.0f} MB of its {limit} MB limit. Export or delete old memories, then try again."
            )

    # ------------------------------------------------------------------------------------------ create
    def create(self, data: MemoryCreate, now: Optional[datetime] = None, *, external_reference_id: Optional[str] = None,
               when: Optional[datetime] = None) -> Optional[dict[str, Any]]:
        """Save the memory FIRST (raw text is never lost), then enrich it with AI in a second step.

        With ``external_reference_id`` (an e-mail Message-ID) the call is idempotent: a repeat stores nothing, returns None."""
        now = now or datetime.now(timezone.utc)
        raw_transcript = data.raw_transcript if data.source == "voice" else None
        with self.db() as conn:
            self._assert_capacity(conn)
            if external_reference_id and conn.execute(
                    "SELECT 1 FROM memory_entries WHERE owner_id = %s AND external_reference_id = %s",
                    (self.owner_id, external_reference_id)).fetchone():
                return None
            row = conn.execute(
                f"""INSERT INTO memory_entries
                    (owner_id, source, title, content, raw_transcript, summary, start_time, end_time, timezone, thread_id,
                     sentiment, tags, projects, people, next_steps, tasks, metadata_json, external_reference_id, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING {COLUMNS}""",
                (
                    self.owner_id, data.source, data.title or fallback_title(data.content), data.content, raw_transcript,
                    data.summary or "", data.start_time, data.end_time, self.settings.timezone, data.thread_id,
                    data.sentiment, data.tags or [], data.projects or [], data.people or [], data.next_steps or [],
                    Jsonb([t.model_dump(mode="json") for t in (data.tasks or [])]),
                    Jsonb({"extraction": {"status": "pending" if data.extract else "skipped"}}),
                    external_reference_id, when or now,
                ),
            ).fetchone()
        memory = _memory(row)
        if data.extract:
            memory = self._enrich(memory["id"], data.content, data.source, now, overwrite=False, user=data)
        return memory

    def re_extract(self, memory_id: str) -> dict[str, Any]:
        """Retry AI extraction (e.g. after a failure). Never overwrites fields you already edited."""
        current = self.get(memory_id)
        previous = (current["metadata_json"].get("extraction") or {}).get("status")
        return self._enrich(memory_id, current["content"], current["source"], datetime.now(timezone.utc),
                            overwrite=previous != "ok", existing=current)

    def _enrich(self, memory_id: str, text: str, source: str, when: datetime, *, overwrite: bool,
                user: Optional[MemoryCreate] = None, existing: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        # The AI call happens with NO database connection open (Neon's free tier has few connections).
        ex = self.extractor.extract(text, source, when)
        with self.db() as conn:
            row = conn.execute(f"SELECT {COLUMNS} FROM memory_entries WHERE id = %s AND owner_id = %s AND deleted_at IS NULL",
                               (memory_id, self.owner_id)).fetchone()
            if row is None:
                raise NotFound(memory_id)
            cur = _memory(row)

            def pick(field: str, ai_value, empty) -> Any:
                value = cur[field]
                if user is not None and getattr(user, field, None):
                    return value                              # the user supplied it: keep it
                if overwrite or value in (empty, None, "", [], ()):
                    return ai_value if ai_value not in (None, "", []) else value
                return value

            title_is_fallback = cur["title"] == fallback_title(text)
            new_title = ex.title if (overwrite or title_is_fallback) and not (user and user.title) else cur["title"]
            new_summary = pick("summary", ex.summary, "")
            meta = dict(cur["metadata_json"])
            meta["extraction"] = _extraction_meta(ex)
            updated = conn.execute(
                f"""UPDATE memory_entries SET title=%s, summary=%s, tags=%s, projects=%s, people=%s, next_steps=%s, tasks=%s,
                       sentiment=%s, metadata_json=%s, updated_at = now()
                    WHERE id=%s AND owner_id=%s RETURNING {COLUMNS}""",
                (new_title, new_summary, pick("tags", ex.tags, []), pick("projects", ex.projects, []),
                 pick("people", ex.people, []), pick("next_steps", ex.next_steps, []),
                 Jsonb(pick("tasks", ex.tasks, [])), pick("sentiment", ex.sentiment, None), Jsonb(meta),
                 memory_id, self.owner_id),
            ).fetchone()
        return _memory(updated)

    # ------------------------------------------------------------------------------------------ read
    def get(self, memory_id: str) -> dict[str, Any]:
        with self.db() as conn:
            row = conn.execute(f"SELECT {COLUMNS} FROM memory_entries WHERE id = %s AND owner_id = %s AND deleted_at IS NULL",
                               (memory_id, self.owner_id)).fetchone()
        if row is None:
            raise NotFound(memory_id)
        return _memory(row)

    def list(self, limit: int = 20, offset: int = 0, source: Optional[str] = None) -> list[dict[str, Any]]:
        sql = f"SELECT {COLUMNS} FROM memory_entries WHERE owner_id = %s AND deleted_at IS NULL"
        params: list[Any] = [self.owner_id]
        if source:
            sql += " AND source = %s"
            params.append(source)
        sql += " ORDER BY created_at DESC, id LIMIT %s OFFSET %s"
        params += [limit, offset]
        with self.db() as conn:
            return [_memory(r) for r in conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------------------------------ update / delete
    def update(self, memory_id: str, patch: MemoryUpdate) -> dict[str, Any]:
        sets: list[str] = []
        params: list[Any] = []
        for field in ("title", "summary", "tags", "projects", "people", "next_steps"):
            value = getattr(patch, field)
            if value is not None:
                sets.append(f"{field} = %s")
                params.append(value)
        if patch.tasks is not None:
            sets.append("tasks = %s")
            params.append(Jsonb([t.model_dump(mode="json") for t in patch.tasks]))
        if patch.clear_sentiment:
            sets.append("sentiment = NULL")
        elif patch.sentiment is not None:
            sets.append("sentiment = %s")
            params.append(patch.sentiment)
        if not sets:
            return self.get(memory_id)
        sets.append("updated_at = now()")
        with self.db() as conn:
            row = conn.execute(
                f"UPDATE memory_entries SET {', '.join(sets)} WHERE id = %s AND owner_id = %s AND deleted_at IS NULL RETURNING {COLUMNS}",
                params + [memory_id, self.owner_id],
            ).fetchone()
        if row is None:
            raise NotFound(memory_id)
        return _memory(row)

    def delete(self, memory_id: str) -> None:
        """Soft delete (hidden everywhere); ``purge_deleted`` removes it for good later, keeping the database small."""
        with self.db() as conn:
            cur = conn.execute("UPDATE memory_entries SET deleted_at = now() WHERE id = %s AND owner_id = %s AND deleted_at IS NULL",
                               (memory_id, self.owner_id))
            if cur.rowcount == 0:
                raise NotFound(memory_id)

    def purge_deleted(self, older_than_days: int = 30) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        with self.db() as conn:
            return conn.execute("DELETE FROM memory_entries WHERE owner_id = %s AND deleted_at IS NOT NULL AND deleted_at < %s",
                                (self.owner_id, cutoff)).rowcount

    def delete_all(self) -> int:
        with self.db() as conn:
            return conn.execute("DELETE FROM memory_entries WHERE owner_id = %s", (self.owner_id,)).rowcount

    # ------------------------------------------------------------------------------------------ search
    def search(self, req: SearchRequest) -> dict[str, Any]:
        where = ["m.owner_id = %(owner)s", "m.deleted_at IS NULL"]
        p: dict[str, Any] = {"owner": self.owner_id}
        q = req.q.strip()
        if q:
            p["q"] = q
            where.append(f"{TSV} @@ websearch_to_tsquery('english'::regconfig, %(q)s)")
        if req.date_from:
            where.append("m.created_at >= %(date_from)s")
            p["date_from"] = req.date_from
        if req.date_to:
            where.append("m.created_at <= %(date_to)s")
            p["date_to"] = req.date_to
        if req.sources:
            where.append("m.source = ANY(%(sources)s)")
            p["sources"] = list(req.sources)
        if req.project:
            where.append("EXISTS (SELECT 1 FROM unnest(m.projects) x WHERE lower(x) = lower(%(project)s))")
            p["project"] = req.project
        if req.person:
            where.append("EXISTS (SELECT 1 FROM unnest(m.people) x WHERE lower(x) = lower(%(person)s))")
            p["person"] = req.person
        if req.tag:
            where.append("lower(%(tag)s) = ANY(m.tags)")
            p["tag"] = req.tag
        if req.task_status:
            where.append("m.tasks <> '[]'::jsonb AND m.tasks @> jsonb_build_array(jsonb_build_object('status', %(task_status)s::text))")
            p["task_status"] = req.task_status
        if req.sentiment:
            where.append("m.sentiment = %(sentiment)s")
            p["sentiment"] = req.sentiment
        if req.has_attachment is True:
            where.append("FALSE")  # attachments arrive with Stage 4 (media); none exist yet
        if req.has_transcript is True:
            where.append("m.raw_transcript IS NOT NULL")
        elif req.has_transcript is False:
            where.append("m.raw_transcript IS NULL")

        if q:
            tsq = "websearch_to_tsquery('english'::regconfig, %(q)s)"
            rank = f"ts_rank_cd({TSV}, {tsq})"
            snippet = (f"ts_headline('english'::regconfig, {SEARCH_TEXT}, {tsq}, "
                       f"'StartSel={MARK_START},StopSel={MARK_END},MaxFragments=2,MaxWords=28,MinWords=10,FragmentDelimiter= … ')")
        else:
            rank, snippet = "0.0", "left(m.content, 220)"
        order = {"newest": "m.created_at DESC, m.id", "oldest": "m.created_at ASC, m.id",
                 "relevance": "rank DESC, m.created_at DESC, m.id" if q else "m.created_at DESC, m.id"}[req.sort]
        p["limit"], p["offset"] = req.page_size, (req.page - 1) * req.page_size
        sql = (f"SELECT {M_COLUMNS}, {rank} AS rank, {snippet} AS snippet, count(*) OVER() AS total "
               f"FROM memory_entries m WHERE {' AND '.join(where)} ORDER BY {order} LIMIT %(limit)s OFFSET %(offset)s")
        with self.db() as conn:
            rows = conn.execute(sql, p).fetchall()
            total = rows[0]["total"] if rows else 0
            if not rows and req.page > 1:  # past the last page: still report the real total
                total = conn.execute(f"SELECT count(*) AS n FROM memory_entries m WHERE {' AND '.join(where)}", p).fetchone()["n"]
        items = []
        for r in rows:
            snip, rk = r.pop("snippet"), float(r.pop("rank"))
            r.pop("total")
            items.append({"memory": _memory(r), "snippet": snip, "rank": rk})
        return {"items": items, "total": int(total), "page": req.page, "page_size": req.page_size,
                "has_more": req.page * req.page_size < int(total)}

    # ------------------------------------------------------------------------------------------ export / stats / settings
    def export(self) -> dict[str, Any]:
        with self.db() as conn:
            rows = conn.execute(f"SELECT {COLUMNS} FROM memory_entries WHERE owner_id = %s AND deleted_at IS NULL "
                                f"ORDER BY created_at LIMIT %s", (self.owner_id, EXPORT_LIMIT)).fetchall()
        return {"exported_at": datetime.now(timezone.utc).isoformat(), "count": len(rows),
                "memories": [_memory(r) for r in rows]}

    def stats(self) -> dict[str, Any]:
        with self.db() as conn:
            counts = {r["source"]: r["n"] for r in conn.execute(
                "SELECT source, count(*) AS n FROM memory_entries WHERE owner_id = %s AND deleted_at IS NULL GROUP BY source",
                (self.owner_id,)).fetchall()}
            table = conn.execute("SELECT pg_total_relation_size('memory_entries') AS b").fetchone()["b"]
            size, limit = database_size_mb(conn), self._soft_limit_mb(conn)
        total = sum(counts.values())
        return {"memories": total, "by_source": counts, "database_mb": size, "limit_mb": limit,
                "used_percent": round(100 * size / limit, 1) if limit else 0,
                "avg_bytes_per_memory": round(table / total) if total else 0}

    def get_settings(self) -> dict[str, Any]:
        with self.db() as conn:
            conn.execute(
                "INSERT INTO memory_settings (owner_id, owner_name, owner_email, timezone, db_soft_limit_mb) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (owner_id) DO NOTHING",
                (self.owner_id, self.settings.owner_name, self.settings.owner_email, self.settings.timezone,
                 self.settings.db_soft_limit_mb))
            row = conn.execute("SELECT * FROM memory_settings WHERE owner_id = %s", (self.owner_id,)).fetchone()
        row = dict(row)
        row.pop("owner_id", None)
        return row

    def update_settings(self, patch: SettingsUpdate) -> dict[str, Any]:
        self.get_settings()  # make sure the row exists
        changes = patch.model_dump(exclude_none=True)
        if changes:
            sets = ", ".join(f"{k} = %s" for k in changes)
            with self.db() as conn:
                conn.execute(f"UPDATE memory_settings SET {sets}, updated_at = now() WHERE owner_id = %s",
                             list(changes.values()) + [self.owner_id])
        return self.get_settings()

    # ------------------------------------------------------------------------------------------ stage 2 helpers
    def open_task_texts(self, limit: int = 5) -> list[str]:
        """Pending tasks from the newest memories, for the check-in e-mail."""
        with self.db() as conn:
            rows = conn.execute(
                "SELECT t ->> 'task' AS task FROM (SELECT tasks FROM memory_entries WHERE owner_id = %s AND deleted_at IS NULL "
                "AND tasks <> '[]'::jsonb ORDER BY created_at DESC LIMIT 20) m, jsonb_array_elements(m.tasks) t "
                "WHERE t ->> 'status' = 'pending' LIMIT %s", (self.owner_id, limit)).fetchall()
        return [r["task"] for r in rows]

    def add_reminder(self, text: str, remind_at: datetime) -> dict[str, Any]:
        with self.db() as conn:
            row = conn.execute("INSERT INTO reminders (owner_id, text, remind_at) VALUES (%s,%s,%s) "
                               "RETURNING id, text, remind_at, status, created_at, sent_at", (self.owner_id, text, remind_at)).fetchone()
        return {**row, "id": str(row["id"])}

    def list_reminders(self, status: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        with self.db() as conn:
            rows = conn.execute(
                "SELECT id, text, remind_at, status, created_at, sent_at FROM reminders WHERE owner_id = %s "
                "AND (%s::text IS NULL OR status = %s) ORDER BY remind_at DESC LIMIT %s",
                (self.owner_id, status, status, limit)).fetchall()
        return [{**r, "id": str(r["id"])} for r in rows]

    def update_reminder(self, reminder_id: str, *, done: bool = False, snooze_minutes: Optional[int] = None) -> dict[str, Any]:
        """Done closes it; Snooze puts it back to pending a few minutes later (the call-style screen's buttons)."""
        with self.db() as conn:
            if done:
                row = conn.execute("UPDATE reminders SET status = 'done' WHERE id = %s AND owner_id = %s "
                                   "RETURNING id, text, remind_at, status, created_at, sent_at", (reminder_id, self.owner_id)).fetchone()
            else:
                row = conn.execute(
                    "UPDATE reminders SET status = 'pending', sent_at = NULL, remind_at = now() + make_interval(mins => %s) "
                    "WHERE id = %s AND owner_id = %s RETURNING id, text, remind_at, status, created_at, sent_at",
                    (snooze_minutes or 10, reminder_id, self.owner_id)).fetchone()
        if not row:
            raise NotFound()
        return {**row, "id": str(row["id"])}

    def delete_reminder(self, reminder_id: str) -> None:
        with self.db() as conn:
            if not conn.execute("DELETE FROM reminders WHERE id = %s AND owner_id = %s RETURNING 1",
                                (reminder_id, self.owner_id)).fetchone():
                raise NotFound()

    def claim_due_reminders(self, now: datetime) -> list[dict[str, Any]]:
        """Reminders whose time has come, marked 'sent' in the same statement so each fires exactly once."""
        with self.db() as conn:
            rows = conn.execute(
                "UPDATE reminders SET status = 'sent', sent_at = %s WHERE owner_id = %s AND status = 'pending' "
                "AND remind_at <= %s RETURNING id, text, remind_at", (now, self.owner_id, now)).fetchall()
        return [{**r, "id": str(r["id"])} for r in rows]

    def record_email_sent(self, now: datetime) -> int:
        """Bookkeeping after a check-in went out (interval + daily limit). Returns how many were sent today."""
        from zoneinfo import ZoneInfo

        s = self.get_settings()
        today = now.astimezone(ZoneInfo(s["timezone"])).date()
        count = (s["email_sent_count"] if s["email_sent_day"] == today else 0) + 1
        with self.db() as conn:
            conn.execute("UPDATE memory_settings SET last_email_at = %s, email_sent_day = %s, email_sent_count = %s "
                         "WHERE owner_id = %s", (now, today, count, self.owner_id))
        return count

    # ------------------------------------------------------------------------------------------ push (devices)
    def register_device(self, token: str) -> None:
        with self.db() as conn:
            conn.execute("INSERT INTO devices (token, owner_id) VALUES (%s,%s) ON CONFLICT (token) DO UPDATE SET seen_at = now(), "
                         "owner_id = EXCLUDED.owner_id", (token, self.owner_id))
            # a person has one or two phones: keep the newest few so stale tokens never pile up
            conn.execute("DELETE FROM devices WHERE owner_id = %s AND token NOT IN "
                         "(SELECT token FROM devices WHERE owner_id = %s ORDER BY seen_at DESC LIMIT 3)", (self.owner_id, self.owner_id))

    def device_tokens(self) -> list[str]:
        with self.db() as conn:
            return [r["token"] for r in conn.execute("SELECT token FROM devices WHERE owner_id = %s", (self.owner_id,)).fetchall()]

    def forget_device(self, token: str) -> None:
        with self.db() as conn:
            conn.execute("DELETE FROM devices WHERE token = %s AND owner_id = %s", (token, self.owner_id))

    def record_push(self, now: datetime, *, call: bool = False, task: bool = False) -> int:
        """Counts one notification for today's limit and stamps the call / task-nudge time. Returns today's count."""
        from zoneinfo import ZoneInfo

        s = self.get_settings()
        today = now.astimezone(ZoneInfo(s["timezone"])).date()
        count = (s["push_sent_count"] if s["push_sent_day"] == today else 0) + 1
        with self.db() as conn:
            conn.execute(
                "UPDATE memory_settings SET push_sent_day = %s, push_sent_count = %s, "
                "last_call_at = CASE WHEN %s THEN %s ELSE last_call_at END, "
                "last_task_push_at = CASE WHEN %s THEN %s ELSE last_task_push_at END WHERE owner_id = %s",
                (today, count, call, now, task, now, self.owner_id))
        return count

    def close_reminders_about(self, task_text: str) -> int:
        """When a task is deleted or finished, its waiting reminders must not ring any more. Returns how many were closed."""
        words = " ".join(task_text.lower().split())
        if len(words) < 4:
            return 0
        with self.db() as conn:
            cur = conn.execute("UPDATE reminders SET status = 'done' WHERE owner_id = %s AND status IN ('pending', 'sent') "
                               "AND (position(%s in lower(text)) > 0 OR position(lower(text) in %s) > 0)", (self.owner_id, words, words))
            return cur.rowcount or 0

    def unclaim_reminder(self, reminder_id: str) -> None:
        """Put a reminder back to pending when no phone accepted the push, so it is retried on the next tick."""
        with self.db() as conn:
            conn.execute("UPDATE reminders SET status = 'pending', sent_at = NULL WHERE id = %s AND owner_id = %s",
                         (reminder_id, self.owner_id))

    def mark_review_emailed(self, day) -> None:
        with self.db() as conn:
            conn.execute("UPDATE daily_reviews SET emailed = true WHERE owner_id = %s AND day = %s", (self.owner_id, day))

    def prune(self, ids: list[str]) -> int:
        """Hard-delete memories by id (owner-scoped) and VACUUM so the database really shrinks. The desktop calls this only
        after it has written those memories to its Word/Excel archive."""
        with self.db() as conn:
            deleted = conn.execute("DELETE FROM memory_entries WHERE owner_id = %s AND id = ANY(%s::uuid[])",
                                   (self.owner_id, ids)).rowcount
        if deleted:
            import psycopg

            try:   # VACUUM cannot run inside a transaction, so it gets its own autocommit connection
                with psycopg.connect(self.settings.database_url, autocommit=True, connect_timeout=15) as conn:
                    conn.execute("VACUUM FULL memory_entries")
            except Exception:
                pass   # space is still reused by new rows; the size figure just drops later
        return deleted

    # ------------------------------------------------------------------------------------------ tasks inside memories
    def find_open_tasks(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        """Open (pending/blocked) tasks whose text contains every word of ``query`` (newest memories first)."""
        words = [w for w in query.lower().split() if len(w) > 1]
        if not words:
            return []
        with self.db() as conn:
            rows = conn.execute("SELECT id, tasks FROM memory_entries WHERE owner_id = %s AND deleted_at IS NULL AND tasks <> '[]'::jsonb "
                                "ORDER BY created_at DESC LIMIT 300", (self.owner_id,)).fetchall()
        found = []
        for r in rows:
            for i, t in enumerate(r["tasks"] or []):
                text = str(t.get("task", ""))
                if t.get("status", "pending") != "done" and all(w in text.lower() for w in words):
                    found.append({"memory_id": str(r["id"]), "index": i, "task": text, "status": t.get("status", "pending")})
        return found[:limit]

    def change_task(self, memory_id: str, index: int, task_text: str, *, status: Optional[str] = None, remove: bool = False) -> bool:
        """Mark one task done, or remove it (kept in metadata so it can be restored). False if it changed in the meantime."""
        with self.db() as conn:
            row = conn.execute("SELECT tasks, metadata_json FROM memory_entries WHERE id = %s AND owner_id = %s AND deleted_at IS NULL",
                               (memory_id, self.owner_id)).fetchone()
            if not row:
                return False
            tasks = list(row["tasks"] or [])
            if index >= len(tasks) or tasks[index].get("task") != task_text:
                return False
            meta = dict(row["metadata_json"] or {})
            if remove:
                removed = list(meta.get("removed_tasks", []))
                removed.append({**tasks[index], "removed_at": datetime.now(timezone.utc).isoformat()})
                meta["removed_tasks"] = removed[-20:]
                del tasks[index]
            else:
                tasks[index] = {**tasks[index], "status": status}
            conn.execute("UPDATE memory_entries SET tasks = %s, metadata_json = %s, updated_at = now() WHERE id = %s AND owner_id = %s",
                         (Jsonb(tasks), Jsonb(meta), memory_id, self.owner_id))
        return True
