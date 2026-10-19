"""Numbers for the desktop Insights screen: activity over time, by source, hour-of-day heat map, tasks, topics, mood, reminders.

One read-only pass of owner-scoped aggregate queries (no text content leaves the database except tag/person/project names).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

SOURCES = ("manual", "voice", "email", "daily_review")
RANGES = {"day": 1, "week": 7, "month": 30}
HEAT_DAYS = 56


def _top(conn, owner, column: str, start: datetime, limit: int = 10) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"SELECT lower(x) AS name, count(*) AS n FROM memory_entries m, unnest(m.{column}) AS x "
        "WHERE m.owner_id = %s AND m.deleted_at IS NULL AND m.created_at >= %s GROUP BY 1 ORDER BY n DESC, name LIMIT %s",
        (owner, start, limit)).fetchall()
    return [{"name": r["name"], "count": r["n"]} for r in rows]


def build(svc, range_name: str = "week", now: Optional[datetime] = None) -> dict[str, Any]:
    if range_name not in RANGES:
        raise ValueError("range must be day, week or month")
    tzname = svc.settings.timezone
    tz = ZoneInfo(tzname)
    now = (now or datetime.now(tz)).astimezone(tz)
    today = now.date()
    days = RANGES[range_name]
    hourly = range_name == "day"
    first_day = today - timedelta(days=days - 1)
    start = datetime.combine(first_day, datetime.min.time(), tzinfo=tz)
    owner = svc.owner_id
    n = 24 if hourly else days
    labels = [f"{h:02d}" for h in range(24)] if hourly else [(first_day + timedelta(days=i)).isoformat() for i in range(days)]
    index = {lab: i for i, lab in enumerate(labels)}

    def bucket_of(d: date, hour: int) -> Optional[int]:
        return index.get(f"{hour:02d}" if hourly else d.isoformat())

    series = {s: [0] * n for s in SOURCES}
    done = [0] * n
    created = [0] * n
    with svc.db() as conn:
        for r in conn.execute(
                "SELECT (created_at AT TIME ZONE %s)::date AS d, extract(hour FROM created_at AT TIME ZONE %s)::int AS h, source, count(*) AS c "
                "FROM memory_entries WHERE owner_id = %s AND deleted_at IS NULL AND created_at >= %s GROUP BY 1, 2, 3",
                (tzname, tzname, owner, start)).fetchall():
            i = bucket_of(r["d"], r["h"])
            if i is not None and r["source"] in series:
                series[r["source"]][i] += r["c"]
        for r in conn.execute(
                "SELECT (m.updated_at AT TIME ZONE %s)::date AS d, extract(hour FROM m.updated_at AT TIME ZONE %s)::int AS h, count(*) AS c "
                "FROM memory_entries m, jsonb_array_elements(m.tasks) t WHERE m.owner_id = %s AND m.deleted_at IS NULL "
                "AND t ->> 'status' = 'done' AND m.updated_at >= %s GROUP BY 1, 2", (tzname, tzname, owner, start)).fetchall():
            i = bucket_of(r["d"], r["h"])
            if i is not None:
                done[i] += r["c"]
        for r in conn.execute(
                "SELECT (m.created_at AT TIME ZONE %s)::date AS d, extract(hour FROM m.created_at AT TIME ZONE %s)::int AS h, count(*) AS c "
                "FROM memory_entries m, jsonb_array_elements(m.tasks) t WHERE m.owner_id = %s AND m.deleted_at IS NULL AND m.created_at >= %s GROUP BY 1, 2",
                (tzname, tzname, owner, start)).fetchall():
            i = bucket_of(r["d"], r["h"])
            if i is not None:
                created[i] += r["c"]
        open_tasks = conn.execute(
            "SELECT count(*) AS c FROM memory_entries m, jsonb_array_elements(m.tasks) t WHERE m.owner_id = %s AND m.deleted_at IS NULL "
            "AND t ->> 'status' = 'pending'", (owner,)).fetchone()["c"]
        heat = [[0] * 24 for _ in range(7)]
        heat_start = datetime.combine(today - timedelta(days=HEAT_DAYS - 1), datetime.min.time(), tzinfo=tz)
        for r in conn.execute(
                "SELECT extract(isodow FROM created_at AT TIME ZONE %s)::int AS dow, extract(hour FROM created_at AT TIME ZONE %s)::int AS h, count(*) AS c "
                "FROM memory_entries WHERE owner_id = %s AND deleted_at IS NULL AND created_at >= %s GROUP BY 1, 2",
                (tzname, tzname, owner, heat_start)).fetchall():
            heat[r["dow"] - 1][r["h"]] = r["c"]
        top = {"tags": _top(conn, owner, "tags", start), "people": _top(conn, owner, "people", start), "projects": _top(conn, owner, "projects", start)}
        sentiment = {r["s"]: r["c"] for r in conn.execute(
            "SELECT coalesce(sentiment, 'unknown') AS s, count(*) AS c FROM memory_entries WHERE owner_id = %s AND deleted_at IS NULL "
            "AND created_at >= %s GROUP BY 1", (owner, start)).fetchall()}
        reminders = {r["status"]: r["c"] for r in conn.execute(
            "SELECT status, count(*) AS c FROM reminders WHERE owner_id = %s AND remind_at >= %s GROUP BY 1", (owner, start)).fetchall()}
        totals_row = conn.execute(
            "SELECT count(*) AS memories, coalesce(sum(length(content)), 0) AS chars FROM memory_entries "
            "WHERE owner_id = %s AND deleted_at IS NULL AND created_at >= %s", (owner, start)).fetchone()
        active_days = {r["d"] for r in conn.execute(
            "SELECT DISTINCT (created_at AT TIME ZONE %s)::date AS d FROM memory_entries WHERE owner_id = %s AND deleted_at IS NULL "
            "AND created_at >= %s", (tzname, owner, now - timedelta(days=400))).fetchall()}
    streak, cursor = 0, today if today in active_days else today - timedelta(days=1)
    while cursor in active_days:
        streak += 1
        cursor -= timedelta(days=1)
    flat = [(c, d, h) for d, row in enumerate(heat) for h, c in enumerate(row)]
    busiest = max(flat) if any(c for c, _, _ in flat) else None
    return {
        "range": range_name, "bucket": "hour" if hourly else "day", "labels": labels, "series": series,
        "tasks": {"done": done, "created": created, "open": open_tasks},
        "heat": heat, "top": top, "sentiment": sentiment, "reminders": reminders,
        "totals": {"memories": totals_row["memories"], "words": round(int(totals_row["chars"]) / 5),
                   "busiest_day": None if not busiest else ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][busiest[1]],
                   "busiest_hour": None if not busiest else busiest[2]},
        "streak": streak,
    }
