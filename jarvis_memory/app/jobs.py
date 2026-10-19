"""The scheduled work: read the owner's e-mail replies, send the hourly check-in, find due reminders.

``run_tick`` is called by the scheduler every ~15 minutes (POST /api/jarvis/jobs/tick). Everything here is idempotent, so a
duplicate or late call never sends two e-mails or stores a reply twice.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from . import creative, review as review_module
from .config import Settings
from .mailer import Mailbox
from .schemas import MemoryCreate
from .service import CapacityError, MemoryService

log = logging.getLogger("jarvis_memory.jobs")
_NO_AI = Settings()  # no API key: build_checkin() falls back to the built-in wording
SEND_ATTEMPTS = 2
LOG_KEEP = 200


def in_quiet_hours(hour: int, start: int, end: int) -> bool:
    if start == end:
        return False
    return (hour >= start or hour < end) if start > end else (start <= hour < end)


def latest_email_slot(local: datetime, times: str) -> Optional[datetime]:
    """The scheduled e-mail time (local) that passed within the last 2 hours, e.g. "09:00,12:00,17:00"."""
    best = None
    for part in times.split(","):
        try:
            h, m = (int(x) for x in part.strip().split(":"))
        except ValueError:
            continue
        t = local.replace(hour=h, minute=m, second=0, microsecond=0)
        if t <= local and local - t <= timedelta(hours=2) and (best is None or t > best):   # a slot missed by more than 2 h is skipped
            best = t
    return best


def build_checkin(owner_name: str, token: str, open_tasks: list[str], checkin: Optional[creative.Checkin] = None
                  ) -> tuple[str, str, str]:
    """(subject, plain text, colourful HTML). The [JM-token] in the subject links the owner's reply back to this e-mail."""
    c = checkin or creative.compose(_NO_AI, owner_name, open_tasks, 12, seed=0)
    lines = [f"{c.emoji} {c.headline}", "", f"Hi {owner_name or 'there'},", "", c.question, "", c.nudge, ""]
    if open_tasks:
        lines += ["Still on your list:"] + [f"  - {t}" for t in open_tasks[:5]] + [""]
    lines += ["Just reply to this e-mail in your own words. JARVIS saves it as a memory.",
              "(Replies from other addresses are ignored.)"]
    return f"{c.emoji} {c.subject} [JM-{token}]", "\n".join(lines), creative.render_html(c, owner_name, open_tasks)


def _log(svc: MemoryService, job: str, ok: bool, detail: str) -> None:
    with svc.db() as conn:
        conn.execute("INSERT INTO job_log (owner_id, job, ok, detail) VALUES (%s,%s,%s,%s)",
                     (svc.owner_id, job, ok, detail[:500]))
        conn.execute("DELETE FROM job_log WHERE owner_id = %s AND id NOT IN "
                     "(SELECT id FROM job_log WHERE owner_id = %s ORDER BY id DESC LIMIT %s)",
                     (svc.owner_id, svc.owner_id, LOG_KEEP))


def poll_replies(svc: MemoryService, mailbox: Mailbox, owner_email: str) -> dict[str, Any]:
    saved = duplicates = 0
    for reply in mailbox.fetch_replies(owner_email):
        # The public API only allows manual/voice sources; "email" is set here, server-side, after the sender was verified.
        data = MemoryCreate(content=reply.text, thread_id=reply.thread_id).model_copy(update={"source": "email"})
        memory = svc.create(data, external_reference_id=reply.message_id, when=reply.sent_at)
        if memory:
            saved += 1
        else:
            duplicates += 1
    return {"saved": saved, "duplicates": duplicates}


def send_checkin_if_due(svc: MemoryService, mailbox: Mailbox, now: datetime) -> dict[str, Any]:
    s = svc.get_settings()
    if not s["email_enabled"]:
        return {"sent": False, "why": "email_disabled"}
    if not s["owner_email"]:
        return {"sent": False, "why": "no_owner_email"}
    local = now.astimezone(ZoneInfo(s["timezone"]))
    slot = latest_email_slot(local, s.get("email_times") or "")
    if slot is None:
        return {"sent": False, "why": "not_due"}
    last = s["last_email_at"]
    if last and last >= slot.astimezone(last.tzinfo):      # this time slot's e-mail already went out
        return {"sent": False, "why": "not_due"}
    count = s["email_sent_count"] if s["email_sent_day"] == local.date() else 0
    if count >= s["max_emails_per_day"]:
        return {"sent": False, "why": "daily_limit"}
    tasks = svc.open_task_texts()
    wording = creative.compose(svc.settings, s["owner_name"], tasks, local.hour)
    subject, body, html_body = build_checkin(s["owner_name"], uuid.uuid4().hex[:12], tasks, wording)
    if svc.settings.smtp_mode == "external" and not svc.settings.brevo_api_key:
        # Render's free plan blocks outgoing SMTP, so the scheduler (GitHub Actions) sends it and then confirms via /mail/sent.
        return {"sent": False, "why": "external_delivery",
                "deliver": {"to": s["owner_email"], "subject": subject, "text": body, "html": html_body}}
    last_error: Optional[Exception] = None
    for _ in range(SEND_ATTEMPTS):
        try:
            mailbox.send(s["owner_email"], subject, body, html_body)
            return {"sent": True, "count_today": svc.record_email_sent(now)}
        except Exception as exc:  # smtplib / network errors; only the type is reported, never credentials
            last_error = exc
    return {"sent": False, "why": "send_failed", "error": type(last_error).__name__}


TASK_NUDGE_MINUTES = 30
SLACK_S = 120   # scheduler jitter: a ping a minute or two early still counts


def _push_all(svc: MemoryService, pusher, data: dict[str, Any]) -> bool:
    """Ring every registered phone. Forgets phones the app was removed from. True if at least one accepted it."""
    delivered = False
    for token in svc.device_tokens():
        outcome = pusher.send(token, data)
        if outcome == "ok":
            delivered = True
        elif outcome == "gone":
            svc.forget_device(token)
    return delivered


def send_pushes(svc: MemoryService, pusher, now: datetime) -> dict[str, Any]:
    """Phone alerts: due reminders (any time), the hourly call and the 30-minute open-task nudge (09:00-22:00 only)."""
    if pusher is None:
        return {"why": "firebase_not_configured"}
    if not svc.device_tokens():
        return {"why": "no_phone_registered"}
    s = svc.get_settings()
    out: dict[str, Any] = {"reminders": 0, "call": False, "task_nudge": False}
    for r in svc.claim_due_reminders(now):
        if _push_all(svc, pusher, {"type": "reminder", "id": r["id"], "line": f"Reminder: {r['text']}", "title": "JARVIS reminder"}):
            svc.record_push(now)
            out["reminders"] += 1
        else:
            svc.unclaim_reminder(r["id"])
    if not s["push_enabled"]:
        out["why"] = "push_disabled"
        return out
    local = now.astimezone(ZoneInfo(s["timezone"]))
    if in_quiet_hours(local.hour, s["quiet_start_hour"], s["quiet_end_hour"]):
        out["why"] = "quiet_hours"
        return out
    s = svc.get_settings()
    sent_today = s["push_sent_count"] if s["push_sent_day"] == local.date() else 0
    if sent_today >= s["max_notifications_per_day"]:
        out["why"] = "daily_limit"
        return out
    tasks = svc.open_task_texts()
    last_call = s["last_call_at"]
    if not last_call or (now - last_call).total_seconds() >= s["push_interval_minutes"] * 60 - SLACK_S:
        wording = creative.compose(svc.settings, s["owner_name"], tasks, local.hour)
        if _push_all(svc, pusher, {"type": "call", "line": wording.spoken, "title": wording.headline, "emoji": wording.emoji}):
            svc.record_push(now, call=True)
            out["call"] = True
            return out
    contacts = [t for t in (s["last_call_at"], s["last_task_push_at"]) if t]   # a call counts as a nudge too: never closer than 30 min
    last_contact = max(contacts) if contacts else None
    if tasks and (last_contact is None or (now - last_contact).total_seconds() >= TASK_NUDGE_MINUTES * 60 - SLACK_S):
        line = f"You still have {len(tasks)} open task{'s' if len(tasks) != 1 else ''}. First: {tasks[0]}. Tell me where it stands."
        if _push_all(svc, pusher, {"type": "task", "line": line, "title": "Open tasks"}):
            svc.record_push(now, task=True)
            out["task_nudge"] = True
    return out


def send_news(svc: MemoryService, pusher, now: datetime, fetcher=None) -> dict[str, Any]:
    """One fresh AI / quantum / space story as a phone notification, at most every news_interval_minutes and max_news_per_day a day."""
    from . import news

    if pusher is None or not svc.device_tokens():
        return {"why": "no_phone"}
    s = svc.get_settings()
    if not s["news_enabled"]:
        return {"why": "disabled"}
    local = now.astimezone(ZoneInfo(s["timezone"]))
    if in_quiet_hours(local.hour, s["quiet_start_hour"], s["quiet_end_hour"]):
        return {"why": "quiet_hours"}
    last = s["last_news_at"]
    if last and (now - last).total_seconds() < s["news_interval_minutes"] * 60 - SLACK_S:
        return {"why": "not_due"}
    sent_today = s["news_sent_count"] if s["news_sent_day"] == local.date() else 0
    if sent_today >= s["max_news_per_day"]:
        return {"why": "daily_limit"}
    topics = [t for t in s["news_topics"].split(",") if t in news.FEEDS]
    if not topics:
        return {"why": "no_topics"}
    with svc.db() as conn:
        conn.execute("DELETE FROM public.news_seen WHERE owner_id = %s AND seen_at < now() - make_interval(days => %s)", (svc.owner_id, news.KEEP_DAYS))
        seen = {r["link_hash"] for r in conn.execute("SELECT link_hash FROM public.news_seen WHERE owner_id = %s", (svc.owner_id,)).fetchall()}
    order = topics[sent_today % len(topics):] + topics[:sent_today % len(topics)]   # rotate topics through the day
    fetch = fetcher or news.fetch
    for topic in order:
        for story in news.latest(topic, now, fetch):
            h = news.link_hash(story["link"])
            if h in seen:
                continue
            data = {"type": "news", "title": f"{news.LABEL[topic]} news", "line": story["title"], "body": story["summary"], "url": story["link"], "topic": topic}
            if _push_all(svc, pusher, data):
                with svc.db() as conn:
                    conn.execute("INSERT INTO public.news_seen (owner_id, link_hash) VALUES (%s, %s) ON CONFLICT DO NOTHING", (svc.owner_id, h))
                    conn.execute("UPDATE memory_settings SET last_news_at = %s, news_sent_day = %s, news_sent_count = %s WHERE owner_id = %s",
                                 (now, local.date(), sent_today + 1, svc.owner_id))
                return {"sent": True, "topic": topic, "title": story["title"][:80]}
            return {"why": "push_failed"}
    return {"why": "nothing_new"}


def send_review(svc: MemoryService, pusher, mailbox: Optional[Mailbox], now: datetime) -> dict[str, Any]:
    """The nightly Daily Review: build it once a day at the chosen time (before quiet hours), push it, and e-mail it."""
    s = svc.get_settings()
    if not s["review_enabled"]:
        return {"why": "disabled"}
    local = now.astimezone(ZoneInfo(s["timezone"]))
    today = local.date()
    out: dict[str, Any] = {}
    if s["last_review_on"] != today:
        if (local.hour, local.minute) < (s["review_hour"], s["review_minute"]):
            return {"why": "not_yet"}
        if local.hour >= s["quiet_start_hour"] and s["quiet_start_hour"] > s["review_hour"]:
            return {"why": "quiet_hours"}               # missed the evening: no review pings at night
        built = review_module.build(svc, today)
        with svc.db() as conn:
            conn.execute("UPDATE memory_settings SET last_review_on = %s WHERE owner_id = %s", (today, svc.owner_id))
        out["generated"] = True
        title, line = review_module.push_text(built)
        out["push"] = bool(pusher and s["push_enabled"] and svc.device_tokens()
                           and _push_all(svc, pusher, {"type": "review", "title": title, "line": line, "day": today.isoformat()}))
    stored = review_module.get(svc, today)
    if stored and not stored["emailed"] and s["email_enabled"] and s["owner_email"]:
        subject, text, html_body = review_module.render_email(stored, s["owner_name"], today)
        if svc.settings.smtp_mode == "external" and not svc.settings.brevo_api_key:
            out["email"] = {"why": "external_delivery", "day": today.isoformat(),
                            "deliver": {"to": s["owner_email"], "subject": subject, "text": text, "html": html_body}}
        elif mailbox is not None:
            try:
                mailbox.send(s["owner_email"], subject, text, html_body)
                svc.mark_review_emailed(today)
                out["email"] = {"sent": True}
            except Exception as exc:
                out["email"] = {"sent": False, "error": type(exc).__name__}
    return out


def run_tick(svc: MemoryService, mailbox: Optional[Mailbox], now: Optional[datetime] = None, pusher=None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    result: dict[str, Any] = {}
    s = svc.get_settings()
    if mailbox is None:
        result["email"] = {"sent": False, "why": "gmail_not_configured"}
    else:
        try:
            result["replies"] = (poll_replies(svc, mailbox, s["owner_email"]) if s["owner_email"]
                                 else {"saved": 0, "why": "no_owner_email"})
        except CapacityError:
            result["replies"] = {"saved": 0, "error": "database_full"}
        except Exception as exc:
            import imaplib
            hint = (str(exc)[:90] if isinstance(exc, imaplib.IMAP4.error) else "")   # server wording like "[AUTHENTICATIONFAILED] ...": no secrets
            result["replies"] = {"saved": 0, "error": type(exc).__name__, **({"detail": hint} if hint else {})}
        result["email"] = send_checkin_if_due(svc, mailbox, now)
    result["push"] = send_pushes(svc, pusher, now)
    try:
        result["news"] = send_news(svc, pusher, now)
    except Exception as exc:  # news is a bonus: it must never break calls, reminders or e-mail
        result["news"] = {"error": type(exc).__name__}
    try:
        result["review"] = send_review(svc, pusher, mailbox, now)
    except Exception as exc:  # the review must never break calls, reminders or e-mail
        result["review"] = {"error": type(exc).__name__}   # reminders are only marked "sent" when a phone actually accepted them
    ok = (not any(isinstance(v, dict) and v.get("error") for v in result.values())
          and result.get("email", {}).get("why") != "send_failed")
    summary = {k: ({a: b for a, b in v.items() if a != "deliver"} if isinstance(v, dict) else v) for k, v in result.items()}
    _log(svc, "tick", ok, str(summary))   # never log the e-mail body
    return result
