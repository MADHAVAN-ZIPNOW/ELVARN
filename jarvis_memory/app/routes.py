"""HTTP API (all under /api/jarvis/memory; the prefix avoids ELVARN's existing /api/memory route)."""
from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Optional

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from .db import DatabaseNotConfigured
from .schemas import (AskRequest, ChatRequest, DeviceRegister, PruneRequest, TaskRef, MemoryCreate, MemoryOut, MemoryUpdate, ReminderCreate, SearchRequest, SearchResponse, SettingsUpdate,
                      SnoozeRequest, Source)
from .backup import PruneExtra, RestorePayload
from .service import CapacityError, MemoryService, NotFound

log = logging.getLogger("jarvis_memory.routes")
PREFIX = "/api/jarvis/memory"


def service_of(request: Request) -> MemoryService:
    return request.app.state.service


def make_router(guard) -> APIRouter:
    router = APIRouter(prefix=PREFIX, tags=["jarvis-memory"], dependencies=[Depends(guard)])

    @router.post("", response_model=MemoryOut, status_code=201)
    def create_memory(body: MemoryCreate, svc: MemoryService = Depends(service_of)):
        """Save a typed or spoken (transcript) memory, then enrich it with grounded AI extraction."""
        return svc.create(body)

    @router.get("", response_model=list[MemoryOut])
    def list_memories(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0, le=100000),
                      source: Optional[Source] = None, svc: MemoryService = Depends(service_of)):
        return svc.list(limit, offset, source)

    # fixed paths first so "all", "search", "settings" are never read as an id
    @router.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, svc: MemoryService = Depends(service_of)):
        return svc.search(body)

    @router.get("/settings")
    def get_settings(svc: MemoryService = Depends(service_of)):
        return svc.get_settings()

    @router.patch("/settings")
    def patch_settings(body: SettingsUpdate, svc: MemoryService = Depends(service_of)):
        return svc.update_settings(body)

    @router.get("/stats")
    def stats(request: Request, svc: MemoryService = Depends(service_of)):
        """Counts, how much of the free database is used, and how much RAM the service uses."""
        from . import resources

        from .chat import LAST_BRAIN_ERROR

        return {**svc.stats(), **resources.status(request.app.state.settings.ram_limit_mb), "last_ai_error": dict(LAST_BRAIN_ERROR) or None}

    @router.post("/export")
    def export(svc: MemoryService = Depends(service_of)):
        return svc.export()

    @router.delete("/all")
    def delete_all(confirm: bool = Query(False), svc: MemoryService = Depends(service_of)):
        if not confirm:
            raise HTTPException(428, "Deleting every memory requires confirm=true.")
        return {"deleted": svc.delete_all()}

    def chat_of(request: Request):
        from .chat import ChatService

        svc = request.app.state.service
        cached = getattr(request.app.state, "chat_service", None)
        if cached is None or cached.svc is not svc:
            request.app.state.chat_service = cached = ChatService(svc)
        return cached

    @router.post("/chat")
    def chat(body: ChatRequest, request: Request):
        """One turn of the shared chat (phone, desktop and the call screen all use this)."""
        return chat_of(request).chat(body.message, str(body.session_id) if body.session_id else None, context=body.context, archive=body.archive, attachment=body.attachment)

    STT_MESSAGES = {"unsupported": (415, "Send the recording as audio/mp4, wav, webm, ogg or mp3."),
                    "not_configured": (503, "Voice chat needs the Groq key on the server."),
                    "rate_limit": (429, "Speech recognition is busy. Try again in a minute."),
                    "unavailable": (502, "Speech recognition is not available right now.")}

    @router.post("/chat/audio")
    async def chat_audio(request: Request, session_id: Optional[uuid.UUID] = Query(None)):
        """Voice turn for the phone: the raw recording is the request body. It is transcribed, answered by the shared chat (same
        history as typing), and nothing audio is stored. Returns {transcript, reply, session_id, actions}."""
        from starlette.concurrency import run_in_threadpool

        from . import stt

        data = await request.body()
        if not data:
            raise HTTPException(400, "The recording was empty.")
        if len(data) > stt.MAX_AUDIO_BYTES:
            raise HTTPException(413, "That recording is too long.")
        try:
            transcript = await run_in_threadpool(stt.transcribe, request.app.state.settings, data,
                                                 request.headers.get("content-type", ""), getattr(request.app.state, "stt_client", None))
        except stt.SttError as exc:
            code, message = STT_MESSAGES.get(exc.kind, STT_MESSAGES["unavailable"])
            raise HTTPException(code, message) from None
        if not transcript:
            return {"session_id": str(session_id) if session_id else None, "transcript": "", "reply": "", "actions": []}
        turn = await run_in_threadpool(chat_of(request).chat, transcript, str(session_id) if session_id else None)
        return {**turn, "transcript": transcript}

    @router.get("/chat/sessions")
    def chat_sessions(request: Request):
        return chat_of(request).sessions()

    @router.get("/chat/sessions/{session_id}")
    def chat_session(session_id: uuid.UUID, request: Request):
        return chat_of(request).session(str(session_id))

    @router.delete("/chat/sessions/{session_id}", status_code=204)
    def chat_delete(session_id: uuid.UUID, request: Request):
        chat_of(request).delete_session(str(session_id))

    @router.get("/chat/sessions/{session_id}/export.pdf")
    def chat_pdf(session_id: uuid.UUID, request: Request):
        return Response(chat_of(request).export_pdf(str(session_id)), media_type="application/pdf",
                        headers={"Content-Disposition": 'attachment; filename="jarvis-chat.pdf"'})

    @router.get("/chat/sessions/{session_id}/export.md")
    def chat_md(session_id: uuid.UUID, request: Request):
        return Response(chat_of(request).export_markdown(str(session_id)), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="jarvis-chat.md"'})

    @router.get("/analytics")
    def analytics(range: str = Query("week", pattern="^(day|week|month)$"), svc: MemoryService = Depends(service_of)):
        """Numbers for the desktop Insights charts (read-only aggregates)."""
        from . import analytics as analytics_module

        return analytics_module.build(svc, range)

    @router.get("/reviews")
    def reviews_month(month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"), svc: MemoryService = Depends(service_of)):
        """Calendar data: for each day of the month that has anything, its memory count and Daily Review."""
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo

        from . import review

        now = _dt.now(ZoneInfo(svc.settings.timezone))
        year, mon = (int(month[:4]), int(month[5:])) if month else (now.year, now.month)
        return {"month": f"{year:04d}-{mon:02d}", "days": review.month(svc, year, mon)}

    @router.get("/reviews/{day}")
    def review_day(day: date, svc: MemoryService = Depends(service_of)):
        from . import review

        return review.detail(svc, day)

    @router.post("/reviews/{day}/generate")
    def review_generate(day: date, svc: MemoryService = Depends(service_of)):
        """Build (or rebuild) the review for a day right now."""
        from . import review

        review.build(svc, day)
        return review.detail(svc, day)

    @router.post("/mail/review-sent")
    def review_mail_sent(day: date, svc: MemoryService = Depends(service_of)):
        """The scheduler confirms it e-mailed that day's review."""
        svc.mark_review_emailed(day)
        return {"ok": True}

    @router.get("/tasks")
    def open_tasks(svc: MemoryService = Depends(service_of)):
        """Open tasks from every memory (what JARVIS adds by voice or chat), for the desktop Tasks page."""
        return svc.list_open_tasks()

    @router.post("/tasks/done")
    def finish_task(body: TaskRef, svc: MemoryService = Depends(service_of)):
        if not svc.change_task(body.memory_id, body.index, body.task, status="done"):
            raise HTTPException(409, "That task just changed; refresh the list.")
        svc.close_reminders_about(body.task)
        return {"ok": True}

    @router.post("/export/all")
    def export_all(request: Request, svc: MemoryService = Depends(service_of)):
        """Everything (memories, chats with messages, reminders, daily reviews) for the PC archive."""
        from . import backup

        return backup.export_everything(svc)

    @router.post("/archive/restore")
    def archive_restore(body: RestorePayload, svc: MemoryService = Depends(service_of)):
        """Put archived data back (new or empty cloud). Existing rows are never overwritten."""
        from . import backup

        return {"added": backup.restore(svc, body)}

    @router.post("/archive/prune-extra")
    def archive_prune_extra(body: PruneExtra, svc: MemoryService = Depends(service_of)):
        """Remove old chats / finished reminders / old reviews that are safely on the PC."""
        from . import backup

        return {"deleted": backup.prune_extra(svc, body)}

    @router.post("/archive/prune")
    def archive_prune(body: PruneRequest, svc: MemoryService = Depends(service_of)):
        """Permanently remove memories the desktop has already archived to Word/Excel, and give the space back to the database."""
        return {"deleted": svc.prune([str(i) for i in body.ids])}

    @router.post("/ask")
    def ask(body: AskRequest, svc: MemoryService = Depends(service_of)):
        """Answer a question about your own memories ("what did I do yesterday this time?")."""
        from . import ask as ask_module

        return ask_module.answer(svc, body.question)

    @router.post("/tick")
    def tick(request: Request, svc: MemoryService = Depends(service_of)):
        """Called by the scheduler: read e-mail replies, send the hourly check-in if due, report reminders that are due."""
        from . import jobs

        return jobs.run_tick(svc, request.app.state.mailbox, pusher=request.app.state.pusher)

    @router.post("/mail/sent")
    def mail_sent(svc: MemoryService = Depends(service_of)):
        """The scheduler confirms it delivered the check-in the last tick handed it (external SMTP mode)."""
        from datetime import datetime, timezone

        return {"count_today": svc.record_email_sent(datetime.now(timezone.utc))}

    @router.post("/devices", status_code=204)
    def register_device(body: DeviceRegister, svc: MemoryService = Depends(service_of)):
        """The phone app registers its Firebase token so the server can ring it."""
        svc.register_device(body.token)

    @router.post("/push/test")
    def push_test(request: Request, svc: MemoryService = Depends(service_of)):
        """Rings the phone once, right now (for checking that the call alert works)."""
        from . import jobs

        pusher = request.app.state.pusher
        if pusher is None:
            raise HTTPException(503, "Firebase is not configured on the server (FIREBASE_SERVICE_ACCOUNT_JSON).")
        if not svc.device_tokens():
            raise HTTPException(409, "No phone is registered yet. Open the JARVIS app once.")
        ok = jobs._push_all(svc, pusher, {"type": "call", "title": "JARVIS", "emoji": "⚡",
                                          "line": "Test call. If you can hear this, I can reach you."})
        return {"delivered": ok}

    @router.post("/reminders", status_code=201)
    def add_reminder(body: ReminderCreate, svc: MemoryService = Depends(service_of)):
        return svc.add_reminder(body.text, body.remind_at)

    @router.get("/reminders")
    def list_reminders(status: Optional[str] = Query(None, pattern="^(pending|sent|done)$"),
                       svc: MemoryService = Depends(service_of)):
        return svc.list_reminders(status)

    @router.post("/reminders/{reminder_id}/done")
    def reminder_done(reminder_id: uuid.UUID, svc: MemoryService = Depends(service_of)):
        return svc.update_reminder(str(reminder_id), done=True)

    @router.post("/reminders/{reminder_id}/snooze")
    def reminder_snooze(reminder_id: uuid.UUID, body: SnoozeRequest, svc: MemoryService = Depends(service_of)):
        return svc.update_reminder(str(reminder_id), snooze_minutes=body.minutes)

    @router.delete("/reminders/{reminder_id}", status_code=204)
    def reminder_delete(reminder_id: uuid.UUID, svc: MemoryService = Depends(service_of)):
        svc.delete_reminder(str(reminder_id))

    @router.get("/{memory_id}", response_model=MemoryOut)
    def get_memory(memory_id: uuid.UUID, svc: MemoryService = Depends(service_of)):
        return svc.get(str(memory_id))

    @router.patch("/{memory_id}", response_model=MemoryOut)
    def patch_memory(memory_id: uuid.UUID, body: MemoryUpdate, svc: MemoryService = Depends(service_of)):
        return svc.update(str(memory_id), body)

    @router.delete("/{memory_id}", status_code=204)
    def delete_memory(memory_id: uuid.UUID, svc: MemoryService = Depends(service_of)):
        svc.delete(str(memory_id))

    @router.post("/{memory_id}/extract", response_model=MemoryOut)
    def retry_extraction(memory_id: uuid.UUID, svc: MemoryService = Depends(service_of)):
        """Run AI extraction again (after a failure). Fields you edited are never overwritten."""
        return svc.re_extract(str(memory_id))

    return router


def install_error_handlers(app) -> None:
    from fastapi.responses import JSONResponse

    @app.exception_handler(NotFound)
    async def _nf(_req, _exc):
        return JSONResponse({"detail": "Memory not found."}, status_code=404)

    @app.exception_handler(CapacityError)
    async def _cap(_req, exc):
        return JSONResponse({"detail": str(exc)}, status_code=507)

    @app.exception_handler(DatabaseNotConfigured)
    async def _nodb(_req, _exc):
        return JSONResponse({"detail": "The database is not configured."}, status_code=503)

    @app.exception_handler(psycopg.OperationalError)
    async def _dbdown(_req, exc):
        log.warning("Database unavailable: %s", type(exc).__name__)  # never log SQL or content
        return JSONResponse({"detail": "The database is temporarily unavailable. Your request was not saved; try again."},
                            status_code=503, headers={"Retry-After": "5"})
