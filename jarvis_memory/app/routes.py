"""HTTP API (all under /api/jarvis/memory; the prefix avoids ELVARN's existing /api/memory route)."""
from __future__ import annotations

import logging
import uuid
from typing import Optional

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from .db import DatabaseNotConfigured
from .schemas import (AskRequest, ChatRequest, DeviceRegister, MemoryCreate, MemoryOut, MemoryUpdate, ReminderCreate, SearchRequest, SearchResponse, SettingsUpdate,
                      SnoozeRequest, Source)
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

        return {**svc.stats(), **resources.status(request.app.state.settings.ram_limit_mb)}

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
        return chat_of(request).chat(body.message, str(body.session_id) if body.session_id else None)

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
