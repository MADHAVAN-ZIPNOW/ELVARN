"""ELVARN local API + web UI.

Run from the project root:
    python -m uvicorn backend.api.server:app --port 8000
then open http://127.0.0.1:8000
"""
import json
import os
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from api.model_service import ModelNotReady, ModelService  # noqa: E402
from model.generation import MODES  # noqa: E402

FRONTEND_DIR = BACKEND_DIR.parent / "frontend"


class Turn(BaseModel):
    user: str = Field(max_length=1000)
    assistant: str = Field(max_length=2000)


class GenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    mode: str = "chat"
    max_new_tokens: int = Field(default=80, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=0, le=500)
    greedy: bool = False
    history: Optional[List[Turn]] = Field(default=None, max_length=6)


GENERATION_PATHS = ("/api/chat", "/api/chat/stream", "/api/generate")


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def create_app(checkpoint_dir=None, load_on_start=True, public=None, rate_limit_per_minute=None, max_concurrent=None):
    """public=True (or ELVARN_PUBLIC=1) is for a website anyone can reach: it turns on a per-visitor rate limit,
    a cap on simultaneous generations, and disables the reload endpoint."""
    public = env_flag("ELVARN_PUBLIC") if public is None else public
    if rate_limit_per_minute is None:
        rate_limit_per_minute = int(os.environ.get("ELVARN_RATE_LIMIT", "20" if public else "0"))
    if max_concurrent is None:
        max_concurrent = int(os.environ.get("ELVARN_MAX_CONCURRENT", "2" if public else "0"))
    app = FastAPI(title="ELVARN", docs_url=None, redoc_url=None)
    # Lets the UI work when opened from another local server such as
    # VS Code Live Server (127.0.0.1:5500). Local origins only.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    service = ModelService(checkpoint_dir)
    app.state.service = service
    app.state.public = public

    # ---- per-visitor rate limit (in memory; one server process) ----
    hits = defaultdict(deque)
    hits_lock = threading.Lock()

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        if rate_limit_per_minute > 0 and request.method == "POST" and request.url.path in GENERATION_PATHS:
            forwarded = request.headers.get("x-forwarded-for", "")
            who = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
            now = time.monotonic()
            with hits_lock:
                q = hits[who]
                while q and now - q[0] > 60:
                    q.popleft()
                if len(q) >= rate_limit_per_minute:
                    wait = max(1, int(60 - (now - q[0])))
                    return JSONResponse({"detail": f"Too many requests. Please wait {wait} seconds and try again."},
                                        status_code=429, headers={"Retry-After": str(wait)})
                q.append(now)
        return await call_next(request)

    # ---- cap on simultaneous generations (the CPU is small) ----
    slots = threading.BoundedSemaphore(max_concurrent) if max_concurrent > 0 else None

    def take_slot():
        if slots is not None and not slots.acquire(blocking=False):
            raise HTTPException(429, "ELVARN is busy answering other visitors. Please try again in a moment.")

    def give_slot():
        if slots is not None:
            slots.release()

    if load_on_start:
        try:
            service.load()
        except ModelNotReady:
            pass  # reported by /api/health; UI shows the message

    def run(req: GenerateRequest):
        if req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")
        if not req.text.strip():
            raise HTTPException(422, "text must not be blank")
        history = [(t.user, t.assistant) for t in req.history] if req.history else None
        take_slot()
        try:
            return service.generate(
                req.text, mode=req.mode, max_new_tokens=req.max_new_tokens,
                temperature=req.temperature, top_k=req.top_k,
                greedy=req.greedy, history=history,
            )
        except ModelNotReady as exc:
            raise HTTPException(503, str(exc))
        finally:
            give_slot()

    # the API docs page gets the ELVARN logo as its tab icon too
    @app.get("/api/docs", include_in_schema=False)
    def docs():
        return get_swagger_ui_html(openapi_url="/openapi.json", title="ELVARN API", swagger_favicon_url="/favicon.png")

    @app.get("/favicon.png", include_in_schema=False)
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        logo = FRONTEND_DIR / "logo.png"
        if not logo.is_file():
            raise HTTPException(404, "logo not found")
        return FileResponse(logo, media_type="image/png")

    @app.post("/api/chat/stream")
    def chat_stream(req: GenerateRequest):
        """Newline-delimited JSON events: start, token..., done (or error)."""
        if req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")
        if not req.text.strip():
            raise HTTPException(422, "text must not be blank")
        if not service.ready:
            raise HTTPException(503, service.last_error or "Model is not loaded. Train ELVARN first.")
        history = [(t.user, t.assistant) for t in req.history] if req.history else None
        take_slot()

        def events():
            try:
                for event in service.generate_stream(
                    req.text, mode=req.mode, max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature, top_k=req.top_k,
                    greedy=req.greedy, history=history,
                ):
                    yield json.dumps(event) + "\n"
            except Exception as exc:  # reported to the UI instead of a broken stream
                yield json.dumps({"event": "error", "message": str(exc)}) + "\n"
            finally:
                give_slot()

        return StreamingResponse(events(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/api/health")
    def health():
        return service.health()

    @app.get("/api/model")
    def model_info():
        info = service.info()
        if info is None:
            raise HTTPException(503, service.last_error or "Model not loaded.")
        return {**info, "modes": list(MODES)}

    @app.post("/api/chat")
    def chat(req: GenerateRequest):
        return run(req)

    @app.post("/api/generate")
    def generate_endpoint(req: GenerateRequest):
        return run(req)

    @app.post("/api/reload")
    def reload_model():
        if public:
            raise HTTPException(403, "Reloading is disabled on the public site.")
        try:
            return {"reloaded": True, "model": service.load()}
        except ModelNotReady as exc:
            raise HTTPException(503, str(exc))

    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        @app.get("/")
        def index():
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()
