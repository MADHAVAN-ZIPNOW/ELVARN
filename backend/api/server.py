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

from files.routes import install_file_routes  # noqa: E402
from files.service import FileService  # noqa: E402
from api.model_service import ModelNotReady, ModelService  # noqa: E402
from model.generation import MAX_NO_REPEAT_NGRAM, MAX_REPETITION_PENALTY, MODES  # noqa: E402
from search.orchestrate import WEB_SEARCH_MODES, plan_search, run_search  # noqa: E402
from search.providers.searxng import SearxngProvider, is_loopback_host, wake_up  # noqa: E402
from search.service import SearchService  # noqa: E402

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
    # "off" (default, so old clients behave exactly as before) or "auto" (search the web when the question needs it)
    web_search: str = "off"
    # Repetition controls. None = "not set by the client": off normally, WEB_REPETITION defaults when web evidence is in the prompt.
    # ids returned by POST /api/files; lets the server answer "convert this to pdf" itself (the model is not used for that)
    file_ids: Optional[List[str]] = Field(default=None, max_length=5)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=MAX_REPETITION_PENALTY)
    no_repeat_ngram_size: Optional[int] = Field(default=None, ge=0, le=MAX_NO_REPEAT_NGRAM)


# Used only when the answer is built from web notes (small models tend to loop there) and the client set nothing.
WEB_REPETITION = {"repetition_penalty": 1.15, "no_repeat_ngram_size": 4}


def repetition_settings(req, evidence_in_prompt):
    """(settings for generate, where they came from). Client values win; the web defaults apply only when evidence
    is really in the prompt; otherwise everything stays off (the old behaviour)."""
    web = evidence_in_prompt > 0
    base = WEB_REPETITION if web else {"repetition_penalty": 1.0, "no_repeat_ngram_size": 0}
    settings = {
        "repetition_penalty": base["repetition_penalty"] if req.repetition_penalty is None else req.repetition_penalty,
        "no_repeat_ngram_size": base["no_repeat_ngram_size"] if req.no_repeat_ngram_size is None else req.no_repeat_ngram_size,
    }
    client = req.repetition_penalty is not None or req.no_repeat_ngram_size is not None
    return settings, ("client" if client else "web_default" if web else "off")


# When the answer is written from web notes, random sampling (temperature 0.8, top-k 40-65) makes the small model drift away
# from the notes and swap in details from memory. In that case the sliders are capped (lower numbers are kept as they are).
WEB_SAMPLING_CAP = {"temperature": 0.4, "top_k": 20}


def sampling_settings(req, evidence_in_prompt):
    """(temperature, top_k, where they came from). Greedy decoding is never changed; without notes in the prompt nothing changes."""
    if evidence_in_prompt > 0 and not req.greedy:
        t = min(req.temperature, WEB_SAMPLING_CAP["temperature"])
        k = req.top_k if req.top_k == 0 else min(req.top_k, WEB_SAMPLING_CAP["top_k"])
        k = WEB_SAMPLING_CAP["top_k"] if req.top_k == 0 else k          # top_k 0 means "no limit": cap it too
        return t, k, ("web_cap" if (t, k) != (req.temperature, req.top_k) else "client")
    return req.temperature, req.top_k, "client"


GENERATION_PATHS = ("/api/chat", "/api/chat/stream", "/api/generate")


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def create_app(checkpoint_dir=None, load_on_start=True, public=None, rate_limit_per_minute=None, max_concurrent=None, search_service=None, file_service=None):
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
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-File-Name"],
    )
    service = ModelService(checkpoint_dir)
    app.state.service = service
    app.state.public = public
    search = search_service if search_service is not None else SearchService.from_env()
    app.state.search = search
    # A remote search service on Render's free plan sleeps when idle: wake it in the background
    # (never blocks start-up, ignores errors). Not done for localhost, injected services (tests) or when switched off.
    provider = getattr(search, "provider", None)
    if (search_service is None and isinstance(provider, SearxngProvider) and not is_loopback_host(provider.base_url)
            and os.environ.get("ELVARN_SEARCH_WAKEUP", "1") != "0"):
        wake_up(provider.base_url)
    files = file_service if file_service is not None else FileService()
    app.state.files = files

    # ---- per-visitor rate limit (in memory; one server process) ----
    hits = defaultdict(deque)
    hits_lock = threading.Lock()

    def visitor(request: Request):
        forwarded = request.headers.get("x-forwarded-for", "")
        return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")

    def take_hit(who):
        """Record one use for this visitor. Returns None if allowed, else the seconds to wait."""
        if rate_limit_per_minute <= 0:
            return None
        now = time.monotonic()
        with hits_lock:
            q = hits[who]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= rate_limit_per_minute:
                return max(1, int(60 - (now - q[0])))
            q.append(now)
        return None

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        path = request.url.path
        limited = ((request.method == "POST" and (path in GENERATION_PATHS or path == "/api/files" or path.endswith("/convert")
                                                   and path.startswith("/api/files/")))
                   or (request.method == "GET" and path.startswith("/api/files/") and (path.endswith("/download") or path.endswith("/preview"))))
        if limited:
            wait = take_hit(visitor(request))
            if wait is not None:
                return JSONResponse({"detail": f"Too many requests. Please wait {wait} seconds and try again."},
                                    status_code=429, headers={"Retry-After": str(wait)})
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

    def check_web_search(req):
        if req.web_search not in WEB_SEARCH_MODES:
            raise HTTPException(422, f"web_search must be one of {list(WEB_SEARCH_MODES)}")

    def search_step(plan, req, request):
        """Run the planned web search. A search counts against the visitor's rate limit like a chat request,
        and it never raises: if it cannot run, the chat simply continues without it."""
        allowed = True
        if plan.query and rate_limit_per_minute > 0:
            allowed = take_hit(visitor(request)) is None
        return run_search(plan, req.text, search, service, req.max_new_tokens, allowed)

    def model_input(req, found, history):
        """Text and history the model sees. With evidence, earlier turns are dropped: a small context window cannot hold both."""
        if found.model_text:
            return found.model_text, None
        return req.text, history

    def file_turn(req, request):
        """A conversion request for an attached file is answered by plain code with a fixed message (no model)."""
        if not req.file_ids:
            return None
        return files.handle_chat(req.text, req.file_ids, visitor(request))

    def file_result(handled, req):
        return {"text": handled["text"], "raw_text": handled["text"], "mode": req.mode, "files": handled["files"],
                "file_action": handled["file_action"], "model_used": False, "generated_tokens": 0, "prompt_tokens": 0,
                "seconds": 0, "stop_reason": "file_action"}

    def run(req: GenerateRequest, request: Request):
        if req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")
        if not req.text.strip():
            raise HTTPException(422, "text must not be blank")
        handled = file_turn(req, request)
        if handled is not None:
            return file_result(handled, req)
        check_web_search(req)
        history = [(t.user, t.assistant) for t in req.history] if req.history else None
        take_slot()
        try:
            plan = plan_search(req.text, history, req.mode, req.web_search)
            found = search_step(plan, req, request)
            text, hist = model_input(req, found, history)
            rep, rep_source = repetition_settings(req, found.evidence_in_prompt)
            temp, topk, sampling_source = sampling_settings(req, found.evidence_in_prompt)
            result = service.generate(
                text, mode=req.mode, max_new_tokens=req.max_new_tokens,
                temperature=temp, top_k=topk,
                greedy=req.greedy, history=hist, **rep,
            )
            result.update(found.fields())
            result["repetition_source"] = rep_source
            result["sampling_source"] = sampling_source
            return result
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
    def chat_stream(req: GenerateRequest, request: Request):
        """Newline-delimited JSON events: start, token..., done (or error)."""
        if req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")
        if not req.text.strip():
            raise HTTPException(422, "text must not be blank")
        handled = file_turn(req, request)
        if handled is not None:
            result = file_result(handled, req)

            def file_events():
                if handled["files"]:
                    yield json.dumps({"event": "file", "files": handled["files"]}) + "\n"
                yield json.dumps({"event": "done", **result}) + "\n"

            return StreamingResponse(file_events(), media_type="application/x-ndjson",
                                     headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        if not service.ready:
            raise HTTPException(503, service.last_error or "Model is not loaded. Train ELVARN first.")
        check_web_search(req)
        history = [(t.user, t.assistant) for t in req.history] if req.history else None
        take_slot()

        def events():
            try:
                plan = plan_search(req.text, history, req.mode, req.web_search)
                if plan.query:
                    yield json.dumps({"event": "status", "status": "searching",
                                      "message": "Searching the web..."}) + "\n"
                found = search_step(plan, req, request)
                if plan.query:
                    yield json.dumps({"event": "sources", **found.fields()}) + "\n"
                text, hist = model_input(req, found, history)
                rep, rep_source = repetition_settings(req, found.evidence_in_prompt)
                temp, topk, sampling_source = sampling_settings(req, found.evidence_in_prompt)
                for event in service.generate_stream(
                    text, mode=req.mode, max_new_tokens=req.max_new_tokens,
                    temperature=temp, top_k=topk,
                    greedy=req.greedy, history=hist, **rep,
                ):
                    if event["event"] == "done":
                        event.update(found.fields())
                        event["repetition_source"] = rep_source
                        event["sampling_source"] = sampling_source
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

    install_file_routes(app, files, visitor)

    @app.post("/api/chat")
    def chat(req: GenerateRequest, request: Request):
        return run(req, request)

    @app.post("/api/generate")
    def generate_endpoint(req: GenerateRequest, request: Request):
        return run(req, request)

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
