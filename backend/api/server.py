"""ELVARN local API + web UI.

Run from the project root:
    python -m uvicorn backend.api.server:app --port 8000
then open http://127.0.0.1:8000
"""
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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


def create_app(checkpoint_dir=None, load_on_start=True):
    app = FastAPI(title="ELVARN", docs_url="/api/docs", redoc_url=None)
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
        try:
            return service.generate(
                req.text, mode=req.mode, max_new_tokens=req.max_new_tokens,
                temperature=req.temperature, top_k=req.top_k,
                greedy=req.greedy, history=history,
            )
        except ModelNotReady as exc:
            raise HTTPException(503, str(exc))

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
