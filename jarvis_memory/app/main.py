"""Jarvis Memory service entry point: ``uvicorn app.main:app`` (run from the jarvis_memory folder)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .db import connect, run_migrations
from .extraction import Extractor
from .routes import install_error_handlers, make_router
from .security import Guard
from .service import MemoryService

log = logging.getLogger("jarvis_memory")
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
MAX_BODY_BYTES = 250_000  # a memory is text; refuse anything larger before reading it


def create_app(settings: Optional[Settings] = None, extractor: Optional[Extractor] = None, migrate: bool = True) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if migrate and settings.database_url:
            try:
                applied = run_migrations(settings.database_url)
                if applied:
                    log.info("Applied migrations: %s", applied)
            except Exception as exc:  # the API still starts and reports the problem on /health
                log.error("Migrations failed: %s", type(exc).__name__)
        yield

    app = FastAPI(title="Jarvis Memory", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.service = MemoryService(settings, extractor)

    if settings.allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                           allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Content-Type", "X-Owner-Key"])

    @app.middleware("http")
    async def hardening(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request too large."}, status_code=413)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store" if request.url.path.startswith("/api/") else "no-cache")
        return response

    install_error_handlers(app)
    app.include_router(make_router(Guard(settings, settings.rate_limit_per_minute)))

    @app.get("/api/jarvis/health")
    def health():
        """Public, no data: is the service up, is the database reachable, is AI extraction configured."""
        db_ok = False
        if settings.database_url:
            try:
                with connect(settings.database_url) as conn:
                    conn.execute("SELECT 1")
                db_ok = True
            except Exception:
                db_ok = False
        from . import resources

        ram = resources.status(settings.ram_limit_mb)["ram_status"]
        return {"status": "ok" if db_ok and ram != "over" else "degraded", "database": db_ok,
                "ai_extraction": settings.extraction_enabled, "configured": len(settings.owner_key) >= 16, "memory": ram}

    if WEB_DIST.is_dir():  # the dashboard, built by `npm run build` in ./web
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


def run() -> None:  # pragma: no cover
    import os

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8100")))


if __name__ == "__main__":  # pragma: no cover
    run()
