"""Pictures for e-mails the server sends itself (Brevo): the e-mail links to them, because the HTTPS mail API cannot attach inline pictures
the way the scheduler's SMTP did. The rocket is a fixed file; each update's 3D chart is kept in mail_outbox (already marked sent, so the
scheduler ignores it) and served from an address nobody can guess: its id plus an HMAC made with the owner key."""
from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, Response

ASSETS = Path(__file__).parent / "assets"
CHART_MARK = "(chart picture)"      # to_addr of the rows that only hold a chart picture
PATH = "/mail-img"                  # outside /api, so no API guard and normal caching


def _sig(secret: str, chart_id: int) -> str:
    return hmac.new(secret.encode(), f"chart:{chart_id}".encode(), hashlib.sha256).hexdigest()[:24]


def _base(settings: Any) -> str:
    return (getattr(settings, "public_url", "") or "").rstrip("/")


def store_chart(svc: Any, settings: Any, png: bytes) -> Optional[str]:
    """Keep the chart and return its public address, or None when the server does not know its own address."""
    if not _base(settings) or not settings.owner_key:
        return None
    with svc.db() as conn:
        row = conn.execute("INSERT INTO mail_outbox (owner_id, to_addr, subject, body_text, chart_png, sent_at) VALUES (%s,%s,%s,%s,%s, now()) RETURNING id",
                           (svc.owner_id, CHART_MARK, "chart", "", png)).fetchone()
    cid = int(row["id"])
    return f"{_base(settings)}{PATH}/chart/{cid}-{_sig(settings.owner_key, cid)}.png"


def link_pictures(page: str, settings: Any, chart_url: Optional[str]) -> str:
    """Swap the attachment references (cid:...) of the team-update page for the hosted pictures."""
    if _base(settings):
        page = page.replace("cid:rocket", f"{_base(settings)}{PATH}/rocket.gif")
    if chart_url:
        page = page.replace("cid:chart3d", chart_url)
    return page


def make_router() -> APIRouter:
    router = APIRouter(prefix=PATH, tags=["mail-images"])

    @router.get("/rocket.gif")
    def rocket():
        data = (ASSETS / "rocket.gif").read_bytes()
        return Response(data, media_type="image/gif", headers={"Cache-Control": "public, max-age=604800"})

    @router.get("/chart/{token}.png")
    def chart(token: str, request: Request):
        settings = request.app.state.settings
        try:
            raw_id, sig = token.split("-", 1)
            cid = int(raw_id)
        except ValueError:
            raise HTTPException(status_code=404)
        if not settings.owner_key or not hmac.compare_digest(sig, _sig(settings.owner_key, cid)):
            raise HTTPException(status_code=404)
        with request.app.state.service.db() as conn:
            row = conn.execute("SELECT chart_png FROM mail_outbox WHERE id = %s AND to_addr = %s", (cid, CHART_MARK)).fetchone()
        if not row or not row["chart_png"]:
            raise HTTPException(status_code=404)
        return Response(bytes(row["chart_png"]), media_type="image/png", headers={"Cache-Control": "public, max-age=604800"})

    return router
