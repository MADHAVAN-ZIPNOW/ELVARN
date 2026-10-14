"""HTTP endpoints for uploads, conversion and downloads.

POST   /api/files                      raw body upload (the file bytes), name in the X-File-Name header (percent-encoded)
POST   /api/files/{file_id}/convert    {"target": "pdf"}
GET    /api/files/{file_id}/preview    text preview (upload or download id), capped, JSON
GET    /api/files/{download_id}/download
DELETE /api/files/{file_id}

The upload is the raw request body (not multipart): this needs no extra dependency and lets us stop reading the
moment the size limit is passed.
"""
import re
from urllib.parse import quote, unquote

from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from typing import List

from .convert import EXT, MIME, convert
from .errors import FileError, TooLarge
from .sniff import ACCEPT_ATTR

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


class ConvertRequest(BaseModel):
    target: str = Field(min_length=1, max_length=20)


class ExportMessage(BaseModel):
    role: str = Field(max_length=10)
    text: str = Field(default="", max_length=20000)
    sources: List[dict] = Field(default_factory=list, max_length=12)


class ExportRequest(BaseModel):
    title: str = Field(default="ELVARN chat", max_length=120)
    format: str = Field(pattern="^(pdf|docx|txt)$")
    messages: List[ExportMessage] = Field(min_length=1, max_length=200)


def export_markdown(req):
    """The chat as Markdown, which the existing converters turn into Word / PDF / plain text."""
    out = [f"# {req.title.strip() or 'ELVARN chat'}", ""]
    for m in req.messages:
        text = m.text.strip()
        if not text:
            continue
        out += ["## You" if m.role == "user" else "## ELVARN", "", text, ""]
        links = [(str(x.get("title") or x.get("domain") or x.get("url"))[:120], str(x.get("url") or "")[:300]) for x in m.sources if x.get("url")]
        if links:
            out += ["Sources:", ""] + [f"- {t}: {u}" for t, u in links] + [""]
    return "\n".join(out)


def content_disposition(name, fmt):
    """attachment with an ASCII fallback plus the RFC 5987 UTF-8 form."""
    ascii_name = re.sub(r'[^A-Za-z0-9._ -]', "_", name) or ("download." + EXT[fmt])
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name, safe='')}"


def fail(exc):
    raise HTTPException(exc.status, exc.message)


def install_file_routes(app, service, visitor):
    cfg = service.cfg

    def valid_id(value):
        if not ID_PATTERN.match(value or ""):
            raise HTTPException(404, "That file was not found. It may have expired.")

    @app.get("/api/files/limits")
    def limits():
        return {"max_upload_bytes": cfg.max_upload_bytes, "ttl_seconds": cfg.ttl_seconds, "accept": ACCEPT_ATTR}

    @app.post("/api/files")
    async def upload(request: Request):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > cfg.max_upload_bytes:
            raise HTTPException(413, f"This file is too large. The limit is {cfg.max_upload_bytes // (1024 * 1024)} MB.")
        raw_name = unquote(request.headers.get("x-file-name", "") or request.query_params.get("name", ""))
        chunks, total = [], 0
        async for chunk in request.stream():       # stop reading as soon as the limit is passed
            total += len(chunk)
            if total > cfg.max_upload_bytes:
                raise HTTPException(413, f"This file is too large. The limit is {cfg.max_upload_bytes // (1024 * 1024)} MB.")
            chunks.append(chunk)
        data = b"".join(chunks)
        try:
            return await run_in_threadpool(service.upload, raw_name, data, visitor(request))
        except FileError as exc:
            fail(exc)

    @app.post("/api/files/{file_id}/convert")
    def convert_endpoint(file_id: str, req: ConvertRequest, request: Request):
        valid_id(file_id)
        try:
            info, lost = service.convert_upload(file_id, req.target, visitor(request))
        except FileError as exc:
            fail(exc)
        return {**info, "lost_characters": lost}

    @app.get("/api/files/{download_id}/download")
    def download(download_id: str):
        valid_id(download_id)
        try:
            entry, data, mime = service.download(download_id)
        except FileError as exc:
            fail(exc)
        return Response(content=data, media_type=mime, headers={
            "Content-Disposition": content_disposition(entry.name, entry.fmt),
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "Content-Length": str(len(data)),
        })

    @app.get("/api/files/{file_id}/preview")
    def preview(file_id: str):
        valid_id(file_id)
        try:
            body = service.preview(file_id)
        except FileError as exc:
            fail(exc)
        return JSONResponse(body, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.post("/api/export")
    async def export_chat(req: ExportRequest):
        md = export_markdown(req)
        if len(md) > 400_000:
            raise HTTPException(413, "This chat is too long to export in one file.")
        try:
            result = await run_in_threadpool(convert, "md", md.encode("utf-8"), req.format, req.title)
        except FileError as exc:
            fail(exc)
        name = (re.sub(r"[^\w .-]+", "_", req.title, flags=re.U).strip(" ._") or "ELVARN chat")[:60] + "." + EXT[req.format]
        return Response(content=result.data, media_type=MIME[req.format], headers={
            "Content-Disposition": content_disposition(name, req.format), "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})

    @app.delete("/api/files/{file_id}")
    def delete(file_id: str):
        valid_id(file_id)
        return {"deleted": service.delete(file_id)}

    return app
