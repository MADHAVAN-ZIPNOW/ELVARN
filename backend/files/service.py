"""FileService: upload, convert, download, and the chat hook. Used by the API routes and by /api/chat."""
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from . import intent as intent_mod
from . import readers, writers
from .convert import LABELS, MIME, PREVIEW_MAX_CHARS, convert, supported_targets
from .errors import Busy, FileError, NotFound
from .sniff import clean_filename, decode_text, detect
from .store import FileConfig, FileStore

log = logging.getLogger("elvarn.files")

EDIT_MESSAGE = ("I can only change a file's format, for example TXT to PDF or CSV to JSON. I cannot summarise, translate, "
                "rewrite or explain what is inside it: ELVARN is a very small model and it does not read uploaded files. "
                "Try asking: \"convert this to pdf\".")
NO_FILE_MESSAGE = ("I do not see an attached file. Use the paperclip button to upload a file (txt, md, csv, json, html or docx), "
                   "then ask again, for example \"convert this to pdf\".")
EXPIRED_MESSAGE = "That file is no longer stored (uploads are deleted after a short time). Please upload it again."
LOSS_NOTE = ("{n} character(s) could not be drawn in the PDF's basic font (it supports English and Western European text) "
             "and were replaced with '?'.")


def human_size(n):
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB" if n >= 10 * 1024 else f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


def output_name(upload_name, target):
    stem = upload_name.rpartition(".")[0] or upload_name
    return clean_filename(stem or "converted") + "." + target


class FileService:
    def __init__(self, config=None, store=None):
        self.cfg = config or FileConfig.from_env()
        self.store = store or FileStore(self.cfg)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="convert")
        self.slots = threading.BoundedSemaphore(2)

    # ---- upload ----
    def upload(self, name, data, owner):
        started = time.monotonic()
        clean = clean_filename(name)
        kind = detect(clean, data)
        entry = self.store.put(data, kind="upload", name=clean, fmt=kind, owner=owner)
        log.info("upload ok type=%s bytes=%d seconds=%.2f", kind, len(data), time.monotonic() - started)
        return {"file_id": entry.id, "name": entry.name, "size": entry.size, "detected_type": kind,
                "supported_targets": supported_targets(kind), "expires_in": self.store.seconds_left(entry)}

    # ---- convert ----
    def _run_conversion(self, source, data, target, name):
        if not self.slots.acquire(blocking=False):
            raise Busy("The converter is busy. Please try again in a moment.")

        def job():
            try:
                return convert(source, data, target, name, self.cfg.max_output_bytes)
            finally:
                self.slots.release()

        future = self.pool.submit(job)
        try:
            return future.result(timeout=self.cfg.convert_timeout)
        except FutureTimeout:
            raise Busy("The conversion took too long and was stopped.")

    def convert_upload(self, file_id, target, owner):
        started = time.monotonic()
        entry = self.store.get(file_id, "upload")
        target = str(target or "").lower().strip().lstrip(".")
        existing = entry.derived.get(target)
        if existing:
            try:
                return self._download_info(self.store.get(existing, "download"), entry), 0
            except NotFound:
                entry.derived.pop(target, None)
        data = self.store.read(entry)
        try:
            result = self._run_conversion(entry.fmt, data, target, entry.name)
        except FileError:
            log.info("convert failed %s->%s", entry.fmt, target)
            raise
        except Exception:
            log.exception("convert crashed %s->%s", entry.fmt, target)
            raise FileError("This file could not be converted (it may be damaged or unusual).")
        out = self.store.put(result.data, kind="download", name=output_name(entry.name, target), fmt=target, owner=owner,
                             source_id=entry.id, preview=result.preview)
        entry.derived[target] = out.id
        log.info("convert ok %s->%s bytes=%d seconds=%.2f", entry.fmt, target, out.size, time.monotonic() - started)
        return self._download_info(out, entry), result.lost_chars

    def _download_info(self, out, src):
        return {"download_id": out.id, "name": out.name, "size": out.size, "format": out.fmt, "label": LABELS[out.fmt],
                "source_name": src.name, "download_url": f"/api/files/{out.id}/download",
                "expires_in": self.store.seconds_left(out)}

    # ---- download ----
    def download(self, download_id):
        entry = self.store.get(download_id, "download")
        return entry, self.store.read(entry), MIME[entry.fmt]

    # ---- preview (text only, capped) ----
    def preview(self, file_id):
        """Works for upload ids and download ids. Returns plain text for the browser to show with textContent."""
        entry = self.store.get(file_id)
        note = ""
        if entry.kind == "download" and entry.fmt in ("docx", "pdf"):
            text = entry.preview
            note = "Preview shows the text only (no layout, fonts or images)."
        else:
            data = self.store.read(entry)
            if entry.fmt == "docx":
                try:
                    text = writers.to_txt(readers.read("docx", data)).decode("utf-8", "replace")
                except FileError:
                    raise
                except Exception:
                    raise FileError("This file could not be previewed (it may be damaged or unusual).")
                note = "Preview shows the text only (no layout, fonts or images)."
            else:
                text = decode_text(data)
        truncated = len(text) > PREVIEW_MAX_CHARS
        return {"name": entry.name, "format": entry.fmt, "size": entry.size, "text": text[:PREVIEW_MAX_CHARS],
                "truncated": truncated, "preview_note": note, "kind": entry.kind,
                "label": LABELS.get(entry.fmt, entry.fmt.upper())}

    def delete(self, file_id):
        return self.store.delete(file_id)

    # ---- chat hook ----
    def handle_chat(self, text, file_ids, owner):
        """None when the message is not a file request (normal chat continues). Otherwise a dict with the fixed
        reply text and the files; the language model is never involved."""
        ids = [i for i in (file_ids or []) if isinstance(i, str)]
        found = None
        for fid in reversed(ids):
            try:
                found = self.store.get(fid, "upload")
                break
            except NotFound:
                continue
        it = intent_mod.parse(text, bool(ids))
        if it.kind == "none":
            return None
        if it.kind == "edit":
            return {"text": EDIT_MESSAGE, "files": [], "file_action": "refused_edit"}
        if it.kind == "no_file":
            return {"text": NO_FILE_MESSAGE, "files": [], "file_action": "no_file"}
        if found is None:
            return {"text": EXPIRED_MESSAGE, "files": [], "file_action": "expired"}
        if it.kind == "unknown":
            opts = ", ".join(LABELS[t] for t in supported_targets(found.fmt))
            return {"text": f"I cannot make .{it.word} files. For {found.name} I can make: {opts}.",
                    "files": [], "file_action": "unknown_target"}
        try:
            info, lost = self.convert_upload(found.id, it.target, owner)
        except FileError as exc:
            return {"text": exc.message, "files": [], "file_action": "convert_failed"}
        msg = f"Converted {found.name} to {info['label']} ({human_size(info['size'])}). Only the text is kept, not layout or images."
        if lost:
            msg += " " + LOSS_NOTE.format(n=lost)
        return {"text": msg, "files": [info], "file_action": "converted"}
