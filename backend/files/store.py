"""Temporary file store.

* Files live on local disk under a server-generated random name (never the user's filename).
* The index (id -> details) is in memory, so after a restart old files are unreachable and are deleted at startup.
* ids come from secrets.token_urlsafe, so they cannot be guessed. There is no listing endpoint.
* Everything expires (default 1 hour) and is removed on access and on every new write.
"""
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .errors import Busy, LimitReached, NotFound

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DIR = BACKEND_DIR / "file_store"
_DISK_NAME = re.compile(r"^[0-9a-f]{32}\.bin$")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


@dataclass
class FileConfig:
    root: Path = DEFAULT_DIR
    ttl_seconds: int = 3600
    max_upload_bytes: int = 5 * 1024 * 1024
    max_store_bytes: int = 20 * 1024 * 1024
    max_per_visitor: int = 20
    max_items: int = 200
    convert_timeout: float = 15.0

    @classmethod
    def from_env(cls, **override):
        cfg = cls(
            root=Path(os.environ.get("ELVARN_FILE_STORE") or DEFAULT_DIR),
            ttl_seconds=max(30, _env_int("ELVARN_FILE_TTL_SECONDS", 3600)),
            max_upload_bytes=max(1024, _env_int("ELVARN_MAX_UPLOAD_MB", 5) * 1024 * 1024),
            max_store_bytes=max(1024, _env_int("ELVARN_MAX_STORE_MB", 20) * 1024 * 1024),
            max_per_visitor=max(1, _env_int("ELVARN_MAX_FILES_PER_VISITOR", 20)),
        )
        for key, value in override.items():
            setattr(cfg, key, value)
        cfg.root = Path(cfg.root)
        return cfg

    @property
    def max_output_bytes(self):
        return min(self.max_upload_bytes * 2, self.max_store_bytes // 2)


@dataclass
class Entry:
    id: str
    kind: str            # "upload" or "download"
    disk: str            # server-generated file name
    name: str            # cleaned display name
    fmt: str             # type name: txt, pdf, ...
    size: int
    owner: str
    expires: float
    source_id: str = ""
    derived: dict = field(default_factory=dict)   # target -> download id (reuse)
    preview: str = ""    # extracted text kept for docx/pdf outputs (their bytes are not text)


class FileStore:
    def __init__(self, config=None, clock=time.time):
        self.cfg = config or FileConfig.from_env()
        self.clock = clock
        self.lock = threading.RLock()
        self.items = {}
        self.order = []
        self._clean_leftovers()

    # ---- disk helpers ----
    def _path(self, entry):
        return self.cfg.root / entry.disk

    def _clean_leftovers(self):
        """Files from an earlier run have no index entry, so nobody can reach them: remove them."""
        try:
            if self.cfg.root.is_dir():
                for p in self.cfg.root.iterdir():
                    if p.is_file() and _DISK_NAME.match(p.name):
                        try:
                            p.unlink()
                        except OSError:
                            pass
        except OSError:
            pass

    def _drop(self, entry):
        self.items.pop(entry.id, None)
        try:
            self._path(entry).unlink()
        except OSError:
            pass
        if entry.source_id in self.items:
            src = self.items[entry.source_id]
            for key in [k for k, v in src.derived.items() if v == entry.id]:
                del src.derived[key]

    def purge(self):
        now = self.clock()
        with self.lock:
            for entry in [e for e in self.items.values() if e.expires <= now]:
                self._drop(entry)

    def total_bytes(self):
        return sum(e.size for e in self.items.values())

    # ---- public API ----
    def put(self, data, *, kind, name, fmt, owner, source_id="", preview=""):
        if len(data) > self.cfg.max_upload_bytes * 2 and kind == "download":
            raise Busy("The converted file is too large.")
        with self.lock:
            self.purge()
            mine = sum(1 for e in self.items.values() if e.owner == owner)
            if mine >= self.cfg.max_per_visitor:
                raise LimitReached("You have too many files in temporary storage. Remove some or wait for them to expire.")
            if len(self.items) >= self.cfg.max_items or self.total_bytes() + len(data) > self.cfg.max_store_bytes:
                raise Busy("The temporary file storage is full right now. Please try again in a few minutes.")
            self.cfg.root.mkdir(parents=True, exist_ok=True)
            entry = Entry(id=secrets.token_urlsafe(24), kind=kind, disk=secrets.token_hex(16) + ".bin", name=name, fmt=fmt,
                          size=len(data), owner=owner, expires=self.clock() + self.cfg.ttl_seconds, source_id=source_id,
                          preview=preview)
            with open(self._path(entry), "wb") as fh:
                fh.write(data)
            self.items[entry.id] = entry
            return entry

    def get(self, file_id, kind=None):
        with self.lock:
            self.purge()
            entry = self.items.get(file_id) if isinstance(file_id, str) else None
            if entry is None or (kind and entry.kind != kind):
                raise NotFound("That file was not found. It may have expired (files are kept for a short time only).")
            return entry

    def read(self, entry):
        try:
            return self._path(entry).read_bytes()
        except OSError:
            with self.lock:
                self._drop(entry)
            raise NotFound("That file was not found. It may have expired.")

    def delete(self, file_id):
        with self.lock:
            entry = self.items.get(file_id) if isinstance(file_id, str) else None
            if entry is None:
                return False
            self._drop(entry)
            return True

    def seconds_left(self, entry):
        return max(0, int(entry.expires - self.clock()))
