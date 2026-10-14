"""Shared pieces for the collectors: a polite fetcher, a raw-file archive, incremental state and message sinks.

Collectors only DOWNLOAD and describe documents. They publish one JSON message per document (see make_message) to a sink
(Kafka in production). Spark (spark_job.py) reads those messages, extracts text and loads the knowledge graph.

Scope: Tamil Nadu corruption, 1947 to 2026 (see in_window / MIN_YEAR / MAX_YEAR).
"""
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path

# Generic on purpose: no personal details are ever sent to the sites being collected from.
USER_AGENT = "ELVARN-research/1.0 (non-commercial research project)"
MIN_YEAR, MAX_YEAR = 1947, 2026
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "kb_raw"
STATE_DIR = DATA_DIR / "kb_state"

TN_TERMS = re.compile(r"Tamil ?Nadu|Madras|Chennai|Madurai|Coimbatore|Tiruchi|Trichy|Salem|Tirunelveli|Tiruppur|Erode|Vellore|"
                      r"Thanjavur|Nilgiris|Kancheepuram|Chengalpet|Tiruvallur|Kanniyakumari|Dindigul|Namakkal|Virudhunagar|"
                      r"Sivaganga|Ramanathapuram|Thoothukudi|Tuticorin|Cuddalore|Villupuram|Kallakurichi|Karur|Ariyalur|"
                      r"Perambalur|Pudukkottai|Theni|Krishnagiri|Dharmapuri|Tiruvannamalai|Tenkasi|Ranipet|Tirupathur|\bTN\b|"
                      r"\bDMK\b|AIADMK|\bTNPSC\b|\bTANGEDCO\b|\bTNEB\b|\bTASMAC\b|\bDVAC\b", re.I)


class FetchError(Exception):
    pass


def in_window(date_or_year, min_year=MIN_YEAR, max_year=MAX_YEAR):
    """True if an ISO date / year (str or int) is within [min_year, max_year]. Unknown dates are NOT in the window."""
    m = re.match(r"(\d{4})", str(date_or_year or ""))
    return bool(m and min_year <= int(m.group(1)) <= max_year)


def is_tamil_nadu(*texts):
    return any(TN_TERMS.search(t or "") for t in texts)


class Fetcher:
    """urllib-based HTTP client: honours robots.txt (a missing robots.txt means allowed), waits `delay` seconds between
    requests, retries with back-off, caps the download size, and sends a generic User-Agent."""

    def __init__(self, delay=3.0, timeout=45, retries=3, max_bytes=40_000_000, respect_robots=True, open_fn=None, sleep=time.sleep):
        self.delay, self.timeout, self.retries, self.max_bytes = delay, timeout, retries, max_bytes
        self.respect_robots = respect_robots
        self._open = open_fn or urllib.request.urlopen
        self._sleep = sleep
        self._last = 0.0
        self._robots = {}
        self._lock = threading.Lock()

    def _wait(self):
        with self._lock:
            gap = self.delay - (time.monotonic() - self._last)
            if gap > 0:
                self._sleep(gap)
            self._last = time.monotonic()

    def allowed(self, url):
        if not self.respect_robots:
            return True
        p = urllib.parse.urlsplit(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                req = urllib.request.Request(base + "/robots.txt", headers={"User-Agent": USER_AGENT})
                with self._open(req, timeout=self.timeout) as resp:
                    ctype = resp.headers.get("Content-Type", "")
                    body = resp.read(500_000).decode("utf-8", "replace")
                rp.parse(body.splitlines() if "text/plain" in ctype or body.lstrip().lower().startswith(("user-agent", "#")) else [])
            except Exception:
                rp.parse([])                      # 404 / unreachable / not a robots file: nothing is disallowed
            self._robots[base] = rp
        return self._robots[base].can_fetch(USER_AGENT, url)

    def request(self, url, data=None, headers=None):
        """GET (or POST when `data` is a dict) -> (bytes, content_type)."""
        if not self.allowed(url):
            raise FetchError(f"robots.txt disallows {url}")
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        last = None
        for attempt in range(self.retries):
            self._wait()
            try:
                req = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT, **(headers or {})})
                with self._open(req, timeout=self.timeout) as resp:
                    payload = resp.read(self.max_bytes + 1)
                    if len(payload) > self.max_bytes:
                        raise FetchError(f"{url} is larger than {self.max_bytes} bytes")
                    return payload, resp.headers.get("Content-Type", "")
            except FetchError:
                raise
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code in (400, 401, 403, 404, 410):
                    break                          # not going to get better by retrying
                self._sleep(2 ** attempt * 2)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
                self._sleep(2 ** attempt * 2)
        raise FetchError(f"{url}: {last}")

    def get(self, url, headers=None):
        return self.request(url, None, headers)


class RawStore:
    """Content-addressed archive of every downloaded file, so any later step can be re-run from the originals."""

    def __init__(self, root=RAW_DIR):
        self.root = Path(root)

    def save(self, data, suffix=".bin"):
        sha = hashlib.sha256(data).hexdigest()
        rel = Path(sha[:2]) / f"{sha}{suffix}"
        path = self.root / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)                      # a half-written file is never visible under the final name
        return sha, rel.as_posix()


class SeenStore:
    """Set of already-collected keys (urls), saved to a JSON file, so each run only fetches what is new."""

    def __init__(self, name, root=STATE_DIR):
        self.path = Path(root) / f"{name}.json"
        self.keys = set(json.loads(self.path.read_text(encoding="utf-8"))) if self.path.is_file() else set()

    def __contains__(self, key):
        return key in self.keys

    def add(self, key):
        self.keys.add(key)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.keys)), encoding="utf-8")
        tmp.replace(self.path)


def make_message(*, site, doc_type, title, url, publisher, licence, published_date=None, raw_path=None, sha256=None,
                 content_type=None, text=None, text_reliable=True, extra=None):
    """The JSON document that goes on the Kafka topic. `text` (if given) is the reliable text to use instead of extracting it
    from the raw file; `text_reliable` False means the raw file's text must not be trusted (e.g. legacy-encoded Tamil PDFs)."""
    return {"site": site, "doc_type": doc_type, "title": title, "url": url, "publisher": publisher, "licence": licence,
            "published_date": published_date, "raw_path": raw_path, "sha256": sha256, "content_type": content_type,
            "text": text, "text_reliable": text_reliable, "extra": extra or {},
            "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


class ListSink:
    """In-memory sink for tests."""

    def __init__(self):
        self.messages = []

    def send(self, msg):
        self.messages.append(msg)

    def flush(self):
        pass


def key_of(msg):
    return msg.get("url") or msg.get("sha256") or ""
