"""ELVARN search gatekeeper: a tiny authenticating front door for a private SearXNG.

    internet --> [ this proxy, 0.0.0.0:$PORT ] --> [ SearXNG, 127.0.0.1:8081 only ]

SearXNG has no login of its own, so on a public address anybody could use (and abuse) it.
This proxy lets through ONE kind of request only: an authenticated  GET /search  with a few
checked parameters. Everything else is refused.

Python standard library only (the SearXNG image has Python but we install nothing extra).

Environment variables
    SEARCH_PROXY_TOKEN            required, at least 24 characters. No token -> the server refuses to start.
    PORT                          port to listen on (Render sets it), default 10000
    SEARXNG_INTERNAL_PORT         where SearXNG listens on 127.0.0.1, default 8081
    SEARCH_PROXY_RATE_PER_MINUTE  global requests per minute, default 60
"""
import hmac
import http.client
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlencode, urlsplit

log = logging.getLogger("search_proxy")

MIN_TOKEN_CHARS = 24
MAX_PATH_CHARS = 2048            # request target (path + query) longer than this is refused (414)
MAX_HEADER_BYTES = 8192          # all request headers together (431)
MAX_QUERY_CHARS = 300
MAX_RESPONSE_BYTES = 1_000_000   # SearXNG answers bigger than this are refused (502)
ALLOWED_CATEGORIES = {"general", "news", "science", "it"}
ALLOWED_TIME_RANGES = {"day", "week", "month", "year"}
_LANGUAGE = re.compile(r"^(all|[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?)$")
_CONTROL = re.compile("[\x00-\x1f\x7f-\x9f]")
UPSTREAM_USER_AGENT = "ELVARN-search-gatekeeper/1.0"   # the ONLY header we invent for SearXNG


class ConfigError(Exception):
    pass


class Config:
    """All settings in one place (tests build this directly with small numbers)."""

    def __init__(self, token, upstream_host="127.0.0.1", upstream_port=8081, rate_per_minute=60,
                 max_concurrent=4, upstream_timeout=8.0, max_response_bytes=MAX_RESPONSE_BYTES,
                 client_timeout=10.0):
        if not isinstance(token, str) or len(token) < MIN_TOKEN_CHARS:
            raise ConfigError(f"SEARCH_PROXY_TOKEN is missing or shorter than {MIN_TOKEN_CHARS} characters")
        self.token = token.encode("utf-8")
        self.upstream_host = upstream_host
        self.upstream_port = int(upstream_port)
        self.rate_per_minute = max(1, int(rate_per_minute))
        self.upstream_timeout = float(upstream_timeout)
        self.max_response_bytes = int(max_response_bytes)
        self.client_timeout = float(client_timeout)
        self.limiter = TokenBucket(self.rate_per_minute)
        self.slots = threading.BoundedSemaphore(max(1, int(max_concurrent)))


def config_from_env(env=None):
    env = os.environ if env is None else env
    try:
        return Config(env.get("SEARCH_PROXY_TOKEN", ""),
                      upstream_port=int(env.get("SEARXNG_INTERNAL_PORT", "8081")),
                      rate_per_minute=int(env.get("SEARCH_PROXY_RATE_PER_MINUTE", "60")))
    except ValueError:
        raise ConfigError("a numeric environment variable is not a number")


class TokenBucket:
    """Global rate limit. Holds up to `per_minute` tokens and refills continuously."""

    def __init__(self, per_minute):
        self.capacity = float(per_minute)
        self.tokens = float(per_minute)
        self.refill_per_second = per_minute / 60.0
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def allow(self):
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_per_second)
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


class BadRequest(Exception):
    pass


def build_upstream_query(raw_query):
    """Check the client's query string and build a NEW one for SearXNG. Unknown parameters are dropped."""
    try:
        pairs = parse_qsl(raw_query, keep_blank_values=True, max_num_fields=30, encoding="utf-8", errors="strict")
    except ValueError:
        raise BadRequest("bad query string")
    first = {}
    for key, value in pairs:
        first.setdefault(key, value)             # if a name repeats, only the first value counts
    q = first.get("q", "")
    q = " ".join(q.split()) if not _CONTROL.search(q.replace("\t", " ")) else None
    if not q or len(q) > MAX_QUERY_CHARS:
        raise BadRequest("bad q")
    out = [("q", q), ("format", "json")]          # format=json is forced, whatever the client sent
    if "language" in first:
        if not _LANGUAGE.match(first["language"]):
            raise BadRequest("bad language")
        out.append(("language", first["language"]))
    if "safesearch" in first:
        if first["safesearch"] not in ("0", "1", "2"):
            raise BadRequest("bad safesearch")
        out.append(("safesearch", first["safesearch"]))
    if "pageno" in first:
        if not re.fullmatch(r"[1-5]", first["pageno"]):
            raise BadRequest("bad pageno")
        out.append(("pageno", first["pageno"]))
    if "time_range" in first:
        if first["time_range"] not in ALLOWED_TIME_RANGES:
            raise BadRequest("bad time_range")
        out.append(("time_range", first["time_range"]))
    if "categories" in first:
        cats = first["categories"].split(",")
        if len(cats) > 4 or any(c not in ALLOWED_CATEGORIES for c in cats):
            raise BadRequest("bad categories")
        out.append(("categories", ",".join(cats)))
    return urlencode(out)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"      # one request per connection: simple, no keep-alive tricks
    server_version = "gatekeeper"      # do not advertise Python/library versions
    sys_version = ""

    def setup(self):
        self.timeout = self.server.cfg.client_timeout     # socket timeout: slow clients get cut off
        super().setup()

    # ---- output helpers -------------------------------------------------------------------
    def log_message(self, *args):      # silence the default log (it would print client IPs and paths)
        pass

    def _reply(self, status, body, path_class, started):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(data)))
            if status == 405:
                self.send_header("Allow", "GET")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        except OSError:
            pass                       # the client went away
        # The only thing logged: result code, time and what kind of path. No query, token or client address.
        log.info("status=%d ms=%d path=%s", status, int((time.monotonic() - started) * 1000), path_class)

    def _error(self, status, started, path_class="other"):
        names = {400: "bad request", 401: "unauthorized", 404: "not found", 405: "method not allowed",
                 414: "uri too long", 429: "too many requests", 431: "headers too large",
                 502: "bad gateway", 503: "busy", 504: "gateway timeout"}
        self._reply(status, {"error": names.get(status, "error")}, path_class, started)

    def send_error(self, code, message=None, explain=None):    # errors raised by http.server itself
        self._error(code, time.monotonic())

    # ---- routing --------------------------------------------------------------------------
    def do_GET(self):
        started = time.monotonic()
        cfg = self.server.cfg
        if len(self.path) > MAX_PATH_CHARS:
            return self._error(414, started)
        if sum(len(k) + len(v) + 4 for k, v in self.headers.items()) > MAX_HEADER_BYTES:
            return self._error(431, started)
        parts = urlsplit(self.path)
        if parts.path == "/healthz":              # no login, and SearXNG is NOT asked: Render health check
            return self._reply(200, b"ok", "healthz", started)
        if parts.path == "/readyz":
            if not self._authorized():
                return self._error(401, started, "readyz")
            ok = self._upstream_alive()
            return self._reply(200 if ok else 503, {"ready": ok}, "readyz", started)
        if parts.path != "/search":
            return self._error(404, started)
        if not self._authorized():
            return self._error(401, started, "search")
        if not cfg.limiter.allow():
            return self._error(429, started, "search")
        try:
            query = build_upstream_query(parts.query)
        except BadRequest:
            return self._error(400, started, "search")
        if not cfg.slots.acquire(blocking=False):
            return self._error(503, started, "search")
        try:
            status, body = self._ask_searxng(query)
        finally:
            cfg.slots.release()
        if status == 200:
            self._reply(200, body, "search", started)
        else:
            self._error(status, started, "search")

    def _method_not_allowed(self):
        self._error(405, time.monotonic())

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _method_not_allowed

    # ---- helpers --------------------------------------------------------------------------
    def _authorized(self):
        header = self.headers.get("Authorization", "")
        scheme, _, supplied = header.partition(" ")
        supplied_bytes = supplied.strip().encode("utf-8", "replace") if scheme.lower() == "bearer" else b""
        # compare_digest takes the same time however many characters match (no timing leak)
        return hmac.compare_digest(supplied_bytes, self.server.cfg.token)

    def _upstream_alive(self):
        cfg = self.server.cfg
        try:
            conn = http.client.HTTPConnection(cfg.upstream_host, cfg.upstream_port, timeout=3)
            conn.request("GET", "/healthz", headers={"User-Agent": UPSTREAM_USER_AGENT})
            conn.getresponse().read(1024)
            conn.close()
            return True
        except (OSError, http.client.HTTPException):
            return False

    def _ask_searxng(self, query):
        """Return (status, body). Only a fixed User-Agent goes upstream: no cookies, no client headers, no client IP."""
        cfg = self.server.cfg
        deadline = time.monotonic() + cfg.upstream_timeout
        conn = None
        try:
            conn = http.client.HTTPConnection(cfg.upstream_host, cfg.upstream_port, timeout=cfg.upstream_timeout)
            conn.request("GET", "/search?" + query,
                         headers={"User-Agent": UPSTREAM_USER_AGENT, "Accept": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                return 502, None
            chunks, size = [], 0
            while True:
                if time.monotonic() > deadline:
                    return 504, None
                chunk = resp.read(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > cfg.max_response_bytes:
                    return 502, None
                chunks.append(chunk)
            body = b"".join(chunks)
            json.loads(body.decode("utf-8"))          # must be valid JSON, otherwise do not pass it on
            return 200, body
        except TimeoutError:
            return 504, None
        except OSError as e:
            return (504, None) if "timed out" in str(e) else (502, None)
        except (http.client.HTTPException, ValueError):
            return 502, None
        finally:
            if conn is not None:
                conn.close()


class GatekeeperServer(ThreadingHTTPServer):
    daemon_threads = True              # do not wait for stuck connections when shutting down
    request_queue_size = 32

    def __init__(self, address, cfg):
        super().__init__(address, Handler)
        self.cfg = cfg


def make_server(cfg, host="0.0.0.0", port=0):
    return GatekeeperServer((host, port), cfg)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        cfg = config_from_env()
    except ConfigError as e:
        print(f"search_proxy: refusing to start: {e}", file=sys.stderr)
        return 2
    port = int(os.environ.get("PORT", "10000"))
    server = make_server(cfg, "0.0.0.0", port)

    def stop(*_):                       # shutdown() must run in another thread than serve_forever()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("gatekeeper listening on port %d", port)
    server.serve_forever()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
