"""Tests for search_service/proxy.py (the token gatekeeper). Everything runs offline on loopback ports."""
import http.client
import importlib.util
import json
import logging
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

PROXY_PATH = pathlib.Path(__file__).resolve().parents[2] / "search_service" / "proxy.py"
spec = importlib.util.spec_from_file_location("search_proxy_under_test", PROXY_PATH)
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)

TOKEN = "fake-test-token-0123456789abcdef-XYZ"
OK_BODY = json.dumps({"results": [{"title": "T", "url": "https://a.com", "content": "C"}]}).encode()


class FakeUpstream:
    """A fake SearXNG. `mode` changes its behaviour; `seen` records the requests it got."""

    def __init__(self):
        self.seen = []
        self.mode = "ok"
        self.body = OK_BODY
        self.delay = 0.0
        self.gate = None                      # threading.Event: when set, requests wait for it
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.seen.append((self.path, dict(self.headers)))
                if self.path.startswith("/healthz") and outer.mode != "broken":
                    return self._send(200, b"OK")
                if outer.gate is not None:
                    outer.gate.wait(5)
                time.sleep(outer.delay)
                if outer.mode == "status500":
                    return self._send(500, b"{}")
                self._send(200, outer.body)

            def _send(self, code, body):
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Set-Cookie", "upstream=1")
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass

            def log_message(self, *a):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Running:
    def __init__(self, cfg):
        self.cfg = cfg
        self.server = proxy.make_server(cfg, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def request(self, path, method="GET", headers=None, token=TOKEN):
        h = dict(headers or {})
        if token is not None and "Authorization" not in h:
            h["Authorization"] = "Bearer " + token
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        conn.request(method, path, headers=h)
        r = conn.getresponse()
        body = r.read()
        out = (r.status, body, r)
        conn.close()
        return out

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream():
    u = FakeUpstream()
    yield u
    u.close()


@pytest.fixture
def make_proxy(upstream):
    started = []

    def make(**kw):
        cfg = proxy.Config(kw.pop("token", TOKEN), upstream_port=kw.pop("upstream_port", upstream.port), **kw)
        p = Running(cfg)
        started.append(p)
        return p
    yield make
    for p in started:
        p.close()


# ------------------------------------------------------------------ start-up
@pytest.mark.parametrize("env", [{}, {"SEARCH_PROXY_TOKEN": ""}, {"SEARCH_PROXY_TOKEN": "short-token"},
                                 {"SEARCH_PROXY_TOKEN": "x" * 23}])
def test_refuses_to_start_without_a_long_token(env):
    with pytest.raises(proxy.ConfigError):
        proxy.config_from_env(env)


def test_main_exits_nonzero_without_token(monkeypatch, capsys):
    monkeypatch.delenv("SEARCH_PROXY_TOKEN", raising=False)
    assert proxy.main() != 0
    assert "refusing to start" in capsys.readouterr().err


def test_accepts_a_24_char_token():
    assert proxy.config_from_env({"SEARCH_PROXY_TOKEN": "x" * 24}).rate_per_minute == 60


# ------------------------------------------------------------------ auth
def test_missing_and_wrong_token_give_the_same_401(make_proxy, upstream):
    p = make_proxy()
    a = p.request("/search?q=hello", token=None)
    b = p.request("/search?q=hello", token="wrong-token-0123456789abcdef-XYZ")
    c = p.request("/search?q=hello", headers={"Authorization": "Basic " + TOKEN})
    assert a[0] == b[0] == c[0] == 401
    assert a[1] == b[1] == c[1]
    assert TOKEN.encode() not in a[1] and not [s for s in upstream.seen if s[0].startswith("/search")]


def test_correct_token_returns_json_with_safe_headers(make_proxy):
    status, body, resp = make_proxy().request("/search?q=hello+world")
    assert status == 200 and json.loads(body)["results"][0]["title"] == "T"
    assert resp.getheader("Content-Type") == "application/json"
    assert resp.getheader("Cache-Control") == "no-store"
    assert resp.getheader("X-Content-Type-Options") == "nosniff"
    assert resp.getheader("Set-Cookie") is None          # upstream cookies are not passed on


# ------------------------------------------------------------------ parameters
def test_only_allowlisted_params_reach_upstream_and_json_is_forced(make_proxy, upstream):
    q = "/search?q=cats&format=html&language=en&safesearch=2&pageno=2&time_range=week&categories=news,it&engines=evil&secret=1&q=second"
    assert make_proxy().request(q)[0] == 200
    path = [s[0] for s in upstream.seen if s[0].startswith("/search")][0]
    from urllib.parse import parse_qs, urlsplit
    got = parse_qs(urlsplit(path).query)
    assert got == {"q": ["cats"], "format": ["json"], "language": ["en"], "safesearch": ["2"], "pageno": ["2"],
                   "time_range": ["week"], "categories": ["news,it"]}


@pytest.mark.parametrize("path", [
    "/search", "/search?q=", "/search?q=%20%20", "/search?q=" + "a" * 301, "/search?q=a%00b", "/search?q=a%0Ab",
    "/search?q=x&safesearch=9", "/search?q=x&pageno=99", "/search?q=x&language=../../etc",
    "/search?q=x&categories=files", "/search?q=x&time_range=decade", "/search?q=x&categories=" + ",".join(["news"] * 9),
])
def test_bad_parameters_are_rejected_and_nothing_is_sent_upstream(make_proxy, upstream, path):
    assert make_proxy().request(path)[0] == 400
    assert not [s for s in upstream.seen if s[0].startswith("/search")]


def test_q_of_300_characters_is_accepted(make_proxy):
    assert make_proxy().request("/search?q=" + "a" * 300)[0] == 200


def test_very_long_target_and_huge_headers_are_refused(make_proxy):
    p = make_proxy()
    assert p.request("/search?q=x&junk=" + "a" * 3000)[0] == 414
    assert p.request("/search?q=x", headers={"X-Big": "a" * 9000})[0] == 431


# ------------------------------------------------------------------ routes and methods
@pytest.mark.parametrize("path", ["/", "/config", "/preferences", "/stats", "/autocompleter?q=x", "/opensearch.xml",
                                  "/static/x.js", "/search/", "/healthz/x", "/search/../config"])
def test_other_paths_are_404(make_proxy, path):
    assert make_proxy().request(path)[0] == 404


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
def test_other_methods_are_405(make_proxy, method):
    status, body, resp = make_proxy().request("/search?q=x", method=method)
    assert status == 405 and resp.getheader("Allow") == "GET"


def test_healthz_needs_no_token_and_does_not_call_upstream(make_proxy, upstream):
    p = make_proxy(upstream_port=1)         # nothing listens on port 1
    status, body, _ = p.request("/healthz", token=None)
    assert status == 200 and body == b"ok" and upstream.seen == []


def test_readyz_needs_token_and_checks_upstream(make_proxy, upstream):
    p = make_proxy()
    assert p.request("/readyz", token=None)[0] == 401
    assert p.request("/readyz")[0] == 200
    assert make_proxy(upstream_port=1).request("/readyz")[0] == 503


# ------------------------------------------------------------------ upstream failures
def test_response_size_cap(make_proxy, upstream):
    upstream.body = json.dumps({"results": ["x" * 5000]}).encode()
    assert make_proxy(max_response_bytes=1000).request("/search?q=x")[0] == 502


def test_upstream_timeout_is_504(make_proxy, upstream):
    upstream.delay = 1.5
    assert make_proxy(upstream_timeout=0.4).request("/search?q=x")[0] == 504


def test_upstream_down_is_502_with_generic_body(make_proxy):
    status, body, _ = make_proxy(upstream_port=1).request("/search?q=x")
    assert status == 502 and json.loads(body) == {"error": "bad gateway"}


def test_upstream_error_status_and_non_json_are_502(make_proxy, upstream):
    p = make_proxy()
    upstream.mode = "status500"
    assert p.request("/search?q=x")[0] == 502
    upstream.mode = "ok"
    upstream.body = b"<html>not json</html>"
    assert p.request("/search?q=x")[0] == 502


# ------------------------------------------------------------------ limits
def test_rate_limit_429(make_proxy):
    p = make_proxy(rate_per_minute=3)
    codes = [p.request("/search?q=x")[0] for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]


def test_unauthenticated_requests_do_not_use_up_the_rate_limit(make_proxy):
    p = make_proxy(rate_per_minute=2)
    for _ in range(6):
        assert p.request("/search?q=x", token=None)[0] == 401
    assert p.request("/search?q=x")[0] == 200


def test_token_bucket_refills(monkeypatch):
    """Uses a fake clock, so a slow or busy computer cannot change the result (the real clock made this test flaky)."""
    now = [1000.0]
    monkeypatch.setattr(proxy.time, "monotonic", lambda: now[0])
    b = proxy.TokenBucket(6000)      # 100 per second
    for _ in range(6000):
        assert b.allow()
    assert not b.allow()             # empty: no time has passed
    now[0] += 0.1                    # 0.1 s later: 10 tokens are back
    assert b.allow()
    for _ in range(9):
        assert b.allow()
    assert not b.allow()             # exactly those 10 were refilled


def test_concurrency_cap_503(make_proxy, upstream):
    upstream.gate = threading.Event()
    p = make_proxy(max_concurrent=2, rate_per_minute=100)
    results = []
    threads = [threading.Thread(target=lambda: results.append(p.request("/search?q=x")[0])) for _ in range(2)]
    for t in threads:
        t.start()
    deadline = time.time() + 5
    while len([s for s in upstream.seen if s[0].startswith("/search")]) < 2 and time.time() < deadline:
        time.sleep(0.02)
    assert p.request("/search?q=x")[0] == 503
    upstream.gate.set()
    for t in threads:
        t.join(10)
    assert results == [200, 200]
    assert p.request("/search?q=x")[0] == 200       # slots are given back


# ------------------------------------------------------------------ privacy
def test_client_headers_and_cookies_are_not_forwarded(make_proxy, upstream):
    h = {"Cookie": "session=abc", "X-Forwarded-For": "9.9.9.9", "X-Real-IP": "8.8.8.8", "Referer": "https://x.example",
         "Connection": "keep-alive", "User-Agent": "client-agent", "Proxy-Authorization": "secret", "X-Custom": "1"}
    assert make_proxy().request("/search?q=x", headers=h)[0] == 200
    sent = {k.lower(): v for k, v in [s for s in upstream.seen if s[0].startswith("/search")][0][1].items()}
    for bad in ("cookie", "x-forwarded-for", "x-real-ip", "referer", "authorization", "proxy-authorization", "x-custom"):
        assert bad not in sent
    assert sent["user-agent"] == proxy.UPSTREAM_USER_AGENT


def test_logs_never_contain_token_or_query(make_proxy, caplog):
    caplog.set_level(logging.DEBUG)
    p = make_proxy()
    p.request("/search?q=very-private-question")
    p.request("/search?q=very-private-question", token="wrong-token-0123456789abcdef-XYZ")
    p.request("/search?q=" + "a" * 400)
    p.request("/healthz")
    assert "status=200" in caplog.text and "status=401" in caplog.text
    for secret in (TOKEN, "very-private-question", "wrong-token", "127.0.0.1"):
        assert secret not in caplog.text


def test_proxy_source_uses_only_the_standard_library():
    import ast
    tree = ast.parse(PROXY_PATH.read_text(encoding="utf-8"))
    std = {"hmac", "http", "json", "logging", "os", "re", "signal", "sys", "threading", "time", "urllib"}
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module.split(".")[0])
    assert mods <= std, mods - std


def test_deploy_files_are_consistent():
    root = PROXY_PATH.parent
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("# ") and "FROM searxng/searxng:2026.10.11-ad46ebd7d" in dockerfile and "latest" not in dockerfile.replace("# ", "").split("FROM")[1].split("\n")[0]
    for name in ("entrypoint.sh", "proxy.py", "settings.yml", "Dockerfile"):
        assert b"\r" not in (root / name).read_bytes(), name
    import yaml
    s = yaml.safe_load((root / "settings.yml").read_text(encoding="utf-8"))
    assert s["server"]["bind_address"] == "127.0.0.1" and s["server"]["limiter"] is False
    assert "json" in s["search"]["formats"] and "secret_key" not in s["server"]
