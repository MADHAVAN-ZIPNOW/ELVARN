"""SearXNG provider: asks a self-hosted SearXNG instance for web results as JSON.

  GET {SEARXNG_BASE_URL}/search?q=...&format=json

No API key and no paid service. Only titles, URLs and short snippets are used: result pages are
NEVER fetched, so there are no redirects to follow ("after redirects" does not apply in this
version; if full-page fetching is ever added, every redirect hop must be re-validated with
search.sanitize.clean_url, and DNS answers must be checked too).

Trust model:
  * SEARXNG_BASE_URL is trusted operator configuration (it may be localhost or a private address).
    It is validated once (http/https only) and is never built from user input.
  * Everything SearXNG returns is untrusted. This class only parses it; the SearchService cleans it
    and filters result URLs (public http/https only).
"""
import json
import logging
import re
import threading
import time
from typing import List
from urllib.parse import urlsplit

import httpx

log = logging.getLogger("elvarn.search")

from .base import RawResult, SearchError, SearchProvider, SearchRateLimited, SearchTimeout, SearchUnavailable

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
MAX_QUERY_CHARS = 300
HARD_MAX_RESULTS = 8               # never keep more than this, whatever the server sends
MAX_BODY_BYTES = 400_000           # cap on the response body we are willing to read
RETRIES = 1                        # one extra try for connection errors, timeouts and 5xx; none for 4xx
_CONTROL = re.compile("[\x00-\x1f\x7f-\x9f]")


def validate_base_url(value):
    """Return the base URL without a trailing slash, or raise ValueError. http/https only."""
    url = (value or "").strip()
    if not url or len(url) > 300 or _CONTROL.search(url) or re.search(r"\s", url):
        raise ValueError("empty or contains forbidden characters")
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018  (raises ValueError for a bad port)
    except ValueError:
        raise ValueError("malformed URL")
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError("must be an http:// or https:// address")
    if "@" in parts.netloc or parts.query or parts.fragment:
        raise ValueError("must not contain a user name, query or fragment")
    return url.rstrip("/")


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def is_loopback_host(url):
    host = (urlsplit(url).hostname or "").lower()
    return host in _LOCAL_HOSTS or host.endswith(".localhost")


def wake_up(base_url, token=None, timeout=60.0):
    """Start a background daemon thread that calls {base_url}/healthz once (the free Render search
    service sleeps when idle). Never blocks, never raises, never logs the URL or any token.
    Returns the thread (tests can join it)."""
    def run():
        try:
            httpx.get(base_url.rstrip("/") + "/healthz", timeout=timeout, follow_redirects=False,
                      headers={"User-Agent": "ELVARN/1.0"})
        except Exception:  # noqa: BLE001 - best effort only
            pass
    t = threading.Thread(target=run, name="search-wakeup", daemon=True)
    t.start()
    return t


def validate_query(query):
    """Return the query ready to send, or raise SearchError (nothing is sent in that case)."""
    raw = str(query or "")
    if _CONTROL.search(raw.replace("\t", " ").replace("\n", " ")):
        raise SearchError("invalid query")
    q = " ".join(raw.split())
    if not q:
        raise SearchError("empty query")
    if len(q) > MAX_QUERY_CHARS:
        raise SearchError("query too long")
    return q


class _Retryable(Exception):
    def __init__(self, error):
        self.error = error


class SearxngProvider(SearchProvider):
    name = "searxng"

    def __init__(self, base_url=DEFAULT_BASE_URL, transport=None, max_body_bytes=MAX_BODY_BYTES, retries=RETRIES,
                 auth_token=None):
        self.base_url = validate_base_url(base_url)     # raises ValueError at start-up if wrong
        self._auth_token = (auth_token or "").strip() or None
        # A bearer token must not travel in clear text over the internet: http:// is only allowed to localhost.
        if self._auth_token and urlsplit(self.base_url).scheme.lower() != "https" and not is_loopback_host(self.base_url):
            raise ValueError("SEARXNG_AUTH_TOKEN is set but SEARXNG_BASE_URL is not https:// (token refused over plain http)")
        self._transport = transport                      # tests inject httpx.MockTransport
        self.max_body_bytes = max_body_bytes
        self.retries = max(0, retries)

    def __repr__(self):                                  # never show the token
        return f"SearxngProvider(base_url={self.base_url!r}, auth={'on' if self._auth_token else 'off'})"

    def is_configured(self):
        return bool(self.base_url)

    def search(self, query, count, timeout) -> List[RawResult]:
        q = validate_query(query)
        count = max(1, min(int(count), HARD_MAX_RESULTS))
        params = {"q": q, "format": "json", "language": "en", "safesearch": "2", "pageno": "1"}
        return self._parse(self._get_with_retry(params, timeout), count)

    # ------------------------------------------------------------------ http
    def _get_with_retry(self, params, timeout):
        last = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(0.2)
            try:
                return self._get_once(params, timeout)
            except _Retryable as e:
                last = e.error
        raise last

    def _get_once(self, params, timeout):
        # follow_redirects=False: the configured server should answer directly
        try:
            headers = {"Accept": "application/json", "User-Agent": "ELVARN/1.0"}
            if self._auth_token:
                headers["Authorization"] = "Bearer " + self._auth_token
            with httpx.Client(transport=self._transport, timeout=timeout, follow_redirects=False,
                              headers=headers) as client:
                with client.stream("GET", self.base_url + "/search", params=params) as resp:
                    status = resp.status_code
                    if status >= 500:
                        raise _Retryable(SearchError(f"search server error {status}"))
                    if status == 429:
                        raise SearchRateLimited("search server is rate limiting")
                    if status in (401, 403) and self._auth_token:
                        # wrong or missing token at the search service: say so without any detail
                        log.warning("search service refused our credentials (HTTP %dxx)", status // 100)
                        raise SearchUnavailable("search service refused the credentials")
                    if status >= 400:
                        raise SearchError(f"search server refused the request ({status}); is format=json enabled?")
                    if status >= 300:
                        raise SearchError("unexpected redirect from the search server")
                    chunks, size = [], 0
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > self.max_body_bytes:
                            raise SearchError("search response too large")
                        chunks.append(chunk)
                    return b"".join(chunks)
        except httpx.TimeoutException:
            raise _Retryable(SearchTimeout("search server timed out"))
        except httpx.ConnectError:
            raise _Retryable(SearchUnavailable("cannot connect to the search server"))
        except httpx.HTTPError:
            raise _Retryable(SearchError("search request failed"))

    # --------------------------------------------------------------- parsing
    @staticmethod
    def _parse(body, count):
        try:
            data = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise SearchError("search server did not return valid JSON")
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise SearchError("unexpected JSON shape from the search server")
        out = []
        for item in data["results"]:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            title, snippet = item.get("title"), item.get("content")
            out.append(RawResult(title if isinstance(title, str) else "", item["url"],
                                 snippet if isinstance(snippet, str) else ""))
            if len(out) >= count:
                break
        return out
