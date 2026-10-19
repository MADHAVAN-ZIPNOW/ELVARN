"""Web access for the chat: search the web and read a page. Pages are UNTRUSTED data; nothing here can act on them.

Safety: https only, no credentials in URLs, every hop (including redirects) must resolve to a public address, responses are size
capped and only text types are read, so a chat can never be pointed at your network or at a huge download.
"""
from __future__ import annotations

import html
import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

MAX_BYTES = 400_000
MAX_CHARS = 12_000
MAX_REDIRECTS = 3
TEXT_TYPES = ("text/html", "text/plain", "application/xhtml+xml", "application/json", "text/xml", "application/xml")
UA = "Mozilla/5.0 (compatible; JarvisMemory/1.0; personal assistant)"
WIKI_UA = "JarvisMemory/1.0 (https://github.com/MADHAVAN-ZIPNOW/Jarvis; personal assistant) httpx"


class WebError(Exception):
    """A problem worth telling the user about in plain words."""


def check_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme != "https":
        raise WebError("Only https:// links can be opened.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise WebError("That link is not allowed.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except OSError:
        raise WebError("I could not find that website.") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:   # private, loopback, link-local, reserved, metadata addresses...
            raise WebError("That address is not a public website, so I will not open it.")
    return parsed.geturl()


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form", "iframe"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip and data.strip():
            self.parts.append(data.strip() + " ")


def html_to_text(raw: str) -> tuple[str, str]:
    parser = _Text()
    parser.feed(raw)
    text = re.sub(r"[ \t]+\n", "\n", "".join(parser.parts))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return " ".join(parser.title.split())[:200], text


def read_page(url: str, client: Optional[httpx.Client] = None) -> dict[str, str]:
    """{"url", "title", "text"} for a public https page. Raises WebError with a human message."""
    owns = client is None
    client = client or httpx.Client(timeout=12.0, follow_redirects=False, headers={"User-Agent": UA})
    try:
        current = check_url(url)
        for _ in range(MAX_REDIRECTS + 1):
            with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    current = check_url(str(httpx.URL(current).join(resp.headers.get("location", ""))))
                    continue
                if resp.status_code >= 400:
                    raise WebError(f"The website answered with an error ({resp.status_code}).")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype not in TEXT_TYPES:
                    raise WebError("That link is not a readable web page (it looks like a file download).")
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        break
                raw = body[:MAX_BYTES].decode(resp.encoding or "utf-8", errors="replace")
                title, text = html_to_text(raw) if "html" in ctype or "xml" in ctype else ("", raw)
                if not text:
                    raise WebError("The page has no readable text.")
                return {"url": current, "title": title, "text": text[:MAX_CHARS]}
        raise WebError("The website redirected too many times.")
    except httpx.HTTPError as exc:
        raise WebError(f"I could not load that page ({type(exc).__name__}).") from None
    finally:
        if owns:
            client.close()


_RESULT = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?(?:class="result__snippet"[^>]*>(.*?)</a>)?', re.S)


def _clean(s: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", s or "")).split())


def _ddg(client: httpx.Client, query: str, endpoint: str, limit: int) -> list[dict[str, str]]:
    resp = client.post(endpoint, data={"q": query[:200]})
    resp.raise_for_status()
    results: list[dict[str, str]] = []
    for href, title, snippet in _RESULT.findall(resp.text):
        link = href
        if "uddg=" in href:
            link = unquote(parse_qs(urlparse(href if href.startswith("http") else "https:" + href).query).get("uddg", [""])[0])
        if link.startswith("https://"):
            results.append({"title": _clean(title)[:150], "url": link[:300], "snippet": _clean(snippet)[:300]})
        if len(results) >= limit:
            break
    return results


def _wikipedia(client: httpx.Client, query: str, limit: int) -> list[dict[str, str]]:
    resp = client.get("https://en.wikipedia.org/w/api.php", headers={"User-Agent": WIKI_UA}, params={"action": "query", "list": "search", "srsearch": query[:200],
                                                                    "srlimit": limit, "format": "json"})
    resp.raise_for_status()
    return [{"title": h["title"][:150], "url": "https://en.wikipedia.org/wiki/" + quote_plus(h["title"].replace(" ", "_"), safe="_()"),
             "snippet": _clean(h.get("snippet", ""))[:300]} for h in resp.json().get("query", {}).get("search", [])[:limit]]


def _tavily(client: httpx.Client, query: str, key: str, limit: int) -> list[dict[str, str]]:
    resp = client.post("https://api.tavily.com/search", json={"query": query[:300], "max_results": limit},
                       headers={"Authorization": f"Bearer {key}"})
    resp.raise_for_status()
    return [{"title": _clean(r.get("title", ""))[:150], "url": r["url"][:300], "snippet": _clean(r.get("content", ""))[:300]}
            for r in resp.json().get("results", []) if str(r.get("url", "")).startswith("https://")][:limit]


def search_web(query: str, client: Optional[httpx.Client] = None, limit: int = 5, tavily_key: str = "") -> list[dict[str, str]]:
    """Top results (title, url, snippet). Tries Tavily (if a key is set), DuckDuckGo, DuckDuckGo Lite, then Wikipedia.

    Servers in data centres are often blocked by search engines, so a chain of sources keeps this working without paid services."""
    owns = client is None
    client = client or httpx.Client(timeout=5.0, headers={"User-Agent": UA})
    attempts = []
    if tavily_key:
        attempts.append(lambda: _tavily(client, query, tavily_key, limit))
    attempts += [lambda: _ddg(client, query, "https://html.duckduckgo.com/html/", limit),
                 lambda: _wikipedia(client, query, limit),          # works from data centres, so it goes before the 2nd DDG try
                 lambda: _ddg(client, query, "https://lite.duckduckgo.com/lite/", limit)]
    last: Optional[Exception] = None
    try:
        for attempt in attempts:
            try:
                found = attempt()
                if found:
                    return found
            except httpx.HTTPError as exc:
                last = exc
    finally:
        if owns:
            client.close()
    if last is not None:
        raise WebError(f"Web search is not available right now ({type(last).__name__}).")
    return []
