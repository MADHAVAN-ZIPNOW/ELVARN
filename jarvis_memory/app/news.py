"""Live AI, quantum and space news: free RSS feeds, one fresh story at a time, never the same story twice."""
from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Optional

import httpx

FEEDS: dict[str, list[str]] = {
    "ai": ["https://www.technologyreview.com/topic/artificial-intelligence/feed", "https://venturebeat.com/category/ai/feed/",
           "https://www.artificialintelligence-news.com/feed/"],
    "quantum": ["https://thequantuminsider.com/feed/", "https://phys.org/rss-feed/physics-news/quantum-physics/"],
    "space": ["https://www.nasa.gov/feed/", "https://spacenews.com/feed/", "https://www.space.com/feeds/all"],
}
LABEL = {"ai": "AI", "quantum": "Quantum", "space": "Space"}
FRESH_HOURS = 18
KEEP_DAYS = 14
UA = "JarvisMemory/1.0 (+personal assistant news reader)"


def _tag(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1].lower()


def _text(el: ET.Element, *names: str) -> str:
    for child in el:
        if _tag(child) in names and (child.text or "").strip():
            return child.text.strip()
    return ""


def _plain(raw: str, limit: int = 180) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", raw or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "…"


def _when(raw: str) -> Optional[datetime]:
    try:
        d = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def parse_feed(xml_text: str) -> list[dict[str, Any]]:
    """Stories from an RSS or Atom document: title, link, short summary, published time (newest first)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    for item in root.iter():
        if _tag(item) not in ("item", "entry"):
            continue
        link = _text(item, "link")
        if not link:
            for c in item:
                if _tag(c) == "link" and c.get("href"):
                    link = c.get("href", "")
                    break
        title = _plain(_text(item, "title"), 140)
        when = _when(_text(item, "pubdate", "published", "updated", "date"))
        if title and link.startswith("https://") and when:
            out.append({"title": title, "link": link, "summary": _plain(_text(item, "description", "summary", "content")), "when": when})
    return sorted(out, key=lambda s: s["when"], reverse=True)


def fetch(url: str) -> str:
    with httpx.Client(timeout=8.0, follow_redirects=True, headers={"User-Agent": UA}) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.text[:600_000]


def latest(topic: str, now: datetime, fetcher: Callable[[str], str] = fetch, hours: int = FRESH_HOURS, per_feed: int = 5) -> list[dict[str, Any]]:
    """Fresh stories for one topic from all its feeds, newest first (a feed that is down is simply skipped)."""
    stories: list[dict[str, Any]] = []
    for url in FEEDS.get(topic, []):
        try:
            items = parse_feed(fetcher(url))
        except Exception:
            continue
        stories += [dict(s, topic=topic) for s in items[:per_feed] if now - s["when"] <= timedelta(hours=hours)]
    return sorted(stories, key=lambda s: s["when"], reverse=True)


def link_hash(link: str) -> str:
    return hashlib.sha1(link.split("?")[0].encode()).hexdigest()[:20]


LIVE_SCORES = "http://static.cricinfo.com/rss/livescores.xml"
CRICKET_NEWS = "https://www.espncricinfo.com/rss/content/story/feeds/0.xml"


def _titles(xml_text: str, limit: int = 60) -> list[str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    for item in root.iter():
        if _tag(item) == "item":
            t = _plain(_text(item, "title"), 160)
            if t:
                out.append(t)
    return out[:limit]


def cricket(query: str, fetcher: Callable[[str], str] = fetch) -> str:
    """Live cricket from the free Cricinfo feeds: matches that fit the query, else what is on now plus the latest headlines."""
    try:
        live = _titles(fetcher(LIVE_SCORES))
    except Exception:
        live = []
    try:
        news = _titles(fetcher(CRICKET_NEWS), 8)
    except Exception:
        news = []
    words = [w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2 and w not in ("the", "match", "score", "result", "versus", "and", "game", "cricket", "live", "who", "won", "what", "was")]
    aliases = {"ind": "india", "wi": "west indies", "aus": "australia", "sa": "south africa", "eng": "england", "pak": "pakistan", "sl": "sri lanka", "nz": "new zealand"}
    words = [aliases.get(w, w) for w in words]
    hits = [t for t in live if words and all(any(part in t.lower() for part in w.split()) for w in words)]
    if not hits:
        hits = [t for t in live if words and any(any(part in t.lower() for part in w.split()) for w in words)]
    parts = []
    if hits:
        parts.append("MATCHES FOUND (live scores feed, newest first): " + " | ".join(hits[:6]))
    else:
        parts.append("No match for that query in the live scores feed right now. SAY THIS CLEARLY (do not ask the owner anything): there is no such match being played or just finished at the moment, then list what is being played now: " + " | ".join(live[:14]))
    if news:
        parts.append("Latest cricket headlines: " + " | ".join(news[:6]))
    return " ".join(parts) if live or news else ""


def headlines(topics: list[str], now: datetime, fetcher: Callable[[str], str] = fetch, limit: int = 6, hours: int = 72) -> str:
    """A plain list of the latest stories, for JARVIS to read out when asked ("any space news?")."""
    picked: list[dict[str, Any]] = []
    for t in topics:
        picked += latest(t, now, fetcher, hours=hours, per_feed=3)[:3]
    picked.sort(key=lambda s: s["when"], reverse=True)
    return "\n".join(f"- [{LABEL[s['topic']]}] {s['title']} — {s['summary']} ({s['link']})" for s in picked[:limit])
