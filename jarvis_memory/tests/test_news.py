"""Live news: feed parsing, one fresh unseen story per push, limits, quiet hours, and the chat tool."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app import jobs, news
from tests.test_email_reminders import FakePusher, TOKEN

NOW = datetime(2026, 1, 5, 6, 0, tzinfo=timezone.utc)   # 11:30 in Kolkata
P = "/api/jarvis/memory"


def rss(*items):
    body = "".join(f"<item><title>{t}</title><link>{l}</link><description>&lt;p&gt;{d}&lt;/p&gt;</description>"
                   f"<pubDate>{(NOW - timedelta(hours=h)).strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate></item>" for t, l, d, h in items)
    return f"<rss><channel>{body}</channel></rss>"


def fetcher(pages):
    return lambda url: pages.get(url.split("/")[2], rss())


def test_parse_keeps_https_stories_with_clean_summaries():
    s = news.parse_feed(rss(("Qubit record", "https://q.example/a", "Longer &lt;b&gt;coherence&lt;/b&gt;", 1), ("Bad", "http://x/y", "no", 1)))
    assert len(s) == 1 and s[0]["summary"] == "Longer coherence" and s[0]["title"] == "Qubit record"
    assert news.parse_feed("not xml") == []


def enable(client, pusher_tokens=True):
    client.patch(f"{P}/settings", json={"news_enabled": True, "timezone": "Asia/Kolkata", "news_interval_minutes": 15})
    client.app_state.service.register_device(TOKEN) if pusher_tokens else None


def test_one_fresh_story_per_push_never_repeated_and_old_ones_skipped(client):
    enable(client)
    svc = client.app_state.service
    pages = {"thequantuminsider.com": rss(("Quantum chip", "https://thequantuminsider.com/1", "Fast", 2), ("Old news", "https://thequantuminsider.com/0", "x", 40))}
    client.patch(f"{P}/settings", json={"news_topics": "quantum"})
    pusher = FakePusher()
    out = jobs.send_news(svc, pusher, NOW, fetcher(pages))
    assert out["sent"] is True and out["topic"] == "quantum"
    sent = pusher.sent[-1][1]
    assert sent["type"] == "news" and sent["line"] == "Quantum chip" and sent["url"].startswith("https://")
    later = NOW + timedelta(minutes=20)
    assert jobs.send_news(svc, pusher, later, fetcher(pages))["why"] == "nothing_new"      # the 40 h old story is too old; the new one was sent


def test_news_respects_interval_daily_limit_and_quiet_hours(client):
    enable(client)
    svc = client.app_state.service
    client.patch(f"{P}/settings", json={"news_topics": "space", "news_interval_minutes": 60, "max_news_per_day": 1})
    pages = {"www.nasa.gov": rss(("Moon base", "https://www.nasa.gov/1", "a", 1), ("Mars", "https://www.nasa.gov/2", "b", 1))}
    pusher = FakePusher()
    assert jobs.send_news(svc, pusher, NOW, fetcher(pages))["sent"] is True
    assert jobs.send_news(svc, pusher, NOW + timedelta(minutes=5), fetcher(pages))["why"] == "not_due"
    assert jobs.send_news(svc, pusher, NOW + timedelta(hours=2), fetcher(pages))["why"] == "daily_limit"
    night = datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc)       # 23:30 Kolkata
    assert jobs.send_news(svc, pusher, night, fetcher(pages))["why"] == "quiet_hours"


def test_news_settings_are_validated(client):
    assert client.patch(f"{P}/settings", json={"news_topics": "ai,space"}).status_code == 200
    assert client.patch(f"{P}/settings", json={"news_topics": "sports"}).status_code == 422


def test_headlines_for_the_chat_tool():
    pages = {"www.nasa.gov": rss(("Moon base", "https://www.nasa.gov/1", "a", 1))}
    text = news.headlines(["space"], NOW, fetcher(pages))
    assert "[Space] Moon base" in text and "https://www.nasa.gov/1" in text
    assert news.headlines(["space"], NOW, lambda u: (_ for _ in ()).throw(OSError())) == ""    # feeds down: empty, no crash
