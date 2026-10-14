"""Central Bureau of Investigation press releases, filtered to Tamil Nadu.

https://cbi.gov.in/press-releases lists releases as links to /press-detail/<id>; each detail page has the date, title and
full text. CBI has no Tamil-Nadu-only list, so every unseen detail page is fetched once and kept only if its title or text
mentions Tamil Nadu (a place or body in the TN term list). Non-TN releases are remembered so they are not fetched again.

Press releases are reports by an investigating agency: they can support 'alleged' / 'reported', never a court finding.
"""
import html
import re

from backend.pipeline.common import in_window, is_tamil_nadu, make_message

BASE = "https://cbi.gov.in/"
LIST_URL = BASE + "press-releases"
PUBLISHER = "Central Bureau of Investigation, Government of India"
LICENCE = "CBI website (no explicit reuse terms; attributed to CBI)"
LINK = re.compile(r'href="(?:https://cbi\.gov\.in/)?(press-detail/[A-Za-z0-9=]+)"[^>]*>\s*([^<]{8,300}?)\s*<', re.I)
DATE = re.compile(r'<div class="date-back">\s*<p>\s*(\d{2})-(\d{2})-(\d{4})\s*</p>', re.I)
TITLE = re.compile(r"<h4>(.*?)</h4>", re.I | re.S)
BODY = re.compile(r'<div class="press-txt">(.*?)</div>', re.I | re.S)


def clean(text):
    return re.sub(r"[ \t]+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def parse_list(page_html):
    out, seen = [], set()
    for rel, title in LINK.findall(page_html):
        if rel not in seen:
            seen.add(rel)
            out.append({"url": BASE + rel, "title": clean(title)})
    return out


def parse_detail(page_html):
    """-> {"date": ISO, "title", "text"} or None if the page does not look like a press release."""
    d, t, b = DATE.search(page_html), TITLE.search(page_html), BODY.search(page_html)
    if not (d and t and b):
        return None
    paragraphs = [clean(p) for p in re.split(r"</p>", b.group(1)) if clean(p)]
    return {"date": f"{d.group(3)}-{d.group(2)}-{d.group(1)}", "title": clean(t.group(1)), "text": "\n\n".join(paragraphs)}


def collect(fetcher, store, sink, seen, limit=None, min_year=1947, max_year=2026, max_fetch=None):
    """Publish every new in-window, Tamil Nadu-related press release. `limit` caps publications, `max_fetch` caps detail fetches."""
    page, _ = fetcher.get(LIST_URL)
    items = parse_list(page.decode("utf-8", "replace"))
    stats = {"listed": len(items), "fetched": 0, "published": 0, "not_tn": 0, "out_of_window": 0, "skipped_seen": 0, "unparsed": 0}
    for it in items:
        if limit is not None and stats["published"] >= limit:
            break
        if max_fetch is not None and stats["fetched"] >= max_fetch:
            break
        if it["url"] in seen:
            stats["skipped_seen"] += 1
            continue
        raw, ctype = fetcher.get(it["url"])
        stats["fetched"] += 1
        detail = parse_detail(raw.decode("utf-8", "replace"))
        if detail is None:
            stats["unparsed"] += 1
            continue                                # not marked seen: the page layout may be fixed later
        seen.add(it["url"])
        if not in_window(detail["date"], min_year, max_year):
            stats["out_of_window"] += 1
        elif not is_tamil_nadu(detail["title"], detail["text"]):
            stats["not_tn"] += 1
        else:
            sha, raw_path = store.save(raw, ".html")
            sink.send(make_message(site="cbi", doc_type="press_release", title=detail["title"][:240], url=it["url"],
                                   publisher=PUBLISHER, licence=LICENCE, published_date=detail["date"], raw_path=raw_path,
                                   sha256=sha, content_type=ctype, text=f"{detail['title']}\n\n{detail['text']}", text_reliable=True))
            sink.flush()
            stats["published"] += 1
        seen.save()
    return stats
