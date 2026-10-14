"""Tamil Nadu Directorate of Vigilance and Anti-Corruption (DVAC): press-release archive.

https://dvac.tn.gov.in/Press_Release.html lists every press release as a link to a PDF (dated 2017 onwards on the site).
The PDFs are in Tamil with a legacy font, so their text cannot be trusted; the English link text on the list page is
reliable. So each message carries that English title + date as the text (the cited passage), points `url` at the PDF,
and archives the PDF itself (when download_pdfs=True) so the original stays available for audit or later OCR.

Press releases are reports by an investigating agency: they can support 'alleged' / 'reported', never a court finding.
"""
import html
import re
import urllib.parse

from backend.pipeline.common import in_window, make_message

BASE = "https://dvac.tn.gov.in/"
LIST_URL = BASE + "Press_Release.html"
PUBLISHER = "Directorate of Vigilance and Anti-Corruption, Government of Tamil Nadu"
LICENCE = "Government of Tamil Nadu website (no explicit reuse terms; attributed to DVAC)"
LINK = re.compile(r'<a\s+href="(pdf/Press_Release/(\d{4})/([^"]+?\.pdf))"[^>]*>(.*?)</a>', re.I | re.S)
FILE_DATE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})")


def clean(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def parse_list(page_html):
    """-> [{"url", "title", "date", "filename"}] for each press-release PDF on the list page."""
    items, seen = [], set()
    for m in LINK.finditer(page_html):
        rel, year, filename, label = m.groups()
        if rel in seen:
            continue
        seen.add(rel)
        d = FILE_DATE.match(filename)
        date = f"{d.group(3)}-{d.group(2)}-{d.group(1)}" if d else None
        items.append({"url": BASE + urllib.parse.quote(rel, safe="/"), "title": clean(label), "date": date or year, "filename": filename})
    return items


def collect(fetcher, store, sink, seen, limit=None, min_year=1947, max_year=2026, download_pdfs=True):
    """Fetch the list, then publish one message per new in-window press release. Returns counts."""
    page, _ = fetcher.get(LIST_URL)
    items = parse_list(page.decode("utf-8", "replace"))
    stats = {"listed": len(items), "published": 0, "skipped_seen": 0, "out_of_window": 0, "pdf_failed": 0}
    for it in items:
        if limit is not None and stats["published"] >= limit:
            break
        if it["url"] in seen:
            stats["skipped_seen"] += 1
            continue
        if not in_window(it["date"], min_year, max_year):
            stats["out_of_window"] += 1
            continue
        sha = raw_path = ctype = None
        if download_pdfs:
            try:
                data, ctype = fetcher.get(it["url"])
                sha, raw_path = store.save(data, ".pdf")
            except Exception:
                stats["pdf_failed"] += 1           # the English title is still collected; the PDF can be fetched on a later run
        text = f"DVAC press release dated {it['date']}: {it['title']}"
        sink.send(make_message(site="dvac", doc_type="press_release", title=it["title"][:240], url=it["url"], publisher=PUBLISHER,
                               licence=LICENCE, published_date=it["date"], raw_path=raw_path, sha256=sha, content_type=ctype,
                               text=text, text_reliable=True, extra={"pdf_text_reliable": False}))
        sink.flush()
        seen.add(it["url"])
        seen.save()
        stats["published"] += 1
    return stats
