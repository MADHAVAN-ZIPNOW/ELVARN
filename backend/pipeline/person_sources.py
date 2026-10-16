"""Where one politician's documents come from, and how they are archived.

Sources (all open: Wikipedia text is CC BY-SA 4.0, Wikidata is CC0; nothing from news sites, nothing behind a login):

    wikidata            the Wikidata record of the person: party, positions with dates, relatives, birth place ... (the relationships)
    wikipedia_en        the English Wikipedia article about the person
    wikipedia_ta        the Tamil Wikipedia article about the person
    corruption_articles English Wikipedia articles about cases, scams and inquiries that name the person (only when the article is about
                        Tamil Nadu AND corruption AND really mentions the person)
    news_records        News reports about the person's cases: found through GDELT (an open news index, about the last 3 months), then the article page itself
                        is read where the site's robots.txt allows (the owner approved this). The article belongs to its publisher: it is kept in the
                        private stores with its link, shown only as short quotes, and never put in the public snapshot. Where the page cannot be read
                        (robots.txt, paywall, error) the headline record from the index is kept instead.
    official_records    Supreme Court and Madras High Court judgments (open datasets, CC-BY-4.0), DVAC and CBI press releases that name the person.
                        Found through the local metadata index (official_index.py); kept only when the document's own text names the person too.

The Wikimedia sites are read through their official APIs with the project's descriptive User-Agent, one request every half second
and a back-off on errors. (Their robots.txt keeps generic crawlers off /api/; the API itself is open to identified bots under the
Wikimedia API policy, which is how the project's other Wikipedia collector already works.) Other websites keep the robots.txt check.

Every original is archived untouched under kb_raw/{source}/{yyyy-mm-dd}/{doc_id}.{ext}; doc_id = "doc_" + sha256(bytes)[:20].
"""
import hashlib
import json
import os
import re
import threading
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backend.pipeline import law_tags, official_index
from backend.pipeline.collectors import cbi, court_opendata, dvac
from backend.pipeline.common import RAW_DIR, FetchError, Fetcher, is_tamil_nadu
from backend.pipeline.roster import UA

CORRUPTION = re.compile(r"corrupt|bribe|scam|disproportionate|embezzl|vigilance|money.?laundering|graft|convicted|acquitted|charge.?sheet|\bFIR\b|"
                        r"commission of inquiry|misappropriat|illegal|fraud", re.I)
SEARCH_TERMS = ('corruption OR scam OR "disproportionate assets" OR bribery OR "money laundering" OR chargesheet OR convicted OR acquitted '
                'OR "commission of inquiry"')
TAG = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S | re.I)
MAX_CORRUPTION_ARTICLES = 8
NEWS_TERMS = re.compile(r"corrupt|scam|bribe|graft|\bED\b|enforcement directorate|income.?tax|\bI-?T\b|\bCBI\b|\bDVAC\b|vigilance|probe|raid|search(es)?\b|"
                        r"\bFIR\b|charge.?sheet|arrest|case|court|\bHC\b|petition|penalt|disproportionate|assets|money.?laundering|PMLA|summon|bail|convict|acquit|"
                        r"allegation|inquiry|lokayukta|tribunal|dismiss|quash", re.I)
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWS_LICENCE = "Headline and link only, from the GDELT open news index; the article belongs to its publisher"
ARTICLE_LICENCE = "Copyright of the publisher; kept privately for research with its link, shown only as short quotes, not redistributed"
MAX_NEWS = 25
MAX_ARTICLES = 10                                            # full pages read per person (the rest stay headline records)
MIN_ARTICLE_CHARS = 1500                                     # a shorter page is a paywall stub or an error page, not an article
ED_PUBLISHER = "Directorate of Enforcement, Government of India"
ED_LICENCE = "Directorate of Enforcement website (no explicit reuse terms; attributed to the Directorate of Enforcement)"


@dataclass
class Found:
    """One downloaded file, not yet archived."""
    topic: str                      # person | corruption | structured
    source: str                     # web | pdf | dataset | upload
    ext: str                        # html | json | pdf
    mime: str
    url: str
    title: str
    language: str
    publisher: str
    licence: str
    doc_type: str
    data: bytes
    wikidata_id: str = None
    companion: bytes = None         # a second archived file that belongs to this one (the resolved Wikidata labels); it gets no message of its own
    tokens: int = 0
    extra: dict = field(default_factory=dict)


def visible_text(html_bytes):
    """Crude plain text of an HTML page (only used to judge relevance and to estimate size; Spark does the real cleaning)."""
    return re.sub(r"\s+", " ", TAG.sub(" ", html_bytes.decode("utf-8", "replace")))


def tokens_of(text):
    return -(-len(text) // 4)


def name_variants(person):
    """The spellings that count as naming the person: full name, article title, and (for names of 5+ letters) the family name."""
    last = person["name"].replace(".", " ").split()[-1] if person["name"].split() else ""
    names = {person["name"], person.get("en_title") or "", last if len(last) >= 5 else ""}
    return [n.lower() for n in names if len(n) >= 4]


def mentions(text, person):
    low = text.lower()
    return any(n in low for n in name_variants(person))


def near_corruption(text, person, window=250):
    """How many times the person is named within `window` characters of a corruption word. This is what separates an article ABOUT a case
    involving the person from a long article (a party, a government) that merely names them and, elsewhere, mentions corruption."""
    low, hits, last_end = text.lower(), 0, -1
    spans = sorted(m.start() for n in name_variants(person) for m in re.finditer(re.escape(n), low))
    for start in spans:
        if start < last_end:
            continue                                   # the same stretch of text is counted once
        if CORRUPTION.search(text[max(0, start - window):start + window]):
            hits += 1
            last_end = start + window
    return hits


class Wikimedia:
    """Polite client for the Wikipedia / Wikidata APIs. Thread-safe (the Fetcher serialises the request rate)."""

    def __init__(self, fetcher=None, delay=0.5):
        self.fetcher = fetcher or Fetcher(delay=delay, respect_robots=False)
        self.headers = {"User-Agent": UA, "Accept-Encoding": "identity"}

    def get(self, url):
        return self.fetcher.request(url, headers=self.headers)

    def json(self, url):
        data, _ = self.get(url)
        return json.loads(data.decode("utf-8"))

    def article_html(self, lang, title):
        """(bytes, canonical url) of the article as Parsoid HTML, or None when the page does not exist."""
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{urllib.parse.quote(title.replace(' ', '_'), safe='')}"
        try:
            data, _ = self.get(url)
        except FetchError as exc:
            if "404" in str(exc) or "410" in str(exc):
                return None
            raise
        return data, f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

    def entity_json(self, qid):
        try:
            data, _ = self.get(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
        except FetchError as exc:
            if "404" in str(exc):
                return None
            raise
        return data

    def labels(self, qids):
        """{QID: English label} for the given Wikidata items (50 per request)."""
        out = {}
        qids = sorted(set(qids))
        for i in range(0, len(qids), 50):
            url = ("https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&languages=en&format=json&maxlag=5&ids="
                   + "|".join(qids[i:i + 50]))
            for qid, ent in (self.json(url).get("entities") or {}).items():
                label = ((ent.get("labels") or {}).get("en") or {}).get("value")
                if label:
                    out[qid] = label
        return out

    def search(self, query, limit=12):
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&formatversion=2&srnamespace=0&maxlag=5"
               f"&srlimit={limit}&srprop=&srsearch={urllib.parse.quote(query)}")
        return [h["title"] for h in self.json(url).get("query", {}).get("search", [])]


def linked_items(entity_json, qid, limit=300):
    """Every item id a Wikidata record points at (party, positions, places, relatives, qualifiers): the names the processing step needs."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            if node.get("entity-type") == "item" and node.get("id"):
                found.add(node["id"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(((entity_json.get("entities") or {}).get(qid) or {}).get("claims") or {})
    return sorted(found)[:limit]


def gdelt_date(seen):
    """'20261016T074500Z' -> '2026-10-16'."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})", seen or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def news_query(person):
    title = re.sub(r"\s*\(.*?\)", "", person.get("en_title") or person["name"]).strip()
    return (f'"{title}" (corruption OR scam OR "Enforcement Directorate" OR "income tax" OR "disproportionate assets" OR CBI OR DVAC OR probe OR raid '
            f'OR FIR OR arrest OR court OR penalty OR petition) sourcelang:english')


def news_docs(fetch, person, limit=MAX_NEWS, fetch_article=None, max_articles=MAX_ARTICLES):
    """Headlines about the person's cases. `fetch(url)` -> bytes (a polite, robots-respecting fetcher). One Found per headline: the GDELT record
    itself, as JSON. Kept only when the HEADLINE names the person and carries a legal / corruption word (the index matches on the article's text,
    which we do not have, so a headline that does not mention the person is not evidence about them)."""
    url = GDELT + "?" + urllib.parse.urlencode({"query": news_query(person), "mode": "artlist", "maxrecords": 75, "format": "json", "sort": "datedesc"})
    body = fetch(url)
    try:
        articles = json.loads(body.decode("utf-8", "replace")).get("articles") or []
    except ValueError:
        return []                                            # GDELT answers a plain-text error when a query is malformed or the service is busy
    names = name_variants(person)
    out, seen = [], set()
    for a in articles:
        title, link = (a.get("title") or "").strip(), a.get("url")
        low = title.lower()
        key = re.sub(r"[^a-z0-9]+", " ", low).strip()                    # the same headline syndicated under another link counts once
        if not title or not link or link in seen or key in seen:
            continue
        if not (any(n in low or n.split()[-1] in low for n in names) and NEWS_TERMS.search(title)):
            continue
        seen.update((link, key))
        out.append(Found("news", "dataset", "json", "application/json", link, title[:240], "en", a.get("domain") or "news", NEWS_LICENCE, "news",
                         json.dumps(a, ensure_ascii=False, sort_keys=True).encode("utf-8"), wikidata_id=person.get("wikidata_id"), tokens=tokens_of(title),
                         extra={"published_date": gdelt_date(a.get("seendate"))}))
        if len(out) >= limit:
            break
    if fetch_article is not None:
        out = [article_or_headline(f, fetch_article, person) if i < max_articles else f for i, f in enumerate(out)]
    return out


def article_or_headline(headline, fetch_article, person):
    """The article page as the original HTML when it can be read and really is about the person; otherwise the headline record."""
    try:
        data, ctype = fetch_article(headline.url)
    except FetchError:                                           # robots.txt, 403 / paywall, 404, timeout: the headline record stands
        return headline
    if "html" not in (ctype or "").lower():
        return headline
    text = visible_text(data)
    if len(text) < MIN_ARTICLE_CHARS or not mentions(text, person) or not NEWS_TERMS.search(text):
        return headline
    return Found("news", "web", "html", "text/html", headline.url, headline.title, "en", headline.publisher, ARTICLE_LICENCE, "news", data,
                 wikidata_id=headline.wikidata_id, tokens=tokens_of(text), extra=headline.extra)


def wikipedia_url(lang, title):
    return f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"


def wikidata_doc(wm, person):
    data = wm.entity_json(person["wikidata_id"]) if person.get("wikidata_id") else None
    if not data:
        return []
    labels = wm.labels(linked_items(json.loads(data.decode("utf-8")), person["wikidata_id"]))
    companion = json.dumps({"for": person["wikidata_id"], "labels": labels}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return [Found("structured", "dataset", "json", "application/json", f"https://www.wikidata.org/wiki/{person['wikidata_id']}",
                  f"{person['name']} (Wikidata)", "en", "Wikidata", "CC0 1.0", "reference", data, wikidata_id=person["wikidata_id"], companion=companion)]


def wikipedia_doc(wm, person, lang):
    title = person.get("en_title" if lang == "en" else "ta_title")
    if not title:
        return []
    got = wm.article_html(lang, title)
    if not got:
        return []
    data, url = got
    return [Found("person", "web", "html", "text/html", url, f"{title} (Wikipedia)", lang, "Wikipedia", "CC BY-SA 4.0", "reference", data,
                  wikidata_id=person.get("wikidata_id"), tokens=tokens_of(visible_text(data)))]


def corruption_candidates(wm, person):
    """Titles of English Wikipedia articles that might be about a case involving the person (not their own article).
    Searches under the article title, the Wikidata name and the family name, because articles use any of them."""
    own = (person.get("en_title") or "").lower()
    last = person["name"].replace(".", " ").split()[-1] if person["name"].split() else ""
    phrases = [f'"{n}"' for n in dict.fromkeys(x for x in (person.get("en_title"), person["name"]) if x)]
    if len(last) >= 6:
        phrases.append(last)
    titles = []
    for phrase in phrases:
        for t in wm.search(f"{phrase} ({SEARCH_TERMS})"):
            if t.lower() != own and t not in titles:
                titles.append(t)
    return titles[:MAX_CORRUPTION_ARTICLES + 8]


def corruption_doc(wm, person, title):
    """The article if it is about Tamil Nadu, about corruption AND names the person; otherwise nothing (and nothing is saved)."""
    got = wm.article_html("en", title)
    if not got:
        return None
    data, url = got
    text = visible_text(data)
    near = near_corruption(text, person)
    in_title = any(n in title.lower() for n in name_variants(person))
    if not (is_tamil_nadu(text) and CORRUPTION.search(text) and mentions(text, person) and (near >= 3 or (in_title and near >= 1))):
        return None
    return Found("corruption", "web", "html", "text/html", url, f"{title} (Wikipedia)", "en", "Wikipedia", "CC BY-SA 4.0", "reference", data,
                 wikidata_id=person.get("wikidata_id"), tokens=tokens_of(text))


class RawArchive:
    """Writes originals once, under kb_raw/{source}/{date}/{doc_id}.{ext}, and keeps kb_raw/checksums.txt. Never rewrites a file."""

    def __init__(self, root=RAW_DIR, today=None):
        self.root = Path(root)
        self.today = today or (lambda: datetime.now(timezone.utc).date().isoformat())
        self._lock = threading.Lock()

    @staticmethod
    def doc_id_of(data):
        sha = hashlib.sha256(data).hexdigest()
        return "doc_" + sha[:20], sha

    def save(self, data, source, ext):
        """-> (doc_id, sha256, raw_path "kb_raw/...", size, created). A file with the same bytes already on disk is not written again."""
        doc_id, sha = self.doc_id_of(data)
        rel = Path(source) / self.today() / f"{doc_id}.{ext}"
        path = self.root / rel
        with self._lock:
            created = False
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".part")
                tmp.write_bytes(data)
                os.replace(tmp, path)                       # a half-written file is never visible under its final name
                created = True
                with open(self.root / "checksums.txt", "a", encoding="utf-8") as f:
                    f.write(f"{sha}  {rel.as_posix()}\n")
        return doc_id, sha, "kb_raw/" + rel.as_posix(), len(data), created

    def exists(self, raw_path):
        return (self.root / raw_path.removeprefix("kb_raw/")).is_file()


# ---- official records: courts and agencies ------------------------------------------------------------------------
def official_doc(row, person, s3, fetcher):
    """Download one candidate record from the official index and keep it only if its own text names the person. -> Found or None."""
    kind = row["kind"]
    if kind in ("sc", "hc"):
        data = s3.read(row["bucket"], row["key"])
        text = court_opendata.pdf_text(data)
        if len(text.strip()) < 500:
            return None                                              # scanned or empty: nothing to learn from it
        if not (official_index.confirms_text(official_index.norm(text[:80000]), person) and (kind == "hc" or court_opendata.TN_TERMS.search(text))):
            return None                                                  # every case that names this politician (a petition, an appeal, an election case ...), not only corruption cases
        sc = kind == "sc"
        return Found("court_record", "pdf", "pdf", "application/pdf", row["url"], (row["title"] or "Judgment")[:240], "en",
                     court_opendata.SC_PUBLISHER if sc else court_opendata.HC_PUBLISHER, court_opendata.LICENCE, "judgment" if sc else "order", data,
                     wikidata_id=person.get("wikidata_id"), tokens=tokens_of(text),
                     extra={"published_date": row.get("date"), **row["meta"], "case_stage": law_tags.stage_of(kind, text, row["meta"].get("outcome")), "petitions": law_tags.petitions_of(text[:20000] + " " + (row.get("title") or ""))})
    if kind == "ed":
        data, ctype = fetcher.get(row["url"])
        text = court_opendata.pdf_text(data)
        if not official_index.confirms_text(official_index.norm(row["title"] + " " + text[:60000]), person):
            return None                                                                  # the name is only a lookalike
        return Found("agency_record", "pdf", "pdf", ctype or "application/pdf", row["url"], row["title"][:240], "en", ED_PUBLISHER, ED_LICENCE,
                     "press_release", data, wikidata_id=person.get("wikidata_id"), tokens=tokens_of(row["title"] + " " + text),
                     extra={"published_date": row.get("date"), "text": row["title"], "text_reliable": True, "case_stage": law_tags.stage_of("ed", row["title"] + " " + text[:3000])})
    if kind == "dvac":
        data, ctype = fetcher.get(row["url"])
        text = f"DVAC press release dated {row.get('date')}: {row['title']}"
        return Found("agency_record", "pdf", "pdf", ctype or "application/pdf", row["url"], row["title"][:240], "en", dvac.PUBLISHER, dvac.LICENCE,
                     "press_release", data, wikidata_id=person.get("wikidata_id"), tokens=tokens_of(text),
                     extra={"published_date": row.get("date"), "text": text, "text_reliable": False, "case_stage": law_tags.stage_of("dvac", row["title"])})       # the Tamil PDF text is unreliable; the English title is not
    if kind == "cbi":
        raw, ctype = fetcher.get(row["url"])
        detail = cbi.parse_detail(raw.decode("utf-8", "replace"))
        if detail is None or not is_tamil_nadu(detail["title"], detail["text"]):
            return None
        if not official_index.confirms_text(official_index.norm(detail["title"] + " " + detail["text"]), person):
            return None
        return Found("agency_record", "web", "html", "text/html", row["url"], detail["title"][:240], "en", cbi.PUBLISHER, cbi.LICENCE, "press_release", raw,
                     wikidata_id=person.get("wikidata_id"), tokens=tokens_of(detail["text"]),
                     extra={"published_date": detail["date"], "case_stage": law_tags.stage_of("cbi", detail["title"] + " " + detail["text"][:3000])})
    return None
