"""Court and agency records, found person by person.

The Supreme Court and Madras High Court judgments (open datasets on AWS, CC-BY-4.0) and the DVAC and CBI press releases are public records, but
they are not organised by politician. This module builds a small local index of their METADATA (case title, parties, date, where the PDF is),
once, and then finds the records that name a given person:

    build(...)          read the metadata of every Supreme Court year (1950-), every Madras High Court bench/year, the DVAC list and the CBI list into
                        backend/data/kb_state/official_index.sqlite. Resumable: finished units are skipped, the current year is read again.
    match(index, person) the index rows whose title or parties name the person.

A match is only a candidate. The collector then downloads the document and keeps it only if its TEXT also names the person and shows corruption
wording (and, for the Supreme Court, a Tamil Nadu link): see person_sources.official_docs. Names are compared after a simple spelling
normalisation, because the records spell the same name many ways (Jayalalithaa / Jayalalitha / Jeyalalitha).

Sources in the index: the Supreme Court and Madras High Court (kinds `sc`, `hc`), and the agencies DVAC (`dvac`), CBI (`cbi`) and the Directorate of
Enforcement (`ed`). More agencies plug in through AGENCY_FEEDS below: a feed is a name, a URL and a function that turns the answer into rows.
Not (yet) possible: NIA and the TN Lokayukta (not reachable), the Income Tax Department (403), CVC and Lokpal (pages built in JavaScript, no list to read).
"""
import json
import re
import sqlite3
import urllib.parse
import threading
import unicodedata
from pathlib import Path

from backend.pipeline.collectors import cbi, court_opendata, dvac
from backend.pipeline.registry import STATE_DIR, now

INDEX_PATH = STATE_DIR / "official_index.sqlite"
SC_FIRST, HC_FIRST = 1950, 2000
NEUTRAL = {"versus", "vs", "the", "and", "of", "shri", "smt", "mrs", "state", "rep", "by", "dr", "sri", "selvi", "thiru", "tmt", "late", "hon", "mr", "ms"}
HONORIFICS = {"thiru", "tmt", "selvi", "dr", "mr", "mrs", "shri", "sri", "smt", "hon", "justice", "thiruvalar", "former", "minister", "chief"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, year INTEGER, date TEXT, title TEXT, parties TEXT, norm TEXT,
    bucket TEXT, key TEXT, url TEXT UNIQUE, meta TEXT);
CREATE INDEX IF NOT EXISTS rows_kind ON rows (kind, year);
CREATE TABLE IF NOT EXISTS units (unit TEXT PRIMARY KEY, rows INTEGER, at TEXT);
"""

_PAIRS = (("th", "t"), ("dh", "d"), ("kh", "k"), ("ph", "p"), ("gh", "g"), ("bh", "b"), ("sh", "s"), ("ee", "i"), ("oo", "u"), ("aa", "a"), ("w", "v"),
          ("zh", "l"), ("z", "l"), ("y", "i"))


def norm(text):
    """Lower case, no accents or punctuation, Indian-name spelling variants folded together, doubled letters collapsed."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z ]+", " ", text)
    for a, b in _PAIRS:
        text = text.replace(a, b)
    text = re.sub(r"(.)\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


ROLE = re.compile(r"chief minister|minister|mla|legislative assembly|leader of opposition|dmk|aiadmk|kazhagam|member of parliament|speaker|"
                  r"deputy chief|general secretary|party president|mp|former minister", re.I)


def person_tokens(person):
    """The distinctive normalised name parts of a person (initials and honorifics dropped), from the Wikidata name and the article title."""
    seen = []
    for source in (person["name"], person.get("en_title") or ""):
        for part in norm(source).split():
            if len(part) >= 4 and part not in HONORIFICS and part not in seen:
                seen.append(part)
    return seen


def family_name(person):
    words = [w for w in norm(person["name"]).split() if w not in HONORIFICS]
    if not words:                                                     # the Wikidata name is in Tamil script only: use the English article title
        words = [w for w in norm(person.get("en_title") or "").split() if w not in HONORIFICS]
    last = words[-1] if words else ""
    return last if len(last) >= 4 else (max(person_tokens(person), key=len) if person_tokens(person) else "")


def initials_of(person):
    """First letters of every part of the name and of the article title: 'Muthuvel Karunanidhi' / 'M. Karunanidhi' -> {m, k}."""
    out = set()
    for source in (person["name"], person.get("en_title") or ""):
        out |= {w[0] for w in norm(source).split() if w and w not in HONORIFICS}
    return out


def candidate_title(norm_title, person):
    """Could this case title (normalised) be about the person? Precision first: Indian family names are shared by thousands of litigants.
    The family name must be there, and either another distinctive name part is, or the family name is long (7+ letters) and the initials
    written before it (if any) are the person's."""
    fam = family_name(person)
    if not fam or fam not in norm_title:
        return False
    others = [t for t in person_tokens(person) if t != fam and len(t) >= 5]
    if any(t in norm_title for t in others):
        return True
    if len(fam) < 7:
        return False
    words = norm_title.split()
    for i, w in enumerate(words):
        if fam in w:
            if i and len(words[i - 1]) >= 3 and words[i - 1] not in NEUTRAL and words[i - 1] not in person_tokens(person):
                return False                                  # another given name before the family name: a relative or a namesake
            before = []
            for prev in reversed(words[:i]):
                if len(prev) <= 2:
                    before.append(prev)
                else:
                    break
            return not before or bool(set(before) & initials_of(person))
    return False


def confirms_text(norm_text, person, raw_text=""):
    """Does the document itself show this is the politician? The family name must appear next to a political role (minister, MLA, chief
    minister, party office ...) or together with another distinctive part of the name."""
    fam = family_name(person)
    if not fam or fam not in norm_text:
        return False
    if any(t in norm_text for t in person_tokens(person) if t != fam and len(t) >= 6):
        return True
    for m in re.finditer(re.escape(fam), norm_text):
        if ROLE.search(norm_text[max(0, m.start() - 200):m.end() + 200]):
            return True
    return False


class OfficialIndex:
    def __init__(self, path=INDEX_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._memory = None

    def unit_done(self, unit):
        with self._lock:
            return self._db.execute("SELECT 1 FROM units WHERE unit=?", (unit,)).fetchone() is not None

    def add_rows(self, unit, rows):
        with self._lock:
            for r in rows:
                self._db.execute(
                    "INSERT OR IGNORE INTO rows (kind, year, date, title, parties, norm, bucket, key, url, meta) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (r["kind"], r.get("year"), r.get("date"), r.get("title"), r.get("parties"), norm(f"{r.get('title') or ''} {r.get('parties') or ''}"),
                     r.get("bucket"), r.get("key"), r["url"], json.dumps(r.get("meta") or {}, ensure_ascii=False)))
            self._db.execute("INSERT OR REPLACE INTO units (unit, rows, at) VALUES (?,?,?)", (unit, len(rows), now()))
            self._db.commit()
            self._memory = None

    def counts(self):
        with self._lock:
            return {r["kind"]: r["n"] for r in self._db.execute("SELECT kind, COUNT(*) AS n FROM rows GROUP BY kind")}

    def _loaded(self):
        """(id, kind, norm) of every row, kept in memory: a person is matched against ~300,000 titles in well under a second."""
        with self._lock:
            if self._memory is None:
                self._memory = [(r["id"], r["kind"], r["norm"] or "") for r in self._db.execute("SELECT id, kind, norm FROM rows")]
            return self._memory

    def match(self, person, limit=40):
        """Candidate rows (dicts) for a person, newest first, at most `limit` (court rows first, then agency rows)."""
        ids = [i for i, _, n in self._loaded() if candidate_title(n, person)]
        if not ids:
            return []
        with self._lock:
            marks = ",".join("?" * len(ids))
            rows = [dict(r) for r in self._db.execute(f"SELECT * FROM rows WHERE id IN ({marks}) ORDER BY COALESCE(date,'') DESC", ids)]
        for r in rows:
            r["meta"] = json.loads(r["meta"] or "{}")
        order = {"sc": 0, "hc": 1, "ed": 2, "dvac": 3, "cbi": 4}
        rows.sort(key=lambda r: order.get(r["kind"], 9))
        return rows[:limit]


ED_BASE = "https://enforcementdirectorate.gov.in"
ED_FEED = ED_BASE + "/umbraco/backoffice/api/PressReleaseApi/GetPublic?language=en"       # the public list the press-release page itself loads


def parse_ed(payload):
    """Rows of the Directorate of Enforcement press-release feed: [{Id, Description, ReleaseDate, PdfUrl}]."""
    out = []
    for r in json.loads(payload.decode("utf-8-sig")):
        pdf = r.get("PdfUrl")
        if not pdf or not r.get("Description"):
            continue
        date = (r.get("ReleaseDate") or "")[:10] or None
        out.append({"kind": "ed", "year": int(date[:4]) if date else None, "date": date, "title": r["Description"],
                    "url": ED_BASE + urllib.parse.quote(pdf, safe="/%"), "meta": {"id": r.get("Id")}})
    return out


# name, URL, parser (bytes -> rows). Every feed is fetched with the polite, robots-respecting fetcher.
AGENCY_FEEDS = (("agency:ed", ED_FEED, parse_ed),)


# ---- building the index -------------------------------------------------------------------------------------------
def _court_rows(source, unit):
    """Metadata rows of one Supreme Court year or one Madras High Court bench/year (all of them, not only the agency cases)."""
    court, year, bench = unit
    if court == "sc":
        cols = ["title", "petitioner", "respondent", "decision_date", "path", "case_id", "judge", "disposal_nature"]
        rows = source.table(court_opendata.SC_BUCKET, f"metadata/parquet/year={year}/metadata.parquet", cols)
        return [{"kind": "sc", "year": year, "date": court_opendata.iso_date(r.get("decision_date")), "title": r.get("title"),
                 "parties": f"{r.get('petitioner') or ''} {r.get('respondent') or ''}", "bucket": court_opendata.SC_BUCKET,
                 "key": f"data/pdf/year={year}/english/{r['path']}_EN.pdf", "url": f"https://{court_opendata.SC_BUCKET}.s3.amazonaws.com/data/pdf/year={year}/english/{r['path']}_EN.pdf",
                 "meta": {"judge": r.get("judge"), "outcome": r.get("disposal_nature"), "ref": r.get("case_id")}} for r in rows if r.get("path")]
    cols = ["title", "pdf_link", "decision_date", "disposal_nature", "judge", "cnr"]
    rows = source.table(court_opendata.HC_BUCKET, f"metadata/parquet/year={year}/court={court_opendata.MADRAS}/bench={bench}/metadata.parquet", cols)
    out = []
    for r in rows:
        if not r.get("pdf_link"):
            continue
        key = f"data/pdf/year={year}/court={court_opendata.MADRAS}/bench={bench}/{r['pdf_link'].rsplit('/', 1)[-1]}"
        out.append({"kind": "hc", "year": year, "date": court_opendata.iso_date(r.get("decision_date")), "title": r.get("title"), "parties": "",
                    "bucket": court_opendata.HC_BUCKET, "key": key, "url": f"https://{court_opendata.HC_BUCKET}.s3.amazonaws.com/{key}",
                    "meta": {"judge": r.get("judge"), "outcome": r.get("disposal_nature"), "ref": r.get("cnr"), "bench": bench}})
    return out


def build(index, fetcher, should_stop=lambda: False, say=print, current_year=2026, kinds=None):
    """Fill the index. Returns True when every unit is done, False when it was stopped half way (call again to resume).
    `kinds` limits the work to some of sc, hc, dvac, cbi, ed (None = all): a request for ED releases only does not read the court metadata."""
    want = lambda k: kinds is None or k in kinds                                                      # noqa: E731
    try:
        if not (want("sc") or want("hc")):
            raise RuntimeError("not needed")
        source = court_opendata.S3Source()
    except Exception as exc:  # noqa: BLE001 - pyarrow missing or no network: the person stages still work without court records
        if str(exc) != "not needed":
            say(f"court index: the open datasets are not reachable ({exc})")
        source = None
    units = []
    if source is not None:
        for year in range(SC_FIRST, current_year + 1):
            if want("sc"):
                units.append(("sc", year, None))
        for year in range(HC_FIRST, current_year + 1):
            if not want("hc"):
                break
            try:
                units += court_opendata.scan_units(source, "hc", year)
            except Exception:  # noqa: BLE001
                continue
    for unit in units:
        name = "court:" + ":".join(str(x) for x in unit if x is not None)
        if should_stop():
            return False
        if index.unit_done(name) and unit[1] < current_year:
            continue
        try:
            index.add_rows(name, _court_rows(source, unit))
            say(f"court index: {name}")
        except Exception as exc:  # noqa: BLE001 - a missing year is normal (no metadata file)
            index.add_rows(name, [])
            say(f"court index: {name} has no metadata ({str(exc)[:60]})")
    for name, page_url, parse, kind in (("agency:dvac", dvac.LIST_URL, dvac.parse_list, "dvac"), ("agency:cbi", cbi.LIST_URL, cbi.parse_list, "cbi")):
        if not want(kind):
            continue
        if should_stop():
            return False
        try:
            page, _ = fetcher.get(page_url)
            items = parse(page.decode("utf-8", "replace"))
            index.add_rows(name, [{"kind": kind, "year": int(str(it.get("date") or "0")[:4] or 0) or None, "date": it.get("date"), "title": it["title"], "parties": "",
                                   "url": it["url"], "meta": {"filename": it.get("filename")}} for it in items])
            say(f"agency index: {name} {len(items)} entries")
        except Exception as exc:  # noqa: BLE001
            say(f"agency index: {name} unavailable ({str(exc)[:80]})")
    for name, url, parse in AGENCY_FEEDS:
        if not want(name.split(":")[1]):
            continue
        if should_stop():
            return False
        try:
            payload, _ = fetcher.get(url)
            rows = parse(payload)
            index.add_rows(name, rows)
            say(f"agency index: {name} {len(rows)} entries")
        except Exception as exc:  # noqa: BLE001
            say(f"agency index: {name} unavailable ({str(exc)[:80]})")
    return True
