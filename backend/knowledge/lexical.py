"""Keyword (BM25) search over the passages, with SQLite FTS5 (built into Python: no new package, no server, low memory).

Vector search finds passages that MEAN the same; it is weak on exact identifiers such as "Section 83(1)(a)". BM25 finds the passages that
contain those exact words and numbers. RagStore.hybrid_query fuses the two lists (reciprocal rank fusion).

    python -m backend.knowledge.lexical build --out backend/data/lexical.db       (your computer: every passage in Chroma)
    LEXICAL_DB=/path/lexical.db                                                    (where the service finds it; default backend/data/lexical.db)

A compound section number such as 83(1)(a) is indexed as the words 83, 1, a: the query adds the phrase "83 1 a" so a passage with the
exact number ranks above one that only has the three words somewhere.
"""
import argparse
import os
import re
import sqlite3
import threading
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "lexical.db"
STOP = {"the", "and", "for", "are", "was", "were", "what", "which", "who", "whom", "when", "where", "how", "why", "does", "did", "that", "this",
        "with", "from", "have", "has", "had", "its", "can", "you", "tell", "about", "give", "show", "list", "please", "say", "said", "under", "into",
        "provide", "provides", "section", "sections", "act", "court", "supreme", "india", "indian", "any", "all", "not", "but", "his", "her", "their"}
SECTION_ID = re.compile(r"\b(\d{1,3}[A-Za-z]?(?:\s*\(\s*\w{1,4}\s*\))+)")
SEC_REF = re.compile(r"\b(sections?|sec\.?|s\.|articles?|art\.?|rules?|order)\s*(\d{1,3}[A-Za-z]?(?:\s*\(\s*\w{1,4}\s*\))*)", re.I)
ACT_NAME = re.compile(r"(?:of|under|in)\s+(?:the\s+)?((?:[A-Z][\w’'.&-]*\s+){0,7}(?:Act|Code|Constitution(?:\s+of\s+India)?)(?:,?\s*\d{4})?)")
ALIASES = {"section": ("section", "s", "sec"), "sections": ("section", "s", "sec"), "sec": ("section", "s", "sec"), "s": ("section", "s", "sec"),
           "article": ("article", "art"), "articles": ("article", "art"), "art": ("article", "art"), "rule": ("rule",), "rules": ("rule",), "order": ("order",)}
MAX_TERMS = 18


def match_query(text):
    """The FTS5 MATCH expression for a question (words OR-ed, exact section numbers as phrases), or "" when nothing usable is left."""
    phrases = []
    for m in SECTION_ID.finditer(text or ""):
        toks = re.findall(r"[A-Za-z0-9]+", m.group(1).lower())
        if len(toks) >= 2:
            phrases.append('"' + " ".join(toks) + '"')
    for m in SEC_REF.finditer(text or ""):                       # "Section 83(1)(a)" is also written "s.83(1)(a)": all spellings as phrases
        ident = re.findall(r"[A-Za-z0-9]+", m.group(2).lower())
        for alias in ALIASES.get(m.group(1).lower().rstrip("."), ()):
            phrases.append('"' + " ".join([alias] + ident) + '"')
    words = []
    for w in re.findall(r"[A-Za-z0-9]+", (text or "").lower()):
        if (len(w) > 2 or (w.isdigit() and len(w) == 2)) and w not in STOP and w not in words:
            words.append(w)
    terms = list(dict.fromkeys(phrases)) + ['"' + w + '"' for w in words[:MAX_TERMS]]
    return " OR ".join(terms)


def exact_matcher(question):
    """A function(passage text) -> bool for a question that names a section ("Section 83(1)(a) of the Representation of the People Act"):
    True when the passage has that exact section number AND most words of the Act's name. None when the question names no section."""
    m = SEC_REF.search(question or "")
    if not m:
        return None
    toks = re.findall(r"[A-Za-z0-9]+", m.group(2))
    number = re.escape(toks[0]) + "".join(r"\s*\(\s*" + re.escape(t) + r"\s*\)" for t in toks[1:])
    section = re.compile(r"(?:sections?|sec\.?|s\.|articles?|art\.?|rules?|order)\s*" + r"(?<![\d(])" + number + r"(?![\d(a-zA-Z])", re.I)
    act = ACT_NAME.search(question)
    act_words = {w for w in re.findall(r"[a-z]{4,}", (act.group(1) if act else "").lower()) if w not in ("code", "india", "constitution")}
    need = -(-len(act_words) // 2)

    def matches(text):
        if not section.search(text or ""):
            return False
        return not act_words or len(act_words & set(re.findall(r"[a-z]{4,}", (text or "").lower()))) >= need
    return matches


class Lexical:
    def __init__(self, path):
        self.path = Path(path)
        self._local = threading.local()

    @classmethod
    def from_env(cls):
        """The keyword index if its file exists (LEXICAL_DB, else backend/data/lexical.db), else None: search then stays vector-only."""
        path = Path(os.environ.get("LEXICAL_DB") or DEFAULT_DB)
        return cls(path) if path.is_file() else None

    def _db(self):
        if getattr(self._local, "db", None) is None:
            self._local.db = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        return self._local.db

    def count(self):
        return self._db().execute("SELECT count(*) FROM passages").fetchone()[0]

    def search(self, text, k=30):
        """[(passage id, score)] best first (BM25; a higher score is better)."""
        q = match_query(text)
        if not q:
            return []
        try:
            rows = self._db().execute("SELECT id, bm25(passages) FROM passages WHERE passages MATCH ? ORDER BY bm25(passages) LIMIT ?", (q, k)).fetchall()
        except sqlite3.OperationalError:                     # a query SQLite cannot parse: no keyword results rather than an error
            return []
        return [(pid, -score) for pid, score in rows]

    @staticmethod
    def build(path, rows, batch=5000):
        """Write a NEW index from rows of (passage id, text). Returns how many passages were indexed."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} already exists; refusing to overwrite.")
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
        db.execute("CREATE VIRTUAL TABLE passages USING fts5(text, id UNINDEXED, tokenize='unicode61 remove_diacritics 2')")
        n, buf = 0, []
        for pid, text in rows:
            buf.append((text, pid))
            if len(buf) >= batch:
                db.executemany("INSERT INTO passages(text, id) VALUES (?, ?)", buf)
                n += len(buf)
                buf = []
        if buf:
            db.executemany("INSERT INTO passages(text, id) VALUES (?, ?)", buf)
            n += len(buf)
        db.execute("INSERT INTO passages(passages) VALUES ('optimize')")
        db.commit()
        db.close()
        return n


def chroma_rows(store, page=5000):
    """(id, text) of every passage in a RagStore."""
    offset = 0
    while True:
        got = store.col.get(limit=page, offset=offset, include=["documents"])
        if not got["ids"]:
            return
        yield from zip(got["ids"], got["documents"])
        offset += page


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="index every passage of the configured Chroma")
    b.add_argument("--out", type=Path, default=DEFAULT_DB)
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    from backend.knowledge.vectorstore import RagStore
    store = RagStore.from_env()
    if store is None:
        raise SystemExit("Chroma is not configured.")
    n = Lexical.build(a.out, chroma_rows(store))
    print(f"indexed {n} passages into {a.out} ({a.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
