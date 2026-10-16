"""The collector's own bookkeeping: who is on the roster, which files were downloaded, and what moved where.

One SQLite file (backend/data/kb_state/collector.sqlite, WAL mode, safe to read from the API while the collector writes):

    people      the roster, one row per politician, with a status the collector resumes from
    documents   one row per downloaded file (sha256 is unique: the same bytes are never stored twice)
    person_docs which documents belong to which person (an article about a scandal is downloaded once and linked to everyone it names)
    doc_stages  when a document was first seen in each later stage (kafka, spark, corpus, neo4j, chroma): the dashboard's movement feed
    events      a short log line per collected document or problem
    meta        counters (retries, failures)

Nothing here talks to Kafka, Neo4j or Chroma; the monitor and the collector call these functions.
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "kb_state"
DB_PATH = STATE_DIR / "collector.sqlite"

STAGES = ("collected", "raw", "kafka", "spark", "corpus", "neo4j", "chroma")

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    key TEXT PRIMARY KEY, name TEXT NOT NULL, wikidata_id TEXT, party TEXT, first_year INTEGER, last_year INTEGER,
    positions TEXT NOT NULL DEFAULT '[]', en_title TEXT, ta_title TEXT, priority INTEGER NOT NULL DEFAULT 2,
    status TEXT NOT NULL DEFAULT 'pending', stage TEXT, sources_done TEXT NOT NULL DEFAULT '[]',
    docs INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0, corruption_docs INTEGER NOT NULL DEFAULT 0,
    started_at TEXT, finished_at TEXT, updated_at TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS people_order ON people (status, priority, first_year, name);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, person_key TEXT, topic TEXT, source TEXT, url TEXT, title TEXT, raw_path TEXT, sha256 TEXT UNIQUE,
    mime TEXT, bytes INTEGER, language TEXT, collected_at TEXT, status TEXT NOT NULL DEFAULT 'saved', tokens INTEGER NOT NULL DEFAULT 0,
    labels_path TEXT, doc_type TEXT, publisher TEXT, licence TEXT, published_date TEXT
);
CREATE INDEX IF NOT EXISTS documents_person ON documents (person_key);
CREATE INDEX IF NOT EXISTS documents_status ON documents (status);
CREATE TABLE IF NOT EXISTS person_docs (person_key TEXT, doc_id TEXT, topic TEXT, graph_linked INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (person_key, doc_id));
CREATE INDEX IF NOT EXISTS person_docs_doc ON person_docs (doc_id);
CREATE TABLE IF NOT EXISTS doc_stages (doc_id TEXT, stage TEXT, at TEXT, PRIMARY KEY (doc_id, stage));
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, doc_id TEXT, title TEXT, person TEXT, stage TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v INTEGER NOT NULL DEFAULT 0);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    """Thread-safe: one connection, every call under a lock (the collector uses 4 download threads)."""

    def __init__(self, path=DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        if "labels_path" not in {row[1] for row in self._db.execute("PRAGMA table_info(documents)")}:
            self._db.execute("ALTER TABLE documents ADD COLUMN labels_path TEXT")       # a database made before the companion files existed
        have = {row[1] for row in self._db.execute("PRAGMA table_info(documents)")}
        for column in ("doc_type", "publisher", "licence", "published_date"):                # what a message needs to be sent again exactly as the first time
            if column not in have:
                self._db.execute(f"ALTER TABLE documents ADD COLUMN {column} TEXT")
        if "graph_linked" not in {row[1] for row in self._db.execute("PRAGMA table_info(person_docs)")}:
            self._db.execute("ALTER TABLE person_docs ADD COLUMN graph_linked INTEGER NOT NULL DEFAULT 0")

    def close(self):
        with self._lock:
            self._db.close()

    def _all(self, sql, *args):
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _one(self, sql, *args):
        rows = self._all(sql, *args)
        return rows[0] if rows else None

    def _exec(self, sql, *args):
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    # ---- roster ------------------------------------------------------------------------------------------
    def upsert_person(self, p):
        """Insert a roster entry, or refresh its descriptive fields without touching its progress."""
        with self._lock:
            self._db.execute(
                "INSERT INTO people (key, name, wikidata_id, party, first_year, last_year, positions, en_title, ta_title, priority, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET name=excluded.name, wikidata_id=excluded.wikidata_id, "
                "party=excluded.party, first_year=excluded.first_year, last_year=excluded.last_year, positions=excluded.positions, "
                "en_title=excluded.en_title, ta_title=excluded.ta_title, priority=excluded.priority",
                (p["key"], p["name"], p.get("wikidata_id"), p.get("party"), p.get("first_year"), p.get("last_year"),
                 json.dumps(p.get("positions", []), ensure_ascii=False), p.get("en_title"), p.get("ta_title"), p.get("priority", 2), now()))
            self._db.commit()

    def person(self, key):
        row = self._one("SELECT * FROM people WHERE key=?", key)
        return self._decode(row) if row else None

    @staticmethod
    def _decode(row):
        row["positions"] = json.loads(row.get("positions") or "[]")
        row["sources_done"] = json.loads(row.get("sources_done") or "[]")
        return row

    def claim_next(self):
        """The person to work on now: one already in progress (resume), else the next pending in roster order. None when the roster is done."""
        with self._lock:
            row = self._one("SELECT * FROM people WHERE status='in_progress' ORDER BY priority, first_year, name LIMIT 1") \
                or self._one("SELECT * FROM people WHERE status='pending' ORDER BY priority, COALESCE(first_year, 9999), name LIMIT 1")
            if row is None:
                return None
            self._exec("UPDATE people SET status='in_progress', started_at=COALESCE(started_at, ?), updated_at=? WHERE key=?", now(), now(), row["key"])
            return self.person(row["key"])

    def set_stage(self, key, stage):
        self._exec("UPDATE people SET stage=?, updated_at=? WHERE key=?", stage, now(), key)

    def source_done(self, key, source):
        with self._lock:
            p = self.person(key)
            done = sorted(set(p["sources_done"]) | {source})
            self._exec("UPDATE people SET sources_done=?, updated_at=? WHERE key=?", json.dumps(done), now(), key)

    def finish_person(self, key, ok=True, error=None):
        self._exec("UPDATE people SET status=?, stage=NULL, finished_at=?, updated_at=?, error=? WHERE key=?",
                   "done" if ok else "failed", now(), now(), error, key)

    def requeue_failed(self):
        self._exec("UPDATE people SET status='pending', error=NULL WHERE status='failed'")

    def counts(self):
        out = {"total": 0, "done": 0, "in_progress": 0, "pending": 0, "failed": 0}
        for r in self._all("SELECT status, COUNT(*) AS n FROM people GROUP BY status"):
            out[r["status"]] = r["n"]
            out["total"] += r["n"]
        return out

    def queue(self, limit=8):
        return self._all("SELECT key, name FROM people WHERE status='pending' ORDER BY priority, COALESCE(first_year, 9999), name LIMIT ?", limit)

    def recent_people(self, limit=8):
        return self._all("SELECT key, name, status, docs, finished_at FROM people WHERE status IN ('done','failed') "
                         "ORDER BY finished_at DESC LIMIT ?", limit)

    def people(self, status=None, q=None, limit=50, offset=0):
        where, args = [], []
        if status:
            where.append("status=?")
            args.append(status)
        if q:
            where.append("name LIKE ?")
            args.append(f"%{q}%")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        total = self._one(f"SELECT COUNT(*) AS n FROM people {clause}", *args)["n"]
        rows = self._all(f"SELECT * FROM people {clause} ORDER BY priority, COALESCE(first_year, 9999), name LIMIT ? OFFSET ?", *args, limit, offset)
        return total, [self._decode(r) for r in rows]

    # ---- documents ---------------------------------------------------------------------------------------
    def has_sha(self, sha):
        return self._one("SELECT doc_id FROM documents WHERE sha256=?", sha) is not None

    def add_document(self, d):
        """Record a downloaded file. Returns False (and records nothing) if a file with the same sha256 is already registered."""
        with self._lock:
            if self.has_sha(d["sha256"]):
                return False
            self._db.execute(
                "INSERT INTO documents (doc_id, person_key, topic, source, url, title, raw_path, sha256, mime, bytes, language, collected_at, status, tokens, labels_path, "
                "doc_type, publisher, licence, published_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (d["doc_id"], d.get("person_key"), d.get("topic"), d["source"], d.get("url"), d.get("title"), d["raw_path"], d["sha256"],
                 d.get("mime"), d.get("bytes", 0), d.get("language"), d.get("collected_at") or now(), "saved", d.get("tokens", 0), d.get("labels_path"),
                 d.get("doc_type"), d.get("publisher"), d.get("licence"), d.get("published_date")))
            if d.get("person_key"):
                self._link(d["person_key"], d["doc_id"], d.get("topic"), d.get("tokens", 0))
            self._db.commit()
            return True

    def _link(self, person_key, doc_id, topic, tokens=0):
        cur = self._db.execute("INSERT OR IGNORE INTO person_docs (person_key, doc_id, topic) VALUES (?,?,?)", (person_key, doc_id, topic))
        if cur.rowcount:
            self._db.execute("UPDATE people SET docs=docs+1, tokens=tokens+?, corruption_docs=corruption_docs+?, updated_at=? WHERE key=?",
                             (tokens, 1 if topic == "corruption" else 0, now(), person_key))

    def link(self, person_key, doc_id, topic):
        """Attach an already collected document to one more person (no download, no new file)."""
        with self._lock:
            row = self._one("SELECT tokens FROM documents WHERE doc_id=?", doc_id)
            self._link(person_key, doc_id, topic, row["tokens"] if row else 0)
            self._db.commit()

    def add_kb_reference(self, person_key, url, title, topic):
        """The knowledge base already holds this page (loaded earlier by another loader): nothing is downloaded or published again, the person is just
        linked to it. The row is marked status 'in_kb' and has no raw file."""
        import hashlib
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO documents (doc_id, person_key, topic, source, url, title, raw_path, sha256, mime, bytes, language, collected_at, status, tokens) "
                             "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("kb_" + h[:20], person_key, topic, "web", url, title, None, "url:" + h, None, 0, None, now(), "in_kb", 0))
            self._link(person_key, "kb_" + h[:20], topic, 0)
            self._db.commit()
        return "kb_" + h[:20]

    def doc_by_url(self, url):
        return self._one("SELECT doc_id, topic FROM documents WHERE url=? LIMIT 1", url)

    def unlinked_in_graph(self, limit=2000):
        """(person_key, doc_id, url) of person-document links not yet written to the graph as (:Person)-[:HAS_DOCUMENT]->(:Source)."""
        return self._all("SELECT pd.person_key AS person_key, pd.doc_id AS doc_id, d.url AS url FROM person_docs pd JOIN documents d ON d.doc_id=pd.doc_id "
                         "WHERE pd.graph_linked=0 LIMIT ?", limit)

    def mark_graph_linked(self, pairs):
        with self._lock:
            self._db.executemany("UPDATE person_docs SET graph_linked=1 WHERE person_key=? AND doc_id=?", [(p, d) for p, d in pairs])
            self._db.commit()

    def mark_published(self, doc_id):
        self._exec("UPDATE documents SET status='published' WHERE doc_id=?", doc_id)
        self.stage_seen(doc_id, "kafka")

    def unpublished(self, limit=500):
        """Files that were saved but whose Kafka message was not confirmed (a crash or Kafka down): published again on the next run."""
        return self._all("SELECT * FROM documents WHERE status='saved' ORDER BY collected_at LIMIT ?", limit)

    def documents_of(self, person_key, limit=200):
        return self._all("SELECT d.doc_id, d.title, pd.topic, d.source, d.url, d.bytes, d.collected_at, d.raw_path, d.status FROM person_docs pd "
                         "JOIN documents d ON d.doc_id=pd.doc_id WHERE pd.person_key=? ORDER BY d.collected_at LIMIT ?", person_key, limit)

    def recent_documents(self, limit=40):
        return self._all("SELECT d.*, p.name AS person FROM documents d LEFT JOIN people p ON p.key=d.person_key ORDER BY d.collected_at DESC, d.rowid DESC LIMIT ?", limit)

    def document_totals(self):
        return self._one("SELECT COUNT(*) AS documents, COALESCE(SUM(bytes),0) AS bytes FROM documents")

    def documents_since(self, iso):
        return self._one("SELECT COUNT(*) AS n FROM documents WHERE collected_at>=?", iso)["n"]

    def last_document(self):
        return self._one("SELECT title, url, collected_at AS at FROM documents ORDER BY collected_at DESC, rowid DESC LIMIT 1")

    # ---- stages seen later in the pipeline ---------------------------------------------------------------
    def stage_seen(self, doc_id, stage, at=None):
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO doc_stages (doc_id, stage, at) VALUES (?,?,?)", (doc_id, stage, at or now()))
            self._db.commit()

    def stages_of(self, doc_ids):
        if not doc_ids:
            return {}
        marks = ",".join("?" * len(doc_ids))
        out = {d: {} for d in doc_ids}
        for r in self._all(f"SELECT doc_id, stage, at FROM doc_stages WHERE doc_id IN ({marks})", *doc_ids):
            out[r["doc_id"]][r["stage"]] = r["at"]
        return out

    def stage_rows(self, limit=120):
        return self._all("SELECT s.doc_id, s.stage, s.at, d.title, p.name AS person FROM doc_stages s JOIN documents d ON d.doc_id=s.doc_id "
                         "LEFT JOIN people p ON p.key=d.person_key ORDER BY s.at DESC LIMIT ?", limit)

    # ---- events and counters -----------------------------------------------------------------------------
    def event(self, stage, detail="", doc_id=None, title=None, person=None):
        self._exec("INSERT INTO events (at, doc_id, title, person, stage, detail) VALUES (?,?,?,?,?,?)", now(), doc_id, title, person, stage, detail)

    def bump(self, name, by=1):
        self._exec("INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=v+?", name, by, by)

    def counter(self, name):
        row = self._one("SELECT v FROM meta WHERE k=?", name)
        return row["v"] if row else 0
