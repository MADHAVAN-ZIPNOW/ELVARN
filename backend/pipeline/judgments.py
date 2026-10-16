"""All-India court judgments, in bulk (not person by person): every Supreme Court judgment and every High Court judgment in the open datasets.

Sources (public AWS buckets, CC-BY-4.0, built by Dattam Labs from the eCourts / Supreme Court records; see collectors/court_opendata.py):
    indian-supreme-court-judgments   1950-    one metadata table per year
    indian-high-court-judgments      all High Courts, one metadata table per year / court / bench

What is kept for each judgment (the case name is the title of the document everywhere: Kafka, Neo4j, Chroma, the folder and the file name):
    case_name     "State of Tamil Nadu vs J. Jayalalithaa"       (High Court titles "CR. MISC./5208/2019 of A Vs B" are split into name, type, number)
    case_type, case_number, court, bench, year, decision date, judges, outcome, case reference (CNR)
    subjects      election, MLA/MP, criminal, constitutional, tax ... (words found in the first pages)
    statutes      the Acts the first pages cite ("Representation of the People Act, 1951")

Order of work: the Supreme Court first (all years), then the High Courts in sweeps. A sweep takes the newest N judgments of every
court / year / bench (N = 300, then 3,000, then 30,000, then all), so a very large court cannot use the whole disk before the others are
read. A disk budget (default 40 GB of raw PDFs) stops the run; Start again with a bigger budget and it carries on. A judgment already
stored (same URL or same bytes) or already rejected (no readable text, off topic) is never fetched again.

    python -m backend.pipeline.judgments run [--max-gb 40] [--scope all|sc|hc] [--from-year 1950] [--to-year 2026] [--topics election,criminal]
    python -m backend.pipeline.judgments status

Files: kb_raw/pdf/{date}/{doc_id}.pdf (the archive, as for every collector) and, hard-linked, kb_raw/courts/{court}/{year}/{case-name}__{doc_id}.pdf.
"""
import argparse
import os
import re
import signal
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from backend.pipeline import control
from backend.pipeline.law_tags import SUBJECT_NAMES, case_of_hc, case_of_sc, clean, full_title, slug, statutes_of, subjects_of
from backend.pipeline.collectors import court_opendata as co
from backend.pipeline.person_sources import RawArchive
from backend.pipeline.registry import STATE_DIR, Registry, now

PLAN_PATH = STATE_DIR / "judgments.sqlite"
MODULE = "backend.pipeline.judgments"
FIRST_YEAR = 1950
DEFAULT_GB = 40
QUOTAS = (300, 3000, 30000, None)          # High Court judgments read per court/year/bench in sweep 0, 1, 2, 3 (None = all)
BATCH = 48                                 # judgments downloaded together, then stored and announced together
WORKERS = 24                               # judgments downloaded at once: one connection fetches only about 2 a second
TAG_PAGES = 3                              # pages read here (scanned-or-not check, topic filter); Spark reads the whole judgment
MIN_TEXT = 500                             # fewer characters in those pages: a scanned judgment, nothing to learn from it
SCOPES = ("all", "sc", "hc")

SCHEMA = """
CREATE TABLE IF NOT EXISTS plan (unit TEXT PRIMARY KEY, kind TEXT, year INTEGER, code TEXT, bench TEXT, court TEXT, sweep INTEGER NOT NULL DEFAULT -1);
CREATE TABLE IF NOT EXISTS planned (year_key TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS seen (url TEXT PRIMARY KEY, why TEXT);
"""


def read_pdf(data, pages=TAG_PAGES):
    """(text of the first `pages` pages, number of pages)."""
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(data))
    total = len(reader.pages)
    return "\n".join((reader.pages[i].extract_text() or "") for i in range(min(pages, total))), total




# ---- the collector ------------------------------------------------------------------------------------------------
class Stopped(Exception):
    """The owner pressed Stop (or the disk budget is used up)."""


class JudgmentCollector:
    def __init__(self, registry, archive, sink, s3, plan_path=PLAN_PATH, should_stop=lambda: False, say=print, max_bytes=DEFAULT_GB * 1024 ** 3,
                 scope="all", from_year=FIRST_YEAR, to_year=2026, topics=None, workers=WORKERS, current_year=2026, on_unit=None, quotas=QUOTAS):
        self.registry, self.archive, self.sink, self.s3 = registry, archive, sink, s3
        self.should_stop, self.say, self.max_bytes, self.workers = should_stop, say, max_bytes, workers
        self.scope, self.from_year, self.to_year, self.current_year = scope, from_year, min(to_year, current_year), current_year
        self.topics = [t for t in (topics or []) if t in SUBJECT_NAMES]
        self.on_unit, self.quotas = on_unit or (lambda text, unit: None), quotas
        Path(plan_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(plan_path), check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.known = {r["url"] for r in registry._all("SELECT url FROM documents WHERE topic='judgment' AND url IS NOT NULL")}
        self.known |= {r["url"] for r in self.db.execute("SELECT url FROM seen")}
        self.used = registry._one("SELECT COALESCE(SUM(bytes),0) AS b FROM documents WHERE topic='judgment'")["b"]
        self.counts = {"stored": 0, "no_text": 0, "off_topic": 0, "missing": 0, "duplicate": 0}

    # -- the plan: which court / year / bench units exist ------------------------------------------------------
    def plan(self):
        want_sc, want_hc = self.scope in ("all", "sc"), self.scope in ("all", "hc")
        for year in range(self.to_year, max(self.from_year, FIRST_YEAR) - 1, -1):
            if want_sc:
                self._add(f"sc:{year}", "sc", year, None, None, "Supreme Court of India")
            key = f"hc:{year}"
            if not want_hc or (self.db.execute("SELECT 1 FROM planned WHERE year_key=?", (key,)).fetchone() and year < self.current_year):
                continue
            if self.should_stop():
                raise Stopped()
            try:
                courts = [c for c in self.s3.children(co.HC_BUCKET, f"metadata/parquet/year={year}/") if c.startswith("court=")]
                for court in courts:
                    for bench in self.s3.children(co.HC_BUCKET, f"metadata/parquet/year={year}/{court}/"):
                        code = court.split("=", 1)[1]
                        self._add(f"hc:{year}:{code}:{bench.split('=', 1)[-1]}", "hc", year, code, bench.split("=", 1)[-1], None)
                self.on_unit(f"Listing the High Courts of {year}", None)
            except Exception as exc:  # noqa: BLE001 - a year without metadata is normal
                self.say(f"plan: {year} has no High Court metadata ({str(exc)[:60]})")
            self.db.execute("INSERT OR REPLACE INTO planned (year_key) VALUES (?)", (key,))
            self.db.commit()

    def _add(self, unit, kind, year, code, bench, court):
        self.db.execute("INSERT OR IGNORE INTO plan (unit, kind, year, code, bench, court) VALUES (?,?,?,?,?,?)", (unit, kind, year, code, bench, court))
        self.db.commit()

    def units(self, sweep):
        rows = self.db.execute("SELECT * FROM plan WHERE sweep < ? AND year BETWEEN ? AND ? ORDER BY CASE kind WHEN 'sc' THEN 0 ELSE 1 END, year DESC, unit",
                               (sweep, self.from_year, self.to_year)).fetchall()
        return [dict(r) for r in rows if (self.scope == "all" or r["kind"] == self.scope)]

    # -- reading one unit's metadata ----------------------------------------------------------------------------
    def rows_of(self, unit):
        if unit["kind"] == "sc":
            path = f"{co.SC_BUCKET}/metadata/parquet/year={unit['year']}/metadata.parquet"
            wanted = ["title", "petitioner", "respondent", "decision_date", "path", "case_id", "judge", "disposal_nature", "court"]
        else:
            path = f"{co.HC_BUCKET}/metadata/parquet/year={unit['year']}/court={unit['code']}/bench={unit['bench']}/metadata.parquet"
            wanted = ["title", "pdf_link", "decision_date", "disposal_nature", "judge", "cnr", "court"]
        table = self.s3.metadata(path, wanted)
        out = []
        for r in table:
            if unit["kind"] == "sc":
                if not r.get("path"):
                    continue
                key = f"data/pdf/year={unit['year']}/english/{r['path']}_EN.pdf"
                case, bucket, court = case_of_sc(r), co.SC_BUCKET, "Supreme Court of India"
                ref = r.get("case_id")
            else:
                if not r.get("pdf_link"):
                    continue
                key = f"data/pdf/year={unit['year']}/court={unit['code']}/bench={unit['bench']}/{r['pdf_link'].rsplit('/', 1)[-1]}"
                case, bucket, court = case_of_hc(r.get("title")), co.HC_BUCKET, clean(r.get("court")) or f"High Court ({unit['code']})"
                ref = r.get("cnr")
            out.append({"bucket": bucket, "key": key, "url": f"https://{bucket}.s3.amazonaws.com/{key}", "case": case, "court": court, "date": co.iso_date(r.get("decision_date")),
                        "judge": clean(r.get("judge")) or None, "outcome": clean(r.get("disposal_nature")) or None, "ref": str(ref) if ref else None,
                        "bench": unit.get("bench"), "year": unit["year"]})
        out.sort(key=lambda r: r["date"] or "", reverse=True)
        return out

    # -- one judgment -------------------------------------------------------------------------------------------
    def fetch(self, row):
        try:
            data = self.s3.read(row["bucket"], row["key"])
        except Exception:  # noqa: BLE001 - listed but not in the bucket
            return row, None, "", 0
        try:
            text, pages = read_pdf(data)
        except Exception:  # noqa: BLE001 - a damaged PDF
            text, pages = "", 0
        return row, data, text, pages

    def reject(self, url, why):
        self.known.add(url)
        self.db.execute("INSERT OR REPLACE INTO seen (url, why) VALUES (?,?)", (url, why))

    def store(self, row, data, text, pages):
        """Archive, register and announce one judgment. -> doc_id or None."""
        subjects, statutes = subjects_of(text), statutes_of(text)
        if self.topics and not (set(self.topics) & set(subjects)):
            self.counts["off_topic"] += 1
            return self.reject(row["url"], "off_topic")
        if self.used + len(data) > self.max_bytes:
            raise Stopped()
        doc_id, sha, raw_path, size, _ = self.archive.save(data, "pdf", "pdf")
        title = full_title(row["case"], row["court"], row["date"])
        sc = row["bucket"] == co.SC_BUCKET
        added = self.registry.add_document({"doc_id": doc_id, "topic": "judgment", "source": "pdf", "url": row["url"], "title": title, "raw_path": raw_path, "sha256": sha,
                                            "mime": "application/pdf", "bytes": size, "language": "en", "doc_type": "judgment", "publisher": row["court"],
                                            "licence": co.LICENCE, "published_date": row["date"]})
        if not added:
            self.counts["duplicate"] += 1
            return self.reject(row["url"], "duplicate")
        self.used += size
        self.known.add(row["url"])
        self.registry.stage_seen(doc_id, "collected")
        self.registry.stage_seen(doc_id, "raw")
        self.registry.event("collected", f"judgment | {row['court']} | {size} bytes", doc_id, title, None)
        self.link_folder(row, doc_id, raw_path)
        self.sink.send(self.message(row, title, doc_id, sha, raw_path, size, subjects, statutes, pages, sc))
        self.counts["stored"] += 1
        return doc_id

    def message(self, row, title, doc_id, sha, raw_path, size, subjects, statutes, pages, sc):
        """The Kafka event (docs/PIPELINE_CONTRACT.md section 3) plus the case fields, which Spark carries into Neo4j."""
        c = row["case"]
        return {"event_type": "document_ingested", "doc_id": doc_id, "source": "pdf", "source_url": row["url"], "raw_path": raw_path, "mime_type": "application/pdf",
                "file_size_bytes": size, "sha256": sha, "language": "en", "collected_at": now(), "status": "NEW", "url": row["url"], "title": title,
                "doc_type": "judgment", "publisher": row["court"], "licence": co.LICENCE, "published_date": row["date"], "site": "law", "topic": "judgment",
                "case_name": c["case_name"], "case_type": c.get("case_type"), "case_number": c.get("case_number"), "court": row["court"], "bench": row["bench"],
                "year": row["year"], "judge": row["judge"], "outcome": row["outcome"], "ref": row["ref"], "subjects": subjects, "statutes": statutes, "pages": pages}

    def link_folder(self, row, doc_id, raw_path):
        """kb_raw/courts/<court>/<year>/<case-name>__<doc id>.pdf, a hard link to the archived file (no second copy on disk)."""
        try:
            root = self.archive.root
            src = root / raw_path.removeprefix("kb_raw/")
            dst = root / "courts" / slug(row["court"], 40) / str(row["year"]) / f"{slug(row['case']['case_name'])}__{doc_id}.pdf"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.link(src, dst)
        except OSError:
            pass                                                   # a file system without hard links: the archive copy is the record

    # -- one unit, one sweep ------------------------------------------------------------------------------------
    def do_unit(self, unit, quota):
        rows = self.rows_of(unit)
        if quota:
            rows = rows[:quota]
        todo = [r for r in rows if r["url"] not in self.known]
        label = f"{rows[0]['court'] if rows else (unit['court'] or unit['unit'])} {unit['year']}"
        if not todo:
            return
        with ThreadPoolExecutor(self.workers) as pool:
            for i in range(0, len(todo), BATCH):
                if self.should_stop():
                    raise Stopped()
                self.on_unit(f"{label}: {i} of {len(todo)} new", unit)
                sent = []
                for row, data, text, pages in pool.map(self.fetch, todo[i:i + BATCH]):
                    if data is None:
                        self.counts["missing"] += 1
                        self.reject(row["url"], "missing")
                    elif len(text.strip()) < MIN_TEXT:
                        self.counts["no_text"] += 1
                        self.reject(row["url"], "no_text")
                    else:
                        doc = self.store(row, data, text, pages)
                        if doc:
                            sent.append(doc)
                self.sink.flush()
                for d in sent:
                    self.registry.mark_published(d)
                self.db.commit()

    def run(self):
        """-> 'done', 'stopped' or 'full' (the disk budget is used up)."""
        try:
            self.on_unit("Listing courts and years (the first run takes a few minutes)", None)
            self.plan()
            for sweep, quota in enumerate(self.quotas):
                for unit in self.units(sweep):
                    if self.should_stop():
                        raise Stopped()
                    try:
                        self.do_unit(unit, None if unit["kind"] == "sc" else quota)
                    except Stopped:
                        raise
                    except Exception as exc:  # noqa: BLE001 - one unreadable unit must not stop the rest
                        self.say(f"unit {unit['unit']} skipped: {exc.__class__.__name__}: {str(exc)[:80]}")
                    self.db.execute("UPDATE plan SET sweep=? WHERE unit=?", (sweep, unit["unit"]))
                    self.db.commit()
                self.say(f"sweep {sweep} finished: {self.counts}")
            return "done"
        except Stopped:
            self.db.commit()
            return "full" if self.used >= self.max_bytes * 0.98 else "stopped"


# ---- the process the Start button launches ------------------------------------------------------------------------
def should_chain(outcome, then):
    """Run plan "judgments, then politicians": the second step starts by itself when the first one finished (or used its whole disk budget).
    A Stop pressed by the owner ends the whole plan."""
    return then == "people" and outcome in ("done", "full")


def run_process(max_gb=DEFAULT_GB, scope="all", from_year=FIRST_YEAR, to_year=2026, topics=None, then=None):
    try:
        from dotenv import load_dotenv
        load_dotenv(control.REPO / ".env", override=False)
    except ImportError:
        pass
    from backend.pipeline.person_collector import StoreSync
    registry = Registry()
    control.write({"state": "running", "pid": os.getpid(), "started_at": now(), "stop_requested": False, "mode": "judgments", "budget_bytes": int(max_gb * 1024 ** 3), "then": then if then == "people" else None,
                   "message": ("Step 1 of 2: " if then == "people" else "") + "Collecting court judgments from all of India."})
    stop = {"flag": False}

    def request(*_):
        stop["flag"] = True
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request)
    outcome, message = "error", ""
    began = now()
    try:
        from backend.pipeline.kafka_io import KafkaSink
        s3 = co.S3Source()
        sync = StoreSync(say=lambda m: print(m, flush=True), registry=registry)
        sync.start()
        before = registry._one("SELECT COUNT(*) AS n FROM documents WHERE topic='judgment'")["n"]

        def on_unit(text, unit):
            control.write({"message": text, "current": {"kind": "judgments", "key": "judgments", "name": text, "stage": "judgments", "since": began, "wanted": None}})
        collector = JudgmentCollector(registry, RawArchive(), KafkaSink(), s3, should_stop=lambda: stop["flag"] or control.stop_requested(),
                                      say=lambda m: print(m, flush=True), max_bytes=int(max_gb * 1024 ** 3), scope=scope, from_year=from_year, to_year=to_year,
                                      topics=topics, on_unit=on_unit)
        try:
            outcome = collector.run()
        finally:
            sync.done.set()
        control.write({"message": "Syncing Chroma with what Spark has processed ..."})
        time.sleep(45)
        sync.once()
        n = registry._one("SELECT COUNT(*) AS n FROM documents WHERE topic='judgment'")["n"] - before
        message = {"done": "All the judgments in the chosen courts and years are collected.", "stopped": "Stopped. Start again to carry on.",
                   "full": f"The {max_gb:g} GB disk budget is used up. Start again with a bigger budget to carry on."}.get(outcome, "") + f" ({n} new judgments in this run.)"
    except Exception as exc:  # noqa: BLE001
        message = f"The judgments collector stopped with an error: {exc}"
        print(message, flush=True)
    if should_chain(outcome, then):
        control.write({"message": message + " Step 2 of 2: the politician-by-politician collector starts now.", "current": None, "then": None})
        from backend.pipeline import person_collector
        return person_collector.run_process()                              # the same Start/Stop control: it writes its own state, pid and message
    control.write({"state": "idle" if outcome in ("done", "stopped", "full") else "error", "pid": None, "stop_requested": False, "message": message, "current": None, "then": None})
    return outcome


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "status"])
    ap.add_argument("--max-gb", type=float, default=DEFAULT_GB)
    ap.add_argument("--scope", choices=SCOPES, default="all")
    ap.add_argument("--from-year", type=int, default=FIRST_YEAR)
    ap.add_argument("--to-year", type=int, default=2026)
    ap.add_argument("--then", choices=["people"], default=None, help="when this run finishes, start the politician-by-politician collector")
    ap.add_argument("--topics", default="", help="only judgments about these subjects: " + ", ".join(SUBJECT_NAMES))
    a = ap.parse_args(argv)
    if a.command == "status":
        r = Registry()
        print(r._one("SELECT COUNT(*) AS judgments, COALESCE(SUM(bytes),0) AS bytes FROM documents WHERE topic='judgment'"))
        return
    topics = [t for t in a.topics.split(",") if t]
    sys.exit(0 if run_process(a.max_gb, a.scope, a.from_year, a.to_year, topics, a.then) in ("done", "stopped", "full") else 1)


if __name__ == "__main__":
    main()
