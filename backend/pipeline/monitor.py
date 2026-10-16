"""What the dashboard shows: one status document that follows a file from the collector to Chroma (shape: docs/PIPELINE_CONTRACT.md section 6).

Every stage is read on its own and may fail on its own (a container is stopped, a store is down): that stage then reports `up: false`
and null numbers, and the rest of the document is still returned. Nothing here writes to Kafka, Neo4j or Chroma.

Where each number comes from
    collector  the collector's SQLite registry (registry.py)
    raw        a walk of backend/data/kb_raw (cached a few seconds)
    kafka      the topic's low / high watermarks; lag = high watermark minus the offsets Spark reported in its stats lines
    spark      the container state (`docker inspect`) and backend/data/corpus/_stats/*.jsonl, which Spark appends to after every batch
    corpus     the same stats lines (passages, clean tokens, duplicates) and the Parquet part files on disk
    neo4j      counts through the knowledge base connection
    chroma     the vector store count; embedded percent = Chroma passages / Neo4j passages
The "movement" feed is the registry's record of when each document was first seen in each stage. Later stages are discovered here: a document
counts as in Neo4j when a Source with its doc_id exists, and in Chroma when all its passages are there.
"""
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.pipeline import control
from backend.pipeline.common import RAW_DIR
from backend.pipeline.registry import Registry, now

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
TARGET_TOKENS = 1_000_000_000
SOURCES = ("web", "pdf", "upload", "dataset")


class Cache:
    """Tiny time-based cache so a 2-second dashboard poll does not walk the disk or query the stores every time."""

    def __init__(self):
        self.items = {}

    def get(self, key, ttl, fn):
        hit = self.items.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        value = fn()
        self.items[key] = (time.time(), value)
        return value


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a stage that cannot be read is reported as unavailable, never an error
        return default


def _age_seconds(iso):
    try:
        return max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds())
    except (TypeError, ValueError):
        return None


class Monitor:
    def __init__(self, registry=None, kb=None, rag=None, raw_dir=RAW_DIR, corpus_dir=CORPUS_DIR, control_path=control.CONTROL_PATH):
        self.registry = registry or Registry()
        self._kb, self._rag, self._own = kb, rag, {}
        self.raw_dir, self.corpus_dir, self.control_path = Path(raw_dir), Path(corpus_dir), control_path
        self.cache = Cache()
        self._kafka_up = True

    # ---- connections (the server's own, or opened here on first use) -------------------------------------
    def kb(self):
        if self._kb is not None:
            return self._kb
        if "kb" not in self._own:
            from backend.knowledge.kb import KnowledgeBase
            self._own["kb"] = _safe(KnowledgeBase.from_env)
        return self._own["kb"]

    def rag(self):
        if self._rag is not None:
            return self._rag
        if "rag" not in self._own:
            from backend.knowledge.vectorstore import RagStore
            self._own["rag"] = _safe(RagStore.from_env)
        return self._own["rag"]

    # ---- stages ------------------------------------------------------------------------------------------
    def collector(self):
        totals = self.registry.document_totals()
        five_min_ago = datetime.fromtimestamp(time.time() - 300, timezone.utc).isoformat(timespec="seconds")
        return {"documents": totals["documents"], "bytes": totals["bytes"], "per_minute": round(self.registry.documents_since(five_min_ago) / 5, 2),
                "failed": self.registry.counter("failures"), "retries": self.registry.counter("retries"), "last": self.registry.last_document()}

    def raw(self):
        def walk():
            by_source, files, size = dict.fromkeys(SOURCES, 0), 0, 0
            for path in self.raw_dir.rglob("*"):
                if not path.is_file() or path.name == "checksums.txt" or path.suffix == ".part" or path.relative_to(self.raw_dir).parts[0] == "people":
                    continue                                          # people/ holds links to the same files: counted once, under their source folder
                files += 1
                size += path.stat().st_size
                top = path.relative_to(self.raw_dir).parts[0]
                if top in by_source:
                    by_source[top] += 1
                else:                                             # the older content-addressed layout (kb_raw/ab/<sha>.pdf): by extension
                    ext = path.suffix.lower()
                    by_source["pdf" if ext == ".pdf" else "web" if ext in (".html", ".htm") else "dataset"] += 1
            return files, size, by_source
        files, size, by_source = self.cache.get("raw", 5, walk)
        return {"files": files, "bytes": size, "by_source": by_source, "free_disk_bytes": shutil.disk_usage(self.raw_dir.anchor or str(self.raw_dir)).free}

    def stats_lines(self):
        def read():
            out = []
            for path in sorted((self.corpus_dir / "_stats").glob("????-??-??.jsonl")):
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
            return out
        return self.cache.get("stats", 3, lambda: _safe(read, []))

    def kafka(self, stats):
        def read():
            from confluent_kafka import Consumer, TopicPartition
            from backend.pipeline.kafka_io import BOOTSTRAP, CLIENT_OPTIONS, TOPIC
            c = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": "elvarn-monitor", "enable.auto.commit": False, **CLIENT_OPTIONS})      # IPv4 only, quiet while Kafka is stopped
            try:
                meta = c.list_topics(TOPIC, timeout=3)
                marks = {p: c.get_watermark_offsets(TopicPartition(TOPIC, p), timeout=3) for p in meta.topics[TOPIC].partitions}
            finally:
                c.close()
            return TOPIC, marks
        got = self.cache.get("kafka", 3 if self._kafka_up else 20, lambda: _safe(read))                 # a stopped Kafka is asked about less often
        self._kafka_up = got is not None
        if not got:
            return {"up": False, "topic": "kb.raw_documents", "messages": None, "consumer_lag": None}
        topic, marks = got
        done = {}
        for line in stats:
            for part, off in (line.get("offsets") or {}).items():
                done[int(part)] = max(done.get(int(part), 0), int(off))
        lag = sum(max(0, high - done[p]) for p, (low, high) in marks.items()) if done else None
        return {"up": True, "topic": topic, "messages": sum(high - low for low, high in marks.values()), "consumer_lag": lag}

    def spark(self, stats):
        def running():
            out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", "elvarn-spark"], capture_output=True, text=True, timeout=8)
            return out.returncode == 0 and out.stdout.strip() == "true"
        up = self.cache.get("spark_up", 10, lambda: _safe(running, False))
        total = lambda key: sum(int(line.get(key) or 0) for line in stats)     # noqa: E731
        return {"up": bool(up), "processed_documents": total("processed"), "skipped": total("skipped"), "failed": total("failed"),
                "dead_letters": total("dead_letters")}

    def corpus(self, stats):
        parts = self.cache.get("parts", 10, lambda: sum(1 for _ in (self.corpus_dir / "passages").rglob("*.parquet")) if (self.corpus_dir / "passages").is_dir() else 0)
        total = lambda key: sum(int(line.get(key) or 0) for line in stats)     # noqa: E731
        return {"passages": total("passages"), "clean_tokens": total("clean_tokens"), "target_tokens": TARGET_TOKENS,
                "duplicates_removed": total("duplicates_removed"), "files": parts}

    def neo4j(self):
        def read():
            kb = self.kb()
            if kb is None:
                raise RuntimeError("no graph connection")
            one = lambda q: kb._read(q)[0]["n"]                                  # noqa: E731
            return {"up": True, "documents": one("MATCH (s:Source) RETURN count(s) AS n"), "passages": one("MATCH (p:Passage) RETURN count(p) AS n"),
                    "entities": one("MATCH (e:Entity) RETURN count(e) AS n"), "relationships": one("MATCH ()-[r:RELATION]->() RETURN count(r) AS n"),
                    "people": one("MATCH (e:Entity {type:'Person'}) RETURN count(e) AS n")}
        return self.cache.get("neo4j", 5, lambda: _safe(read, {"up": False, "documents": None, "passages": None, "entities": None, "relationships": None, "people": None}))

    def chroma(self, neo):
        def read():
            rag = self.rag()
            if rag is None:
                raise RuntimeError("no vector store")
            return rag.count()
        n = self.cache.get("chroma", 5, lambda: _safe(read))
        if n is None:
            return {"up": False, "passages": None, "embedded_pct": None}
        pct = round(min(100.0, 100.0 * n / neo["passages"]), 1) if neo.get("passages") else None
        return {"up": True, "passages": n, "embedded_pct": pct}

    # ---- following documents to the later stages ---------------------------------------------------------
    def observe(self, stats):
        """Record when recent documents were first seen in Spark, the corpus, Neo4j and Chroma. Returns nothing; reads are best effort."""
        recent = self.registry.recent_documents(30)
        ids = [d["doc_id"] for d in recent]
        seen = self.registry.stages_of(ids)
        in_stats = {d for line in stats for d in (line.get("doc_ids") or [])}
        pending = [d for d in ids if not {"neo4j", "chroma"} <= set(seen[d])]
        for d in ids:
            if d in in_stats:
                self.registry.stage_seen(d, "spark")
                self.registry.stage_seen(d, "corpus")
        kb = self.kb() if pending else None
        if kb is None:
            return
        found = _safe(lambda: {r["d"]: r["pids"] for r in kb._read(
            "MATCH (s:Source) WHERE s.doc_id IN $ids OPTIONAL MATCH (p:Passage)-[:FROM]->(s) RETURN s.doc_id AS d, collect(p.id) AS pids", ids=pending)}, {})
        for d, pids in found.items():
            self.registry.stage_seen(d, "spark")
            self.registry.stage_seen(d, "neo4j")
            rag = self.rag()
            if rag is not None and pids and not _safe(lambda: rag.missing(pids), pids):
                self.registry.stage_seen(d, "chroma")

    def movement(self):
        return [{"at": r["at"], "doc_id": r["doc_id"], "title": r["title"], "person": r["person"], "stage": r["stage"], "detail": ""}
                for r in self.registry.stage_rows(50)]

    # ---- people ------------------------------------------------------------------------------------------
    def people(self):
        counts = self.registry.counts()
        current = None
        running = control.is_running(self.control_path)
        cur = control.read(self.control_path).get("current") if running else None
        if cur:                                                  # what the collector itself says it is doing (also for a one-person request)
            p = self.registry.person(cur["key"]) or {}
            current = {"key": cur["key"], "name": cur.get("name"), "party": cur.get("party"), "stage": cur.get("stage"), "docs": p.get("docs", 0),
                       "since": cur.get("since"), "done": p.get("sources_done", []), "wanted": cur.get("wanted")}
        row = self.registry._one("SELECT key, name, party, stage, docs, started_at FROM people WHERE status='in_progress' ORDER BY updated_at DESC LIMIT 1")
        if not current and row and running:
            current = {"key": row["key"], "name": row["name"], "party": row["party"], "stage": row["stage"] or "finishing", "docs": row["docs"],
                       "since": row["started_at"]}
        return {**counts, "current": current, "recent": self.registry.recent_people(8), "queue": self.registry.queue(8)}

    # ---- the whole document ------------------------------------------------------------------------------
    def status(self):
        ctl = control.state(self.control_path)
        stats = self.stats_lines()
        neo = self.neo4j()
        self.observe(stats)
        elapsed = _age_seconds(ctl["started_at"]) if ctl["state"] in ("running", "stopping") else 0
        return {
            "running": ctl["state"] in ("running", "stopping"), "state": ctl["state"], "started_at": ctl["started_at"], "elapsed_s": int(elapsed or 0),
            "scope": {"from_year": 1947, "to_year": 2026}, "message": ctl.get("message", ""), "people": self.people(),
            "stages": {"collector": self.collector(), "raw": _safe(self.raw), "kafka": self.kafka(stats), "spark": self.spark(stats),
                       "corpus": self.corpus(stats), "neo4j": neo, "chroma": self.chroma(neo)},
            "movement": self.movement(), "at": now()}

    def people_page(self, status=None, q=None, limit=50, offset=0):
        total, rows = self.registry.people(status or None, q or None, max(1, min(int(limit), 200)), max(0, int(offset)))
        return {"total": total, "items": [{
            "key": p["key"], "name": p["name"], "party": p["party"], "first_year": p["first_year"], "last_year": p["last_year"], "positions": p["positions"],
            "status": p["status"], "docs": p["docs"], "tokens": p["tokens"], "corruption_docs": p["corruption_docs"], "updated_at": p["updated_at"]} for p in rows]}

    def person_detail(self, key):
        p = self.registry.person(key)
        if p is None:
            return None
        item = {k: p[k] for k in ("key", "name", "party", "first_year", "last_year", "positions", "status", "docs", "tokens", "corruption_docs", "updated_at")}
        return {"person": item, "documents": self.registry.documents_of(key)}
