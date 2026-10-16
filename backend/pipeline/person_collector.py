"""The politician-by-politician collector: take the next person on the roster, collect everything about them, then move on.

For each person, in this order (a finished source is skipped when a stopped run resumes):

    wikidata -> wikipedia_en -> wikipedia_ta -> corruption_articles -> court_records -> ed_records -> dvac_records -> cbi_records -> news_records

Every download is archived untouched under kb_raw/{source}/{date}/{doc_id}.{ext}, registered (sha256, size, url ...) in the collector's
SQLite file, and announced with ONE Kafka message that points at the file (never the bytes). Spark takes it from there.

    python -m backend.pipeline.person_collector run [--limit N] [--person wd:Q123]    # until the roster is done or you stop it
    python -m backend.pipeline.person_collector roster                                # (re)build the roster from Wikidata
    python -m backend.pipeline.person_collector status

The app's Start / Stop button starts and stops this process (backend/pipeline/control.py). Stop is graceful and resumable.
"""
import argparse
import json
import os
import signal
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

from backend.pipeline import collect_request, control, official_index, person_folder, person_sources, roster
from backend.pipeline.common import FetchError, Fetcher
from backend.pipeline.person_sources import MAX_CORRUPTION_ARTICLES, RawArchive, Wikimedia
from backend.pipeline.registry import Registry, now

STAGES = collect_request.ALL_STAGES
OFFICIAL_KINDS = {"court_records": ("sc", "hc"), "ed_records": ("ed",), "dvac_records": ("dvac",), "cbi_records": ("cbi",)}
MAX_OFFICIAL_RECORDS = 40
WORKERS = 4                         # concurrent downloads (the contract allows 4 to 8)
STAGE_TRIES = 3
FAILURES_IN_A_ROW = 5               # then the network (or Wikimedia) is down: stop instead of failing the whole roster


class Stopped(Exception):
    """The owner pressed Stop (or the process was asked to end)."""


def aliases_of(person):
    """The ways documents write this person's name: the Wikidata name, the article title and (when long enough to be distinctive) the family name."""
    names = [person.get("en_title"), person.get("name")]
    words = (person.get("name") or "").replace(".", " ").split()
    if words and len(words[-1]) >= 7:
        names.append(words[-1])
    out = []
    for n in names:
        if n and n not in out:
            out.append(n)
    return out


def build_message(found, person, doc_id, sha, raw_path, size, labels_path=None):
    """The Kafka event for one archived file (schema in docs/PIPELINE_CONTRACT.md section 3)."""
    return {"event_type": "document_ingested", "doc_id": doc_id, "source": found.source, "source_url": found.url, "raw_path": raw_path,
            "mime_type": found.mime, "file_size_bytes": size, "sha256": sha, "language": found.language, "collected_at": now(), "status": "NEW",
            "url": found.url, "title": found.title, "doc_type": found.doc_type, "publisher": found.publisher, "licence": found.licence,
            "published_date": found.extra.get("published_date"), "site": "politician", "person_key": person["key"], "person_name": person["name"], "topic": found.topic,
            "wikidata_id": found.wikidata_id, "aliases": aliases_of(person), **{k: found.extra[k] for k in ("text", "text_reliable", "judge", "outcome", "ref", "bench", "case_stage", "petitions") if k in found.extra},
            **({"labels_path": labels_path} if labels_path else {})}


class PersonCollector:
    def __init__(self, registry, wikimedia, archive, sink, should_stop=lambda: False, workers=WORKERS, say=print, sleep=time.sleep, index=None, s3=None, agency_fetcher=None, known_urls=None, folders=None, news_fetcher=None, article_fetcher=None, on_stage=None):
        self.registry, self.wm, self.archive, self.sink = registry, wikimedia, archive, sink
        self.index, self.s3 = index, s3                          # the official-records index (None: that stage finds nothing)
        self.agency_fetcher = agency_fetcher or Fetcher(delay=3.0)       # DVAC / CBI: robots.txt respected, a pause between requests
        self.on_stage = on_stage or (lambda person, stage, wanted: None)      # called as each source starts and when the person ends (stage None)
        self.article_fetcher = article_fetcher                        # news sites: robots.txt respected, a pause between pages, small pages only (None: headlines only)
        self.news_fetcher = news_fetcher                              # GDELT: one request every 5 seconds, robots.txt respected (None: no news stage)
        self.known_urls = known_urls or (lambda: set())               # URLs the knowledge base already holds (any loader): never downloaded or stored again
        self.kb_urls = set()
        self.folders = folders or person_folder.PersonFolders(registry, archive.root)           # one folder per person (hard links to the archive)
        self.should_stop, self.workers, self.say, self.sleep = should_stop, workers, say, sleep

    def check_stop(self):
        if self.should_stop():
            raise Stopped()

    # ---- one file -----------------------------------------------------------------------------------------
    def store(self, found, person):
        """Archive, register and announce one download. Returns the doc_id, or None when the same bytes were already collected."""
        doc_id, sha, raw_path, size, _ = self.archive.save(found.data, found.source, found.ext)
        labels_path = self.archive.save(found.companion, "dataset", "json")[2] if found.companion else None
        added = self.registry.add_document({"labels_path": labels_path,
            "doc_id": doc_id, "person_key": person["key"], "topic": found.topic, "source": found.source, "url": found.url, "title": found.title,
            "raw_path": raw_path, "sha256": sha, "mime": found.mime, "bytes": size, "language": found.language, "tokens": found.tokens,
            "doc_type": found.doc_type, "publisher": found.publisher, "licence": found.licence, "published_date": found.extra.get("published_date")})
        if not added:
            row = self.registry._one("SELECT doc_id, raw_path FROM documents WHERE sha256=?", sha)
            if row:
                self.registry.link(person["key"], row["doc_id"], found.topic)
                self.folders.add(person, row, found.topic)
            return None
        self.folders.add(person, {"doc_id": doc_id, "raw_path": raw_path}, found.topic)
        self.registry.stage_seen(doc_id, "collected")
        self.registry.stage_seen(doc_id, "raw")
        self.registry.event("collected", f"{found.topic} | {found.source} | {size} bytes", doc_id, found.title, person["name"])
        self.sink.send(build_message(found, person, doc_id, sha, raw_path, size, labels_path))
        return doc_id

    def flush(self, doc_ids):
        """Wait until Kafka has acknowledged the messages, then mark the files as published (so a crash republishes the rest)."""
        if not doc_ids:
            return
        self.sink.flush()
        for d in doc_ids:
            self.registry.mark_published(d)

    def publish_pending(self):
        """Messages of files that were saved but never confirmed (a crash, or Kafka was down): send them again."""
        rows = self.registry.unpublished()
        for r in rows:
            person = self.registry.person(r["person_key"]) or {"key": r["person_key"], "name": ""}
            wikidata = r["topic"] == "structured"                            # rows written before the registry kept these fields fall back to what the topic implies
            f = person_sources.Found(r["topic"], r["source"], r["raw_path"].rsplit(".", 1)[-1], r["mime"], r["url"], r["title"], r["language"],
                                     r["publisher"] or ("Wikidata" if wikidata else "Wikipedia"), r["licence"] or ("CC0 1.0" if wikidata else "CC BY-SA 4.0"),
                                     r["doc_type"] or ("news" if r["topic"] == "news" else "reference"), b"", wikidata_id=person.get("wikidata_id"),
                                     extra={"published_date": r["published_date"]} if r["published_date"] else {})
            self.sink.send(build_message(f, person, r["doc_id"], r["sha256"], r["raw_path"], r["bytes"], r.get("labels_path")))
        self.flush([r["doc_id"] for r in rows])
        if rows:
            self.say(f"re-sent {len(rows)} unconfirmed messages")

    # ---- one source of one person -------------------------------------------------------------------------
    def collect_source(self, source, person):
        if source == "wikidata":
            return person_sources.wikidata_doc(self.wm, person)
        if source in ("wikipedia_en", "wikipedia_ta"):
            lang = source[-2:]
            title = person.get("en_title" if lang == "en" else "ta_title")
            if title and self.in_kb(person, person_sources.wikipedia_url(lang, title), title + " (Wikipedia)", "person"):
                return []                                                 # already in the knowledge base: linked, not downloaded again
            return person_sources.wikipedia_doc(self.wm, person, lang)
        if source in OFFICIAL_KINDS:
            return self.official_records(person, OFFICIAL_KINDS[source])
        if source == "news_records":
            return self.news(person)
        return self.corruption_articles(person)

    def news(self, person):
        """Headlines about the person's cases (GDELT): only the ones not stored yet, in this collector or in the knowledge base."""
        if self.news_fetcher is None:
            return []
        reader = (lambda url: self.article_fetcher.get(url)) if self.article_fetcher is not None else None
        found = person_sources.news_docs(lambda url: self.news_fetcher.get(url)[0], person, fetch_article=reader)
        fresh = []
        for f in found:
            known = self.registry.doc_by_url(f.url)
            if known:
                self.registry.link(person["key"], known["doc_id"], "news")
                self.folders.add(person, self.known_file(known["doc_id"]), "news")
            elif not self.in_kb(person, f.url, f.title, "news"):
                fresh.append(f)
        return fresh

    def official_records(self, person, kinds=None):
        """Court judgments and agency releases that name the person: candidates from the index, kept only after their text is checked."""
        if self.index is None:
            return []
        rows, found, fetch = [r for r in self.index.match(person) if kinds is None or r["kind"] in kinds], [], []
        for row in rows:
            known = self.registry.doc_by_url(row["url"])
            topic = "court_record" if row["kind"] in ("sc", "hc") else "agency_record"
            if known:
                self.registry.link(person["key"], known["doc_id"], topic)
                self.folders.add(person, self.known_file(known["doc_id"]), topic)
            elif not self.in_kb(person, row["url"], row["title"], topic):
                fetch.append(row)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for result in pool.map(lambda r: self._official(person, r), fetch[:MAX_OFFICIAL_RECORDS]):
                if result is not None:
                    found.append(result)
        return found

    def _official(self, person, row):
        if self.should_stop():
            return None
        try:
            return person_sources.official_doc(row, person, self.s3, self.agency_fetcher)
        except Exception as exc:  # noqa: BLE001 - one unreadable record never stops the person
            self.registry.bump("retries")
            self.say(f"  skipped {row['title'][:50]}: {str(exc)[:80]}")
            return None

    def in_kb(self, person, url, title, topic):
        """True (after linking the person to it) when the knowledge base already holds this page, loaded earlier by any loader."""
        if url in self.kb_urls:
            self.registry.add_kb_reference(person["key"], url, title, topic)
            return True
        return False

    def known_file(self, doc_id):
        return self.registry._one("SELECT doc_id, raw_path FROM documents WHERE doc_id=?", doc_id) or {"doc_id": doc_id, "raw_path": ""}

    def corruption_articles(self, person):
        titles = person_sources.corruption_candidates(self.wm, person)
        found, fetch = [], []
        for title in titles:
            url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
            known = self.registry.doc_by_url(url)
            if known:
                self.registry.link(person["key"], known["doc_id"], "corruption")      # already downloaded for someone else: only link it
                self.folders.add(person, self.known_file(known["doc_id"]), "corruption")
            elif not self.in_kb(person, url, title + " (Wikipedia)", "corruption"):
                fetch.append(title)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for result in pool.map(lambda t: self._candidate(person, t), fetch):
                if result is not None and len(found) < MAX_CORRUPTION_ARTICLES:
                    found.append(result)
        return found

    def _candidate(self, person, title):
        if self.should_stop():
            return None
        try:
            return person_sources.corruption_doc(self.wm, person, title)
        except FetchError as exc:
            self.registry.bump("retries")
            self.say(f"  skipped {title}: {exc}")
            return None

    # ---- one person ---------------------------------------------------------------------------------------
    def collect_person(self, person, only=None):
        """Every stage of one person. `only` = a list of stages: just those, again even if they ran before (a targeted request)."""
        self.say(f"== {person['name']} ({person.get('first_year') or '?'}-{person.get('last_year') or '?'})" + (f"  [only: {', '.join(only)}]" if only else ""))
        self.kb_urls = self.known_urls()                                    # what the knowledge base holds right now
        for source in STAGES:
            if (only and source not in only) or (not only and source in person["sources_done"]):
                continue
            self.check_stop()
            self.registry.set_stage(person["key"], source)
            self.on_stage(person, source, only)
            for attempt in range(1, STAGE_TRIES + 1):
                try:
                    items = self.collect_source(source, person)
                    break
                except FetchError as exc:
                    self.registry.bump("retries")
                    self.registry.event("retry", f"{source}: {exc}", person=person["name"])
                    if attempt == STAGE_TRIES:
                        raise
                    self.sleep(5 * attempt)
            stored = [d for d in (self.store(f, person) for f in items) if d]
            self.flush(stored)
            self.registry.source_done(person["key"], source)
            person["sources_done"].append(source)
            self.say(f"  {source}: {len(items)} file(s), {len(stored)} new")
        self.folders.sync_person(person)                                    # the person's own folder and manifest.json
        self.on_stage(person, None, only)

    def run_targeted(self, key, stages):
        """One person, some stages, on request. The person's roster status is not touched (a half-collected person stays in line for the full run).
        Returns 'done', 'stopped' or 'error'."""
        self.publish_pending()
        person = self.registry.person(key)
        if person is None:
            return "error"
        try:
            self.collect_person(person, only=list(stages))
            return "done"
        except Stopped:
            return "stopped"
        except Exception as exc:  # noqa: BLE001
            self.registry.bump("failures")
            self.registry.event("failed", str(exc)[:200], person=person["name"])
            self.say(f"  FAILED {person['name']}: {exc}")
            return "error"

    def run(self, limit=None, only=None):
        """Work through the roster. Returns 'done' (roster finished), 'stopped' or 'error'. `only` = a single roster key."""
        self.publish_pending()
        in_a_row, handled = 0, 0
        while True:
            try:
                self.check_stop()
            except Stopped:
                return "stopped"
            if (limit is not None and handled >= limit) or (only and handled):
                return "done"
            if only:
                person = self.registry.person(only)
                if person is not None:
                    self.registry._exec("UPDATE people SET status='in_progress', updated_at=? WHERE key=?", now(), only)
            else:
                person = self.registry.claim_next()
            if person is None:
                return "done"
            try:
                self.collect_person(person)
                self.registry.finish_person(person["key"], ok=True)
                in_a_row = 0
            except Stopped:
                self.say("stopped: the current person stays 'in progress' and resumes next time")
                return "stopped"
            except Exception as exc:  # noqa: BLE001 - one person must not end the run
                self.registry.finish_person(person["key"], ok=False, error=str(exc)[:300])
                self.registry.bump("failures")
                self.registry.event("failed", str(exc)[:200], person=person["name"])
                self.say(f"  FAILED {person['name']}: {exc}")
                in_a_row += 1
                if in_a_row >= FAILURES_IN_A_ROW:
                    self.say("too many failures in a row: the network or Wikimedia seems to be down; stopping")
                    return "error"
            handled += 1


class StoreSync(threading.Thread):
    """Keeps Chroma (and the corpus flags) in step while the collector runs, so no nightly Airflow job is needed: every `interval` seconds it runs the
    same check_stores pass: embed the new passages into Chroma, write what Neo4j still lacks, update the flags and the report."""

    def __init__(self, interval=20, say=print, registry=None):
        super().__init__(daemon=True)
        self.interval, self.say, self.done, self.lock, self.registry = interval, say, threading.Event(), threading.Lock(), registry

    def once(self):
        with self.lock:                                      # never two passes at the same time: they rewrite the same Parquet files
            try:
                from backend.knowledge import check_stores
                from backend.knowledge.kb import KnowledgeBase
                from backend.knowledge.vectorstore import RagStore
                kb = KnowledgeBase.from_env()
                if kb is None:
                    raise RuntimeError("Neo4j is not configured (NEO4J_URI / NEO4J_PASSWORD)")
                try:
                    report = check_stores.run(kb=kb, rag=RagStore.from_env())
                    pairs = self.registry.unlinked_in_graph() if self.registry is not None else []
                    if pairs:
                        self.registry.mark_graph_linked(kb.link_person_documents(pairs))
                finally:
                    kb.close()
                self.say(f"stores synced: {report['passages']} passages, {report['embedded_passages']} in Chroma, {report['neo4j_indexed_passages']} in Neo4j")
            except Exception as exc:  # noqa: BLE001 - a store being down must not stop the collector; the next pass tries again
                self.say(f"store sync skipped: {exc.__class__.__name__}: {str(exc)[:120]}")

    def run(self):
        self.once()
        while not self.done.wait(self.interval):
            self.once()


def run_process(limit=None, only=None, stages=None):
    """What the Start button runs: the whole collector as a process, reporting its state in the control file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(control.REPO / ".env", override=False)            # the store addresses and passwords, when started from a terminal
    except ImportError:
        pass
    registry = Registry()
    control.write({"state": "running", "pid": os.getpid(), "started_at": now(), "stop_requested": False,
                   "message": "Collecting politician by politician." if not stages else "Collecting " + ", ".join(stages) + " for one person."})
    stop = {"flag": False}

    def request(*_):
        stop["flag"] = True
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request)
    outcome, message = "error", ""
    try:
        if registry.counts()["total"] == 0 and not stages:
            control.write({"message": "Building the roster of politicians from Wikidata ..."})
            n = roster.build(registry, progress=lambda m: print(m, flush=True))
            if n == 0:
                raise RuntimeError("the roster is empty (Wikidata did not answer); try again in a few minutes")
        from backend.pipeline.kafka_io import KafkaSink
        from backend.pipeline.collectors import court_opendata
        from backend.knowledge.kb import KnowledgeBase
        halt = lambda: stop["flag"] or control.stop_requested()                          # noqa: E731
        index = official_index.OfficialIndex()
        control.write({"message": "Indexing court and agency records (the first run takes several minutes) ..."})
        needed = {k for st in (stages or STAGES) for k in OFFICIAL_KINDS.get(st, ())}
        if needed and not official_index.build(index, Fetcher(delay=3.0), should_stop=halt, say=lambda m: print(m, flush=True), kinds=needed):
            raise Stopped()
        try:
            s3 = court_opendata.S3Source()
        except Exception:  # noqa: BLE001 - no pyarrow / no network: court records are skipped, the rest still works
            s3 = None
        control.write({"message": "Collecting politician by politician." if not stages else "Collecting " + ", ".join(stages) + " for one person."})
        sync = StoreSync(say=lambda m: print(m, flush=True), registry=registry)
        docs_before = registry.document_totals()["documents"]
        began = {}

        def on_stage(person, stage, wanted):
            """Tell the dashboard (through the control file) which person and which source the collector is on right now."""
            if stage is None:
                control.write({"current": None})
                return
            began.setdefault(person["key"], now())
            control.write({"current": {"key": person["key"], "name": person["name"], "party": person.get("party"), "stage": stage,
                                       "wanted": list(wanted) if wanted else None, "since": began[person["key"]]}})
        sync.start()
        def known_urls():
            kb = KnowledgeBase.from_env()
            if kb is None:
                return set()
            try:
                return {r["u"] for r in kb._read("MATCH (s:Source) WHERE s.url IS NOT NULL RETURN s.url AS u")}
            except Exception:  # noqa: BLE001 - the graph is down: collect as usual, the loader's own duplicate check still protects the stores
                return set()
            finally:
                kb.close()
        collector = PersonCollector(registry, Wikimedia(), RawArchive(), KafkaSink(), should_stop=halt, say=lambda m: print(m, flush=True),
                                    index=index if s3 is not None else None, s3=s3, known_urls=known_urls, news_fetcher=Fetcher(delay=5.0), on_stage=on_stage,
                                    article_fetcher=Fetcher(delay=3.0, max_bytes=3_000_000))
        try:
            outcome = collector.run_targeted(only, stages) if stages else collector.run(limit=limit, only=only)
        finally:
            sync.done.set()
        control.write({"message": "Syncing Chroma with what Spark has processed ..."})
        time.sleep(45)                                       # Spark is still working through the last messages
        sync.once()
        new_docs = registry.document_totals()["documents"] - docs_before
        who = (registry.person(only) or {}).get("name", "") if only else ""
        found = f"{new_docs} new document{'' if new_docs == 1 else 's'}"
        if stages and outcome == "done":
            message = f"Finished {', '.join(stages)} for {who}: {found}." + ("" if new_docs else " Nothing new was found: the sources have no matching record, or it was collected before.")
        else:
            message = {"done": "The roster is finished." if not limit else "Finished the requested people.",
                       "stopped": "Stopped. Start again to resume with the same person.",
                       "error": "Stopped after repeated failures (network or Wikimedia down). Start again to retry."}[outcome] + f" ({found} in this run.)"
    except Stopped:
        outcome, message = "stopped", "Stopped while indexing court records. Start again to continue."
    except Exception as exc:  # noqa: BLE001
        message = f"The collector stopped with an error: {exc}"
        print(message, flush=True)
    control.write({"state": "idle" if outcome in ("done", "stopped") else "error", "pid": None, "stop_requested": False, "message": message, "current": None})
    return outcome


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "roster", "status"])
    ap.add_argument("--limit", type=int, default=None, help="stop after this many people")
    ap.add_argument("--person", default=None, help="collect only this roster key, e.g. wd:Q123")
    ap.add_argument("--sources", default=None, help="with --person: only these source groups, e.g. ed,courts (wikidata wikipedia articles courts ed dvac cbi)")
    a = ap.parse_args(argv)
    if a.command == "roster":
        print(f"roster: {roster.build()} people")
    elif a.command == "status":
        r = Registry()
        print(json.dumps({"people": r.counts(), "documents": r.document_totals(), "control": control.state()}, indent=1))
    else:
        stages = collect_request.stages_for([g for g in (a.sources or "").split(",") if g]) if a.sources else None
        sys.exit(0 if run_process(a.limit, a.person, stages) in ("done", "stopped") else 1)


if __name__ == "__main__":
    main()
