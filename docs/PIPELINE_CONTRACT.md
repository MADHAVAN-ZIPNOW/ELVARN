# ELVARN politician-wise collection pipeline: contract

One document describes the interfaces between the four parts, so each can be built and tested on its own.

```
Collector (per politician) -> kb_raw/ -> Kafka -> Spark -> corpus/ (Parquet) + Neo4j (+ entities, relations) -> Chroma (nightly check_stores)
```

Scope: Tamil Nadu politicians, 1947 to 2026. The collector works through the roster one person at a time (chief ministers first, then
ministers, then MLAs), finishes a person, then moves to the next. It can be started and stopped from the app (owner only).
This pipeline only collects and ingests. It never trains a model and never writes answers.

## 1. Who owns which files

| Part | Owner | Files |
|---|---|---|
| Collector, registry, roster, control, monitor, HTTP routes | collector side | `backend/pipeline/{registry,roster,person_sources,person_collector,control,monitor}.py`, `backend/api/server.py` (`/api/collect/*` only), `backend/tests/test_person_collector.py`, `test_collect_api.py` |
| Spark / processing / stores | processing side | `backend/pipeline/{corpus,entities,process,spark_job}.py`, `backend/knowledge/{kb.py,schema.cypher,check_stores.py}`, `dags/elvarn_tn_corruption.py`, `docker-compose.pipeline.yml`, `backend/tests/test_corpus_pipeline.py` |
| App button + dashboard | frontend side | `frontend-react/src/**`, `frontend-react/dist/**` |

Nobody edits another part's files. Read them freely.

## 2. Raw archive (collector writes, Spark reads)

* Layout: `backend/data/kb_raw/{source}/{yyyy-mm-dd}/{doc_id}.{ext}`; `source` is one of `web`, `pdf`, `upload`, `dataset`.
  Examples: `kb_raw/web/2026-10-16/doc_3f2a....html`, `kb_raw/dataset/2026-10-16/doc_91c0....json`.
* `doc_id` = `"doc_" + sha256(file bytes)[:20]`. The same bytes always give the same doc_id (idempotent).
* Originals are never modified. A file whose sha256 is already registered is skipped (no new file, no new message).
* `kb_raw/checksums.txt`: one line per file, `<sha256>  <relative path>`.
* The old content-addressed files (`kb_raw/ab/<sha>.pdf`) stay where they are; both layouts are valid `raw_path` values.
* A `raw_path` in a message starts with `kb_raw/`. Code that opens it must accept it with or without that prefix
  (root = `backend/data/kb_raw`, inside the Spark container `/opt/elvarn/backend/data/kb_raw`).

Wikidata files (`dataset`, `.json`) hold the raw entity JSON returned by `Special:EntityData/<QID>.json`.
Wikipedia files (`web`, `.html`) hold the raw HTML returned by the REST endpoint `/api/rest_v1/page/html/<title>`.

## 3. Kafka message (topic `kb.raw_documents`, one per downloaded file, pointer only, never the file bytes)

```json
{
  "event_type": "document_ingested",
  "doc_id": "doc_3f2a...",
  "source": "web",
  "source_url": "https://en.wikipedia.org/wiki/...",
  "raw_path": "kb_raw/web/2026-10-16/doc_3f2a....html",
  "mime_type": "text/html",
  "file_size_bytes": 123456,
  "sha256": "...",
  "language": "en",
  "collected_at": "2026-10-16T09:30:00+00:00",
  "status": "NEW",

  "url": "same as source_url (the existing code keys on it)",
  "title": "Article title",
  "doc_type": "reference",
  "publisher": "Wikipedia",
  "licence": "CC BY-SA 4.0",
  "published_date": null,
  "site": "politician",
  "person_key": "wd:Q123",
  "person_name": "Muthuvel Karunanidhi",
  "topic": "person | corruption | structured",
  "wikidata_id": "Q123 or null"
}
```

* `topic = person`: the politician's own article. `corruption`: an article about a case or scandal that mentions them.
  `structured`: the Wikidata record (relationships).
* The existing messages from the DVAC / CBI / court collectors (no `event_type`) keep working.

## 4. Spark outputs

### 4.1 Parquet corpus (canonical training corpus)

`backend/data/corpus/passages/source={source}/date={yyyy-mm-dd}/part-*.parquet` (date = collection date of the document).

Columns: `passage_id, doc_id, source, source_url, title, text, language (en|ta), char_count, token_estimate, sha256, created_at,
quality_score, embedded (bool), indexed_neo4j (bool)` plus the extra columns `person_key, person_name, topic, corruption_related (bool), position (int)`.

* `token_estimate = ceil(char_count / 4)`.
* Passages are 300 to 1,000 estimated tokens with 10 to 20 percent overlap. The last passage of a document may be shorter only if
  the whole document is shorter than 300 tokens (then it is kept if the document passes the filters, otherwise dropped).
* `passage_id` = `kb.passage_id(sha256_of_document_text, position)`: the same id Neo4j and Chroma use.
* Exact duplicate passages (same sha256 of the text) are written once.
* Quality filters (reject the document or passage): fewer than 1,000 characters (document level), under 70 percent letters or useful text,
  mostly links/menus/forms, excessive repeated lines, poor extraction, exact duplicate, garbled legacy-font Tamil.
* Boilerplate removal for HTML: scripts, CSS, navigation, headers, footers, cookie notices, reference lists, infoboxes, edit links, hatnotes.
* A run writes its counters to `backend/data/corpus/_stats/<yyyy-mm-dd>.jsonl` (one JSON object per batch) so the monitor can read them.

### 4.2 Neo4j (additive: nothing existing is renamed or removed; the chat keeps working)

* `(:Source:Document)`: existing `Source` nodes also get the label `Document` and the properties `doc_id, source, source_url, created_at, language, person_key`.
* `(:Source)-[:HAS_PASSAGE]->(:Passage)` in addition to the existing `(:Passage)-[:FROM]->(:Source)`.
* `Passage` gets `passage_id` (= `id`), `doc_id, source, source_url, created_at, language, token_estimate, sha256, quality_score, corruption_related`.
* `(:Passage)-[:MENTIONS]->(:Entity)`.
* `(:Entity)-[:RELATION {type, method, confidence, doc_id, passage_id, source, source_url, created_at, assertion_level, start, end}]->(:Entity)`.
  `method` is `wikidata` (structured, high confidence) or `cooccurrence` (text, low). Entity nodes are shared across documents, so they keep
  `first_doc_id, first_passage_id, source, source_url, created_at`; the relationship carries the exact `doc_id` and `passage_id`.
* Entity types: the existing `Person, Organisation, InvestigatingAgency, Court, PoliticalParty, Case, Act, Section, Place, Event, Concept, Scheme`
  plus `Date`, `Reference`, `Position`.
* Person entities from a roster record carry `person_key` and `wikidata_id`, and use the existing `Person:<lowercase name>` key.
* **Assertion levels** (the project rule): a relation or fact from a reference work is `reported` at most; `court_finding` only from a
  document of type `judgment` or `order`. A person is never linked to a case with a stronger level than the document supports.
  Co-occurrence is not an accusation: it is stored as `type = "MENTIONED_WITH"`, `method = "cooccurrence"`, `confidence = "low"`.
* Wikidata relations (`method = "wikidata"`): `MEMBER_OF_PARTY`, `HELD_POSITION`, `RELATIVE_OF` (with kind), `BORN_IN`, `EDUCATED_AT`, `CONSTITUENCY`
  where the record has them, with `start` / `end` dates.
* All writes are idempotent (`MERGE` on stable keys). Re-processing a doc_id never duplicates a node or relation.

### 4.3 Chroma

* Id = `passage_id`. Metadata: `doc_id, source, source_url, language, title, created_at` (plus the existing keys).
* Written by the nightly `check_stores` (the Spark image cannot run Chroma's client).

### 4.4 Nightly `check_stores` (`python -m backend.knowledge.check_stores`)

1. Passages with `embedded = false` get Chroma embeddings. 2. Passages with `indexed_neo4j = false` get their Neo4j nodes and relations.
3. The flags are updated (Parquet rewritten atomically). 4. Prints and saves a JSON report to `backend/data/corpus/_stats/report.json`:

```json
{"raw_files": 0, "extracted_documents": 0, "passages": 0, "clean_tokens": 0, "duplicates_removed": 0,
 "failed_extractions": 0, "embedded_passages": 0, "neo4j_indexed_passages": 0}
```

## 5. Collector state (collector side only, others read it through the HTTP API)

`backend/data/kb_state/collector.sqlite`: tables `people` (roster + status), `documents` (one row per downloaded file), `events` (movement feed).
`backend/data/kb_state/collector.control.json`: `{"state": "running|stopping|idle", "pid": 0, "started_at": "...", "stop_requested": false}`.

## 6. HTTP API (all `/api/collect/*` routes need the owner key header `X-Owner-Key`; otherwise 403)

`GET /api/collect/status`

```json
{
  "running": true, "state": "idle | running | stopping | error", "started_at": "ISO or null", "elapsed_s": 0,
  "scope": {"from_year": 1947, "to_year": 2026},
  "message": "human readable line",
  "people": {
    "total": 0, "done": 0, "in_progress": 0, "pending": 0, "failed": 0,
    "current": {"key": "wd:Q1", "name": "...", "party": "...", "stage": "wikipedia_en | wikipedia_ta | wikidata | corruption_articles | finishing",
                "docs": 0, "since": "ISO"},
    "recent": [{"key": "...", "name": "...", "status": "done | failed", "docs": 0, "finished_at": "ISO"}],
    "queue": [{"key": "...", "name": "..."}]
  },
  "stages": {
    "collector": {"documents": 0, "bytes": 0, "per_minute": 0.0, "failed": 0, "retries": 0, "last": {"title": "", "url": "", "at": "ISO"}},
    "raw":       {"files": 0, "bytes": 0, "by_source": {"web": 0, "pdf": 0, "dataset": 0, "upload": 0}, "free_disk_bytes": 0},
    "kafka":     {"up": true, "topic": "kb.raw_documents", "messages": 0, "consumer_lag": 0},
    "spark":     {"up": true, "processed_documents": 0, "skipped": 0, "failed": 0, "dead_letters": 0},
    "corpus":    {"passages": 0, "clean_tokens": 0, "target_tokens": 1000000000, "duplicates_removed": 0, "files": 0},
    "neo4j":     {"up": true, "documents": 0, "passages": 0, "entities": 0, "relationships": 0, "people": 0},
    "chroma":    {"up": true, "passages": 0, "embedded_pct": 0.0}
  },
  "movement": [{"at": "ISO", "doc_id": "doc_...", "title": "", "person": "", "stage": "collected | raw | kafka | spark | corpus | neo4j | chroma", "detail": ""}]
}
```

Any field may be `null` when the stage cannot be read (a store is down): the UI must show "unavailable", never crash. `movement` is newest first, at most 50.

`POST /api/collect/start` (body `{}`) -> same JSON as status. `POST /api/collect/stop` -> same JSON as status (graceful: the current document finishes).
`GET /api/collect/people?status=&q=&limit=50&offset=0` ->
`{"total": 0, "items": [{"key": "", "name": "", "party": "", "first_year": 1952, "last_year": 1957, "positions": [""], "status": "pending|in_progress|done|failed",
"docs": 0, "tokens": 0, "corruption_docs": 0, "updated_at": "ISO"}]}`.
`GET /api/collect/person/{key}` -> `{"person": {...as above...}, "documents": [{"doc_id": "", "title": "", "topic": "", "source": "", "url": "", "bytes": 0, "collected_at": ""}]}`.

## 7. Limits and rules for the collector

* 4 concurrent downloads, a polite delay per host, `robots.txt` respected, retry with back-off, resume after a stop or crash (a person stays `in_progress`
  until all its sources are finished, finished sources are skipped on resume).
* Only openly licensed or public-record sources: Wikipedia (English and Tamil, CC BY-SA), Wikidata (CC0), the existing government and court collectors.
  No news-site scraping, no login-only or private content.
* The 1,000,000,000 clean-token target is a target to measure against, not a promise: the politician-wise sources cannot reach it alone.
