"""Turn one collected document message into something loadable: text -> passages -> proposed facts.

Pure Python (no Spark, no Neo4j), so it is unit-tested directly and Spark just maps it over the stream.

Two kinds of message: the old collectors' (no event_type: text -> chunks -> proposed facts, loaded by KnowledgeBase.add_document) and
the politician-wise collector's `document_ingested` (clean corpus: HTML/PDF/DOCX/JSON -> corpus.py -> passage records, entities and
relations, loaded by KnowledgeBase.add_document_graph; the records are also the rows of the Parquet corpus).
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.knowledge import review
from backend.knowledge.ingest_wikipedia import chunk
from backend.pipeline import corpus, entities

RAW_ROOT = Path(__file__).resolve().parent.parent / "data" / "kb_raw"
MIN_STRUCTURED_CHARS = 100          # a Wikidata record is a short summary: the 1,000 character rule is for articles

looks_garbled = corpus.looks_garbled          # extended in corpus.py (Tamil vowel signs count as letters, legacy-font patterns); old name kept


def resolve_raw(raw_root, raw_path):
    """The archived file for a message's raw_path, with or without the leading 'kb_raw/' (contract section 2)."""
    rel = str(raw_path).replace(chr(92), "/").lstrip("/")
    for prefix in ("backend/data/kb_raw/", "kb_raw/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
    return Path(raw_root) / rel


def read_raw(msg, raw_root):
    """-> {"pages": [(locator, text)], "link_ratio": float, "json": parsed JSON or None}. PDFs with unreadable text give no pages."""
    path = resolve_raw(raw_root, msg["raw_path"])
    data = path.read_bytes()
    ext = path.suffix.lower()
    mime = msg.get("mime_type") or ""
    out = {"pages": [], "link_ratio": 0.0, "json": None}
    if ext == ".json" or "json" in mime:
        out["json"] = json.loads(data.decode("utf-8-sig"))
        if isinstance(out["json"], dict) and msg.get("labels_path"):                     # Special:EntityData holds ids only: the collector archives the
            try:                                                                          # resolved names as a second file, the original is never edited
                labels = json.loads(resolve_raw(raw_root, msg["labels_path"]).read_bytes().decode("utf-8-sig")).get("labels") or {}
                out["json"]["labels"] = {**(out["json"].get("labels") or {}), **labels}
            except (OSError, ValueError):
                pass
    elif ext == ".pdf":
        from io import BytesIO
        from pypdf import PdfReader
        pages = [(f"page {i + 1}", (p.extract_text() or "").strip()) for i, p in enumerate(PdfReader(BytesIO(data)).pages)]
        pages = [(loc, t) for loc, t in pages if t]
        out["pages"] = [] if looks_garbled(" ".join(t for _, t in pages)) else pages
    elif ext in (".html", ".htm") or "html" in mime:
        text, out["link_ratio"] = corpus.html_to_text_info(data)
        out["pages"] = [(None, text)]
    else:
        from backend.files import readers
        from backend.knowledge.ingest_folder import doc_text
        kind = {".txt": "txt", ".md": "md", ".docx": "docx"}.get(ext, "txt")
        out["pages"] = [(None, doc_text(readers.read(kind, data)))]
    return out


def raw_pages(msg, raw_root):
    """[(locator, text)] from the archived raw file (PDF, HTML, text). PDFs with unreadable text return []."""
    return read_raw(msg, raw_root)["pages"]


def is_ingest_event(msg):
    """Messages from the politician-wise collector (contract section 3); the older collectors send no event_type."""
    return msg.get("event_type") == "document_ingested"


def process_message(msg, raw_root=RAW_ROOT):
    """-> {"title", "doc_type", "url", "publisher", "published_date", "licence", "text", "passages": [...], "notes": [...]}
    or {"skipped": True, ...} when there is no usable text (the raw file is still archived).
    A document_ingested message goes through process_document; the others keep the old path."""
    if is_ingest_event(msg):
        return process_document(msg, raw_root)
    notes = []
    if msg.get("text"):
        pages = [(msg.get("locator") or "listing", msg["text"])]
    elif msg.get("raw_path"):
        pages = raw_pages(msg, raw_root)
        if not pages:
            notes.append("no reliable text in the raw file (scanned or legacy-encoded); needs OCR or a manual transcript")
    else:
        pages = []
    if not pages:
        return {"skipped": True, "url": msg.get("url"), "notes": notes}
    passages = []
    for locator, text in pages:
        for part in chunk(text):
            cands = []
            for sentence in review.SENTENCE.split(part):
                sentence = sentence.strip()
                if 20 <= len(sentence) <= 600:
                    cands += [{**p, "sentence": sentence} for p in review.propose(sentence, msg["doc_type"], operative=bool(review.OPERATIVE.search(sentence)))]
            passages.append({"locator": locator or f"chunk {len(passages) + 1}", "text": part, "candidates": cands})
    return {"skipped": False, "title": msg["title"], "doc_type": msg["doc_type"], "url": msg.get("url"),
            "publisher": msg.get("publisher"), "published_date": msg.get("published_date"), "licence": msg.get("licence"),
            "text": "\n\n".join(t for _, t in pages), "passages": passages, "notes": notes}


def collection_date(msg):
    """yyyy-mm-dd the document was collected: the folder date of its raw_path, else collected_at, else today."""
    m = re.search(r"/(\d{4}-\d{2}-\d{2})/", "/" + str(msg.get("raw_path") or "").replace(chr(92), "/"))
    return m.group(1) if m else (str(msg.get("collected_at") or "")[:10] or datetime.now(timezone.utc).date().isoformat())


def news_text(item):
    """One sentence from a news-index record: the headline, who published it and when. Nothing beyond what the record says."""
    seen = str(item.get("seendate") or "")
    date = f"{seen[0:4]}-{seen[4:6]}-{seen[6:8]}" if len(seen) >= 8 and seen[:8].isdigit() else None
    where = item.get("domain") or "a news site"
    country = f" ({item['sourcecountry']})" if item.get("sourcecountry") else ""
    return f"{str(item['title']).strip().rstrip('.')}. Reported by {where}{country}" + (f" on {date}." if date else ".")


def process_document(msg, raw_root=RAW_ROOT):
    """A collected document -> clean text, quality gate, passage records, entities and relations (no store is touched).
    -> {"skipped": True, "reasons": [...], ...} or {"skipped": False, "doc": {...}, "records": [...], "entities": [...],
    "relations": [...], "passages": [...], "text", ...}. `records` are the Parquet rows (plus per-passage entities and relations);
    `passages` keeps the old shape so load_result / load_many accept the result too."""
    doc_id, notes = msg["doc_id"], []

    def skip(reasons):
        return {"skipped": True, "new": True, "doc_id": doc_id, "url": msg.get("url") or msg.get("source_url"), "title": msg.get("title"),
                "reasons": reasons, "notes": notes + [f"rejected: {', '.join(reasons)}"]}

    raw = read_raw(msg, raw_root) if msg.get("raw_path") else {"pages": [(None, msg.get("text") or "")], "link_ratio": 0.0, "json": None}
    person = {"name": msg.get("person_name"), "person_key": msg.get("person_key"), "wikidata_id": msg.get("wikidata_id"),
              "aliases": msg.get("aliases") or []}
    doc_entities, doc_relations, min_chars = [], [], corpus.MIN_DOC_CHARS
    if raw["json"] is not None and msg.get("topic") == "news" and isinstance(raw["json"], dict) and raw["json"].get("title"):
        text, min_chars = news_text(raw["json"]), MIN_STRUCTURED_CHARS               # a headline record: the article itself is not copied
    elif raw["json"] is not None:
        if not isinstance(raw["json"], dict) or "entities" not in raw["json"]:
            return skip(["unsupported_json"])
        text = entities.wikidata_summary(raw["json"])
        doc_entities, doc_relations = entities.relations_from_wikidata(raw["json"], msg.get("person_key"))
        min_chars = MIN_STRUCTURED_CHARS
        if not person["name"]:
            person["name"] = next((e["name"] for e in doc_entities if e["type"] == "Person"), None)
    else:
        text = "\n\n".join(corpus.unwrap_lines(t) if (loc or "").startswith("page") else t for loc, t in raw["pages"])
        if not raw["pages"]:
            notes.append("no reliable text in the raw file (scanned or legacy-encoded)")
    text = corpus.normalize(text)
    language = corpus.detect_language(text)
    if language == "unknown":
        return skip(["language_unknown"])
    score, reasons = corpus.quality(text, raw["link_ratio"], min_chars)
    if reasons:
        return skip(reasons)
    meta = {"doc_id": doc_id, "source": msg.get("source") or "web", "source_url": msg.get("source_url") or msg.get("url"), "title": msg.get("title"),
            "person_key": msg.get("person_key"), "person_name": person["name"], "topic": msg.get("topic"), "created_at": msg.get("collected_at")}
    records, info = corpus.build_passages(meta, text, language, score)
    roster = [person] if person["name"] else []
    for rec in records:
        rec["entities"] = entities.extract_entities(rec["text"], roster)
        rec["relations"] = entities.cooccurrence_relations(rec["entities"])
    first_pid = records[0]["passage_id"] if records else None
    doc_relations = [{**r, "passage_id": first_pid} for r in doc_relations]
    graph_doc = {"doc_id": doc_id, "title": msg.get("title") or doc_id, "doc_type": msg.get("doc_type") or "reference",
                 "source": meta["source"], "source_url": meta["source_url"], "language": language, "person_key": msg.get("person_key"),
                 "created_at": msg.get("collected_at"), "published_date": msg.get("published_date"), "publisher": msg.get("publisher"),
                 "licence": msg.get("licence"), "sha256": info["document_sha256"], "raw_sha256": msg.get("sha256")}
    if msg.get("case_stage"):                                             # a court or agency record of a person's case: where it is in the case lifecycle
        graph_doc.update(case_stage=msg.get("case_stage"), petitions=msg.get("petitions") or [])
    if msg.get("topic") == "judgment":                                    # a court judgment: the case, the court, and what the WHOLE text is about
        from backend.pipeline import law_tags
        graph_doc.update({k: msg.get(k) for k in ("case_name", "case_type", "case_number", "court", "bench", "year", "judge", "outcome")},
                         case_ref=msg.get("ref"), subjects=law_tags.subjects_of(text), statutes=law_tags.statutes_of(text, 20))
    return {"skipped": False, "new": True, "doc_id": doc_id, "date": collection_date(msg), "source": meta["source"], "language": language,
            "quality_score": score, "title": graph_doc["title"], "doc_type": graph_doc["doc_type"], "url": meta["source_url"],
            "publisher": msg.get("publisher"), "published_date": msg.get("published_date"), "licence": msg.get("licence"),
            "text": text, "passages": [{"locator": f"passage {r['position'] + 1}", "text": r["text"], "candidates": []} for r in records],
            "doc": graph_doc, "records": records, "entities": doc_entities, "relations": doc_relations,
            "duplicates": info["duplicates"], "rejected_passages": info["rejected"], "notes": notes}


def load_document_graph(kb, result):
    """Neo4j part of a processed document (Document, Passages, entities, mentions, relations). 'loaded' or 'duplicate'; idempotent."""
    _, created = kb.add_document_graph(result["doc"], result["records"], result["entities"], result["relations"])
    return "loaded" if created else "duplicate"


def load_result(kb, result, rag=None, cache=None):
    """Store a processed document in BOTH stores (Neo4j graph + Chroma vectors). Returns 'loaded', 'duplicate' or 'skipped'.
    A document_ingested result goes to Neo4j only: its Chroma vectors are written by the nightly check_stores."""
    if result.get("skipped"):
        return "skipped"
    if result.get("records") is not None:
        return load_document_graph(kb, result)
    from backend.knowledge.store import store_document
    return store_document(kb, rag, title=result["title"], doc_type=result["doc_type"], text=result["text"], passages=result["passages"],
                          url=result["url"], publisher=result["publisher"], published_date=result["published_date"],
                          licence=result["licence"], cache=cache)


def is_transient(exc):
    """Failures that will probably go away by themselves (a store is down or busy): the batch must be retried, not skipped."""
    from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
    return isinstance(exc, (ServiceUnavailable, SessionExpired, TransientError, ConnectionError, TimeoutError))


def load_many(kb, results, rag=None, cache=None, on_failure=None):
    """Load processed documents one by one and count the outcomes ('loaded', 'duplicate', 'skipped', 'failed').
    A document that cannot be stored for a reason of its own (bad data) is reported to `on_failure(result, exc)` and skipped,
    so one poisonous document never blocks the ones behind it. A transient failure (a store is down) is raised: the caller
    retries the whole batch, and already-loaded documents are recognised as duplicates, so nothing is lost or doubled."""
    from collections import Counter
    counts = Counter()
    for result in results:
        try:
            counts[load_result(kb, result, rag, cache)] += 1
        except Exception as exc:  # noqa: BLE001
            if is_transient(exc):
                raise
            counts["failed"] += 1
            if on_failure is not None:
                on_failure(result, exc)
    return counts
