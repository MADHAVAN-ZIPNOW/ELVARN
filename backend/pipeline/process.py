"""Turn one collected document message into something loadable: text -> passages -> proposed facts.

Pure Python (no Spark, no Neo4j), so it is unit-tested directly and Spark just maps it over the stream.
"""
import re
from pathlib import Path

from backend.knowledge import review
from backend.knowledge.ingest_wikipedia import chunk

RAW_ROOT = Path(__file__).resolve().parent.parent / "data" / "kb_raw"


def looks_garbled(text):
    """Legacy-font Tamil PDFs extract as symbol soup. Real English/Unicode text is mostly letters, digits, spaces and punctuation."""
    sample = text[:4000]
    if not sample.strip():
        return True
    good = sum(1 for ch in sample if ch.isalnum() or ch in " \n\t.,;:()-/'\"%&₹")
    return "�" in sample or good / len(sample) < 0.85 or sum(1 for ch in sample if ord(ch) > 0xFFF0) > 0


def raw_pages(msg, raw_root):
    """[(locator, text)] from the archived raw file (PDF, HTML, text). PDFs with unreadable text return []."""
    path = Path(raw_root) / msg["raw_path"]
    data = path.read_bytes()
    ext = path.suffix.lower()
    if ext == ".pdf":
        from io import BytesIO
        from pypdf import PdfReader
        pages = [(f"page {i + 1}", (p.extract_text() or "").strip()) for i, p in enumerate(PdfReader(BytesIO(data)).pages)]
        pages = [(loc, t) for loc, t in pages if t]
        return [] if looks_garbled(" ".join(t for _, t in pages)) else pages
    from backend.files import readers
    from backend.knowledge.ingest_folder import doc_text
    kind = {".html": "html", ".htm": "html", ".txt": "txt", ".md": "md", ".docx": "docx"}.get(ext, "txt")
    return [(None, doc_text(readers.read(kind, data)))]


def process_message(msg, raw_root=RAW_ROOT):
    """-> {"title", "doc_type", "url", "publisher", "published_date", "licence", "text", "passages": [...], "notes": [...]}
    or None when there is no usable text (the raw file is still archived)."""
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


def load_result(kb, result, rag=None, cache=None):
    """Store a processed document in BOTH stores (Neo4j graph + Chroma vectors). Returns 'loaded', 'duplicate' or 'skipped'."""
    if result.get("skipped"):
        return "skipped"
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
