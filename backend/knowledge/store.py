"""Write a document to BOTH stores: Neo4j (the knowledge graph) and Chroma (the RAG vector store).

Order matters: Neo4j first (one all-or-nothing transaction), then Chroma (idempotent upserts with the same passage ids).
If Chroma fails after Neo4j succeeded, the next attempt sees the document as a duplicate in Neo4j but still tops up
whatever is missing in Chroma, so the two stores converge instead of drifting apart. `consistency()` reports any gap.
"""
import hashlib
import os

from backend.knowledge.kb import passage_id


def digest_of(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def graph_wants_reference():
    """GRAPH_REFERENCE_DOCS=0 keeps Wikipedia / project-note passages out of Neo4j (they live in Chroma only). Used for a hosted graph with a
    node limit; the default (unset) stores everything in both, as before."""
    return os.environ.get("GRAPH_REFERENCE_DOCS", "1").strip().lower() not in ("0", "false", "no", "off")


def store_document(kb, rag, *, title, doc_type, text, passages, url=None, publisher=None, published_date=None, licence=None, cache=None):
    """Returns 'loaded' (new) or 'duplicate' (already stored). `rag` and `cache` may be None.
    A reference document goes to the vector store only when GRAPH_REFERENCE_DOCS is off."""
    if doc_type == "reference" and not graph_wants_reference():
        if rag is None or not passages:
            return "duplicate"
        h = digest_of(text)
        ids = [passage_id(h, i) for i in range(len(passages))]
        todo = set(rag.missing(ids))
        if todo:
            rag.add([{"id": ids[i], "text": p["text"], "meta": {"title": title, "url": url, "doc_type": doc_type, "publisher": publisher,
                                                                "published_date": published_date, "licence": licence, "locator": p.get("locator"), "position": i}}
                     for i, p in enumerate(passages) if ids[i] in todo])
            if cache is not None:
                cache.bump_version()
        return "loaded" if todo else "duplicate"
    sid, created = kb.add_document(title, doc_type, text, passages, url=url, publisher=publisher,
                                   published_date=published_date, licence=licence)
    if rag is not None and passages:
        h = digest_of(text)
        ids = [passage_id(h, i) for i in range(len(passages))]
        todo = set(rag.missing(ids))
        if todo:
            rag.add([{"id": ids[i], "text": p["text"],
                      "meta": {"source_id": sid, "title": title, "url": url, "doc_type": doc_type, "publisher": publisher,
                               "published_date": published_date, "licence": licence, "locator": p.get("locator"), "position": i}}
                     for i, p in enumerate(passages) if ids[i] in todo])
    if created and cache is not None:
        cache.bump_version()
    return "loaded" if created else "duplicate"


def consistency(kb, rag, sample=2000):
    """Compare the two stores: counts, and (for a sample of Neo4j passages) which ids are missing from Chroma."""
    neo = kb._read("MATCH (p:Passage) RETURN count(p) AS n")[0]["n"]
    rows = kb._read("MATCH (p:Passage) RETURN p.id AS id LIMIT $n", n=sample)
    missing = rag.missing([r["id"] for r in rows])
    return {"neo4j_passages": neo, "chroma_passages": rag.count(), "sampled": len(rows), "sample_missing_in_chroma": len(missing)}
