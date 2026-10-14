"""Bring Chroma in step with Neo4j: index every Neo4j passage that Chroma does not have yet.

Use it after Chroma was empty / restored, or to repair a gap (store.consistency() reports gaps).

    python -m backend.knowledge.reindex [--batch 256] [--verify]

--verify exits with an error if, after the repair, a sample of Neo4j passages is still missing from Chroma (used by the nightly job).
"""
import argparse
import time

from backend.knowledge.kb import KnowledgeBase
from backend.knowledge.store import consistency
from backend.knowledge.vectorstore import RagStore


def reindex(kb, rag, batch=256, progress=print):
    """Copy missing passages (text + source details) from Neo4j into Chroma. Returns the number written."""
    written, last = 0, ""
    while True:
        rows = kb._read("MATCH (p:Passage)-[:FROM]->(s:Source) WHERE p.id > $last "
                        "RETURN p.id AS id, p.text AS text, p.locator AS locator, p.position AS position, s.id AS source_id, s.title AS title, "
                        "s.url AS url, s.doc_type AS doc_type, s.publisher AS publisher, s.published_date AS published_date, s.licence AS licence "
                        "ORDER BY p.id LIMIT $n", last=last, n=batch)
        if not rows:
            return written
        last = rows[-1]["id"]
        todo = set(rag.missing([r["id"] for r in rows]))
        if todo:
            written += rag.add([{"id": r["id"], "text": r["text"], "meta": r} for r in rows if r["id"] in todo])
            progress(f"  indexed {written} passages so far", flush=True) if written % (batch * 4) < batch else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    kb, rag, t0 = KnowledgeBase(), RagStore(), time.time()
    n = reindex(kb, rag, a.batch)
    report = consistency(kb, rag, sample=5000)
    print(f"indexed {n} passages in {time.time() - t0:.0f}s;", report)
    kb.close()
    if a.verify and report["sample_missing_in_chroma"]:
        raise SystemExit("Chroma is still missing passages after the repair")


if __name__ == "__main__":
    main()
