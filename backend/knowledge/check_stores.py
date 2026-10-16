"""Nightly check of the three stores: Parquet corpus, Neo4j and Chroma (contract section 4.4).

    python -m backend.knowledge.check_stores [--verify] [--no-reindex]

1. Passages with embedded=false get their Chroma embedding (id = passage_id, metadata doc_id, source, source_url, language, title, created_at).
2. Passages with indexed_neo4j=false get their Document, Passage, entity and relation nodes in Neo4j (the same MERGE writes Spark uses).
3. The two flags are updated in the Parquet files (each file is rewritten to a temporary name and swapped in atomically); rows whose
   passage_id already appeared earlier (a retried Spark batch) are dropped on the way.
4. The JSON report is printed and saved to <corpus>/_stats/report.json.
Then reindex.reindex copies passages that only exist in Neo4j (the older collectors' documents) into Chroma, as before.
A store that is not configured or down is skipped and named in the report ("unavailable"): the Parquet counts are still reported.
Needs pyarrow (the corpus), neo4j and chromadb: run it in the Airflow RAG virtualenv.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from backend.pipeline import corpus, entities
from backend.pipeline.common import RAW_DIR

CORPUS_DIR = corpus.CORPUS_DIR
EMBED_BATCH = 256                     # one GPU call and one Chroma upsert per 256 passages (64 left the GPU waiting)


def parquet_files(corpus_dir=CORPUS_DIR):
    return sorted((Path(corpus_dir) / "passages").glob("source=*/date=*/*.parquet"))


def count_raw_files(raw_dir=RAW_DIR):
    skip = {"checksums.txt"}
    root = Path(raw_dir)
    return sum(1 for p in root.rglob("*") if p.is_file() and p.name not in skip and not p.name.endswith((".part", ".tmp"))
               and p.relative_to(root).parts[0] not in ("people", "courts"))                   # people/ holds links to the same files


def embed_rows(rag, rows):
    """Upsert the given rows (dicts) into Chroma with the contract's metadata. Idempotent. Returns how many were written."""
    todo = [r for r in rows if r["text"].strip()]
    for i in range(0, len(todo), EMBED_BATCH):
        part = todo[i:i + EMBED_BATCH]
        meta = [{k: v for k, v in {"doc_id": r["doc_id"], "source_id": r["doc_id"], "source": r["source"], "source_url": r["source_url"],
                                   "url": r["source_url"], "language": r["language"], "title": r["title"], "created_at": r["created_at"],
                                   "doc_type": "reference", "position": r["position"], "person_key": r["person_key"]}.items() if v is not None}
                for r in part]
        rag.col.upsert(ids=[r["passage_id"] for r in part], documents=[r["text"] for r in part], metadatas=meta)
    return len(todo)


def graph_rows(kb, rows):
    """Re-create the Neo4j nodes of passages whose indexed_neo4j flag is false (entities and co-occurrence are re-extracted from the text;
    a Wikidata record's own relations are only written by the Spark job). Returns how many passages were written."""
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)
    done = 0
    for doc_id, recs in by_doc.items():
        known = kb._read("MATCH (s:Source {id:$id}) RETURN s.doc_type AS t, s.publisher AS pub, s.licence AS lic, s.published_date AS pd", id=doc_id)
        first = recs[0]
        roster = [{"name": first["person_name"], "person_key": first["person_key"]}] if first["person_name"] else []
        records = []
        for r in recs:
            ents = entities.extract_entities(r["text"], roster)
            records.append({**r, "entities": ents, "relations": entities.cooccurrence_relations(ents)})
        doc = {"doc_id": doc_id, "title": first["title"] or doc_id, "doc_type": (known[0]["t"] if known else None) or "reference",
               "source": first["source"], "source_url": first["source_url"], "language": first["language"], "person_key": first["person_key"],
               "created_at": first["created_at"], "publisher": known[0]["pub"] if known else None,
               "licence": known[0]["lic"] if known else None, "published_date": known[0]["pd"] if known else None}
        kb.add_document_graph(doc, records)
        done += len(records)
    return done


def rewrite(path, table):
    """Swap `table` in for the Parquet file atomically (a reader sees the old file or the new one, never half of one)."""
    import pyarrow.parquet as pq
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def run(corpus_dir=CORPUS_DIR, raw_dir=RAW_DIR, kb=None, rag=None, batch_report=None):
    """Update the stores and flags, return the report dict (also saved to _stats/report.json)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    seen, docs, passages, tokens, embedded, indexed, dropped = set(), set(), 0, 0, 0, 0, 0
    problems = []
    for path in parquet_files(corpus_dir):
        table = pq.read_table(path)
        data = table.to_pydict()
        n = table.num_rows
        keep = []
        for i in range(n):
            pid = data["passage_id"][i]
            if pid not in seen:
                seen.add(pid)
                keep.append(i)
        dropped += n - len(keep)
        rows = [{c: data[c][i] for c in data} for i in keep]
        changed = len(keep) != n
        todo = [r for r in rows if not r["embedded"]]
        if todo and rag is not None:
            try:
                embed_rows(rag, todo)
                for r in todo:
                    r["embedded"], changed = True, True
            except Exception as exc:  # noqa: BLE001
                problems.append(f"chroma: {exc.__class__.__name__}: {str(exc)[:200]}")
                rag = None
        todo = [r for r in rows if not r["indexed_neo4j"]]
        if todo and kb is not None:
            try:
                graph_rows(kb, todo)
                for r in todo:
                    r["indexed_neo4j"], changed = True, True
            except Exception as exc:  # noqa: BLE001
                problems.append(f"neo4j: {exc.__class__.__name__}: {str(exc)[:200]}")
                kb = None
        if changed:
            rewrite(path, pa.Table.from_pylist(rows, schema=table.schema) if rows else table.slice(0, 0))
        docs.update(r["doc_id"] for r in rows)
        passages += len(rows)
        tokens += sum(r["token_estimate"] or 0 for r in rows)
        embedded += sum(1 for r in rows if r["embedded"])
        indexed += sum(1 for r in rows if r["indexed_neo4j"])
    if dropped:                                                          # counted once, in the stats, so a second night reports the same number
        corpus.append_stats(corpus_dir, datetime.now(timezone.utc).date().isoformat(), {"source": "check_stores", "duplicates_removed": dropped})
    totals = corpus.stats_totals(corpus_dir)
    report = {"raw_files": count_raw_files(raw_dir) if Path(raw_dir).exists() else 0, "extracted_documents": len(docs), "passages": passages,
              "clean_tokens": tokens, "duplicates_removed": totals["duplicates_removed"],
              "failed_extractions": totals["failed"] + totals["skipped"], "embedded_passages": embedded, "neo4j_indexed_passages": indexed}
    if problems or kb is None or rag is None:
        report["unavailable"] = problems + [name for name, store in (("neo4j", kb), ("chroma", rag)) if store is None and not any(p.startswith(name) for p in problems)]
    out = Path(corpus_dir) / "_stats" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="exit with an error if Chroma still lacks passages that Neo4j has")
    ap.add_argument("--no-reindex", action="store_true", help="skip the Neo4j -> Chroma repair of passages that are not in the corpus")
    a = ap.parse_args()
    from backend.knowledge.kb import KnowledgeBase
    from backend.knowledge.vectorstore import RagStore
    kb = KnowledgeBase.from_env()
    rag = RagStore.from_env()
    if kb is not None:
        kb.label_documents()
    report = run(kb=kb, rag=rag)
    if kb is not None and rag is not None and not a.no_reindex:
        from backend.knowledge.reindex import reindex
        from backend.knowledge.store import consistency
        reindex(kb, rag)
        gap = consistency(kb, rag, sample=5000)
        report["consistency"] = gap
        if a.verify and gap["sample_missing_in_chroma"]:
            print(json.dumps(report, indent=2))
            raise SystemExit("Chroma is still missing passages after the repair")
    if kb is not None:
        kb.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
