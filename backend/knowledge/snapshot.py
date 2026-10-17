"""Build and load the snapshot the free hosted service serves (committed in knowledge_service/snapshot/, baked into the image).

    knowledge_service/snapshot/
        manifest.json        counts, sha256 of each file, embedding model, when it was made
        graph.jsonl.gz       the lean graph (official documents, proposed + approved facts, entities): export_jsonl(skip_reference=True)
        passages.jsonl.gz    one line per passage for the vector store: {"id", "text", "meta"}
        vectors.npz          the passages' embeddings (float16, same order as passages.jsonl.gz): nothing is re-embedded when loading

What goes into the vector store (about 9,800 passages): every passage of every OFFICIAL document, the project's own notes/documents, and
the Wikipedia articles that are about Tamil Nadu AND corruption. General Wikipedia stays out (the live web search answers general
questions), which keeps the hosted copy small enough for a free 512 MB service.

    python -m backend.knowledge.snapshot export [--out knowledge_service/snapshot]     (needs the local Neo4j + Chroma)
    python -m backend.knowledge.snapshot check  [--dir knowledge_service/snapshot]     (verifies the files against the manifest)
    python -m backend.knowledge.snapshot load   --chroma-path /app/chroma_data         (fills an embedded Chroma; done once when the image is built)
"""
import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_service" / "snapshot"
MANIFEST, GRAPH, PASSAGES, VECTORS = "manifest.json", "graph.jsonl.gz", "passages.jsonl.gz", "vectors.npz"
NOTES_PUBLISHERS = ("ELVARN project",)
EMBEDDING_MODEL = "chroma default (all-MiniLM-L6-v2, 384 dimensions)"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


JUDGMENT_SUBJECTS = ("corruption", "election")   # which judgments the hosted copy carries (the subject tags of backend/pipeline/law_tags.py)
JUDGMENT_PASSAGES_PER_CASE = 5                    # the first passages of a judgment (headnote and the start of the reasoning): more cases instead of more of one. Measured: 15 per case (31,647 passages) was killed by the 512 MB limit under load, 5 (18,103 passages) stays under 380 MB with 4 questions at a time
JUDGMENT_REASON = "judgments (corruption + election, first %d passages of each)" % JUDGMENT_PASSAGES_PER_CASE


def select_passage_ids(kb):
    """Ids of the passages the hosted vector store should hold, with the reason (for the report)."""
    from backend.knowledge.ingest_wikipedia import is_relevant       # only the exporting machine needs this (the image does not contain it)
    ids, reasons = [], {"official": 0, "project notes/files": 0, "wikipedia (Tamil Nadu + corruption)": 0, JUDGMENT_REASON: 0}
    rows = kb._read("MATCH (s:Source)<-[:FROM]-(p:Passage) WHERE s.doc_type <> 'judgment' RETURN s.id AS sid, s.doc_type AS t, s.publisher AS pub, "
                    "collect(p.id) AS ids, collect(p.text) AS texts ORDER BY sid")      # one row per DOCUMENT (never per type)
    for r in rows:
        if r["t"] == "news":
            continue                                          # news articles belong to their publishers: kept in the private stores only, never in the public snapshot
        if r["t"] != "reference":
            reason = "official"
        elif (r["pub"] or "").startswith(NOTES_PUBLISHERS):
            reason = "project notes/files"
        elif is_relevant(" ".join(r["texts"])):
            reason = "wikipedia (Tamil Nadu + corruption)"
        else:
            continue
        ids.extend(r["ids"])
        reasons[reason] += len(r["ids"])
    cases = kb._read("MATCH (s:Source {doc_type: 'judgment'}) WHERE any(x IN coalesce(s.subjects, []) WHERE x IN $subjects) "
                     "MATCH (s)<-[:FROM]-(p:Passage) WITH s, p ORDER BY p.position "
                     "WITH s, collect(p.id)[0..$cap] AS ids RETURN s.id AS sid, ids ORDER BY sid",
                     subjects=list(JUDGMENT_SUBJECTS), cap=JUDGMENT_PASSAGES_PER_CASE)
    for r in cases:
        ids.extend(r["ids"])
        reasons[JUDGMENT_REASON] += len(r["ids"])
    return sorted(set(ids)), reasons


def export_snapshot(kb, rag, out_dir=DEFAULT_DIR, batch=400):
    """Write the snapshot files. Every selected passage must already be in Chroma (run reindex first); returns the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids, reasons = select_passage_ids(kb)
    missing = rag.missing(ids)
    if missing:
        raise RuntimeError(f"{len(missing)} selected passages are not in Chroma yet; run `python -m backend.knowledge.reindex` first")
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "graph.jsonl"
        counts = kb.export_jsonl(raw, skip_reference=True)
        with open(raw, "rb") as src, open(out / GRAPH, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, compresslevel=9, mtime=0) as dst:
            shutil.copyfileobj(src, dst)                     # no name and mtime=0 in the gzip header: the same graph gives the same bytes
    vectors = []
    with open(out / PASSAGES, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, compresslevel=9, mtime=0) as gz:
        for start in range(0, len(ids), batch):
            got = rag.col.get(ids=ids[start:start + batch], include=["documents", "metadatas", "embeddings"])
            order = {i: k for k, i in enumerate(got["ids"])}
            for pid in ids[start:start + batch]:
                k = order[pid]
                gz.write((json.dumps({"id": pid, "text": got["documents"][k], "meta": got["metadatas"][k]}, ensure_ascii=False, sort_keys=True) + chr(10)).encode("utf-8"))
                vectors.append(np.asarray(got["embeddings"][k], dtype=np.float16))
    np.savez_compressed(out / VECTORS, vectors=np.vstack(vectors) if vectors else np.zeros((0, 384), dtype=np.float16))
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "embedding_model": EMBEDDING_MODEL,
                "graph": counts, "passages": len(ids), "passages_by_reason": reasons,
                "files": {name: sha256(out / name) for name in (GRAPH, PASSAGES, VECTORS)}}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    return manifest


def check_snapshot(directory=DEFAULT_DIR):
    """Verify the files against the manifest; returns the manifest, raises ValueError on any mismatch."""
    d = Path(directory)
    manifest = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if not (d / name).is_file() or sha256(d / name) != digest:
            raise ValueError(f"snapshot file {name} is missing or does not match the manifest")
    return manifest


def read_passages(directory):
    """(ids, texts, metadatas, embeddings float32 matrix) in the snapshot's order."""
    d = Path(directory)
    ids, texts, metas = [], [], []
    with gzip.open(d / PASSAGES, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["id"])
            texts.append(r["text"])
            metas.append(r["meta"])
    vectors = np.load(d / VECTORS)["vectors"].astype(np.float32)
    if len(vectors) != len(ids):
        raise ValueError("the snapshot has a different number of vectors and passages")
    return ids, texts, metas, vectors


def clear(store):
    """Remove every passage from the store (in pages: Chroma limits how many ids one call may carry)."""
    while store.count():
        store.col.delete(ids=store.col.get(include=[], limit=2000)["ids"])


def load_vectors(directory, store, marker_file=None, batch=500):
    """Fill a RagStore from the snapshot (embeddings included, no model run). If `marker_file` already says the store holds exactly this
    snapshot (and the count agrees) nothing is done; otherwise the store is emptied and refilled. Returns how many passages were loaded."""
    manifest = check_snapshot(directory)
    marker = manifest["files"][PASSAGES]
    marker_file = Path(marker_file) if marker_file else None
    if marker_file and marker_file.is_file() and marker_file.read_text(encoding="utf-8").strip() == marker and store.count() == manifest["passages"]:
        return 0
    clear(store)
    ids, texts, metas, vectors = read_passages(directory)
    for start in range(0, len(ids), batch):
        sl = slice(start, start + batch)
        store.col.upsert(ids=ids[sl], documents=texts[sl], metadatas=metas[sl], embeddings=[v.tolist() for v in vectors[sl]])
    if marker_file:
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.write_text(marker + chr(10), encoding="utf-8")
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "check", "load"])
    ap.add_argument("--out", "--dir", dest="dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--chroma-path", type=Path, default=Path("/tmp/elvarn_chroma"))
    a = ap.parse_args()
    if a.cmd == "load":
        import chromadb
        from backend.knowledge.vectorstore import RagStore
        store = RagStore(client=chromadb.PersistentClient(path=str(a.chroma_path)))
        print("loaded", load_vectors(a.dir, store, marker_file=a.chroma_path / "snapshot.marker"), "passages;", store.count(), "in the store")
        return
    if a.cmd == "check":
        m = check_snapshot(a.dir)
        print("snapshot ok:", {k: m[k] for k in ("created_at", "graph", "passages")})
        return
    from backend.knowledge.kb import KnowledgeBase
    from backend.knowledge.vectorstore import RagStore
    kb, rag = KnowledgeBase(), RagStore()
    m = export_snapshot(kb, rag, a.dir)
    print(json.dumps({k: m[k] for k in ("created_at", "graph", "passages", "passages_by_reason")}, indent=2))
    print("sizes (MB):", {n: round((a.dir / n).stat().st_size / 1e6, 2) for n in (GRAPH, PASSAGES, VECTORS)})
    kb.close()


if __name__ == "__main__":
    main()
