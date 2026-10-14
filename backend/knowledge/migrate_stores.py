"""Copy the local knowledge stores to hosted ones (for example Neo4j Aura and Chroma Cloud), without re-computing anything.

The SOURCE is the normal environment (NEO4J_*, CHROMA_*). The TARGET is the same variables with a TARGET_ prefix:

    TARGET_NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io   TARGET_NEO4J_USER=neo4j   TARGET_NEO4J_PASSWORD=...
    TARGET_CHROMA_API_KEY=... TARGET_CHROMA_TENANT=... TARGET_CHROMA_DATABASE=...        (Chroma Cloud)
    or TARGET_CHROMA_HOST=... TARGET_CHROMA_PORT=443 TARGET_CHROMA_SSL=1 [TARGET_CHROMA_TOKEN=...]   (a server you host)

    python -m backend.knowledge.migrate_stores --neo4j          # graph: export locally, restore into the EMPTY target graph
    python -m backend.knowledge.migrate_stores --neo4j --official-only   # same, without Wikipedia/notes passages (about 8,000 nodes instead of 64,000)
    python -m backend.knowledge.migrate_stores --chroma         # vectors: copy passages WITH their existing embeddings
    python -m backend.knowledge.migrate_stores --neo4j --chroma --dry-run     # only report what would be copied

Both steps are safe to repeat: the graph step refuses a target that is not empty, and the Chroma step skips ids the target has.
Redis needs no migration (it is only a cache). Afterwards `--verify` compares the counts of the two sides.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path


def copy_chroma(src, dst, batch=500, progress=print):
    """Copy every passage Chroma `src` has and `dst` lacks, together with its stored embedding (no model needed). Returns the count."""
    total, copied = src.count(), 0
    for offset in range(0, total, batch):
        got = src.col.get(include=["documents", "metadatas", "embeddings"], limit=batch, offset=offset)
        ids = got["ids"]
        todo = set(dst.missing(ids))
        keep = [i for i, x in enumerate(ids) if x in todo]
        if keep:
            dst.col.upsert(ids=[ids[i] for i in keep], documents=[got["documents"][i] for i in keep],
                           metadatas=[got["metadatas"][i] for i in keep], embeddings=[[float(v) for v in got["embeddings"][i]] for i in keep])
            copied += len(keep)
        progress(f"  chroma: {min(offset + batch, total)}/{total} scanned, {copied} copied", flush=True)
    return copied


def copy_graph(src_kb, dst_kb, workdir=None, official_only=False):
    """Export the source graph to a JSON-lines file and restore it into the (empty) target graph. Returns the counts.
    official_only leaves out Wikipedia / project-note passages (they are copied to the vector store, which is what serves them)."""
    path = Path(workdir or tempfile.mkdtemp()) / "graph_backup.jsonl"
    exported = src_kb.export_jsonl(path, skip_reference=official_only)
    imported = dst_kb.import_jsonl(path)
    if exported != imported:
        raise RuntimeError(f"the restored graph does not match the export: {exported} vs {imported}")
    return imported


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--neo4j", action="store_true")
    ap.add_argument("--chroma", action="store_true")
    ap.add_argument("--official-only", action="store_true", help="graph: leave out reference (Wikipedia/notes) passages; the vector store keeps them")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="compare counts of source and target")
    a = ap.parse_args(argv)
    from backend.knowledge.kb import KnowledgeBase
    from backend.knowledge.vectorstore import RagStore, configured
    if not (a.neo4j or a.chroma or a.verify):
        ap.error("choose --neo4j and/or --chroma (or --verify)")
    if a.neo4j or a.verify:
        if not os.environ.get("TARGET_NEO4J_URI"):
            sys.exit("TARGET_NEO4J_URI is not set")
        src, dst = KnowledgeBase(), KnowledgeBase(os.environ["TARGET_NEO4J_URI"], os.environ.get("TARGET_NEO4J_USER", "neo4j"), os.environ.get("TARGET_NEO4J_PASSWORD", ""))
        count = lambda kb: kb._read("MATCH (n) RETURN count(n) AS n")[0]["n"]          # noqa: E731
        print(f"graph: source has {count(src):,} nodes; target has {count(dst):,}")
        if a.neo4j and a.dry_run and a.official_only:
            would = src.export_jsonl(Path(tempfile.mkdtemp()) / "dry.jsonl", skip_reference=True)
            print(f"graph: --official-only would copy {would['nodes']:,} nodes and {would['relationships']:,} relationships")
        if a.neo4j and not a.dry_run:
            print("restored into the target:", copy_graph(src, dst, official_only=a.official_only))
            print(f"graph: target now has {count(dst):,} nodes; audit problems on the target: {dst.audit()}")
    if a.chroma or a.verify:
        if not configured("TARGET_"):
            sys.exit("no TARGET_CHROMA_* variables are set")
        s, d = RagStore(), RagStore(prefix="TARGET_")
        print(f"chroma: source has {s.count():,} passages; target has {d.count():,}")
        if a.chroma and not a.dry_run:
            print("chroma: copied", copy_chroma(s, d), f"passages; target now has {d.count():,}")


if __name__ == "__main__":
    main()
