"""Load the data that already exists in the project into BOTH stores (Neo4j graph + Chroma RAG), so retrieval covers it too.

Sources (all under backend/data):
  cache_wikipedia/*.json        the cached Wikipedia articles (CC BY-SA 4.0): one document each, with its article link
  raw_v2/domain_knowledge.txt   the project's own domain notes (DOMAIN_KNOWLEDGE / GENERAL_ENGLISH blocks): one small document per block
  raw_v2/{case1,collegium_statements_and_constitution,jay}.txt   the owner's own documents: one document per file

Skipped on purpose: dolly_qa.txt and oasst_chat.txt (generic chat/Q&A with no source documents), the CASUAL_CONVERSATION and HUMOUR
blocks, and raw_v2/wikipedia_*.txt (the training-format copy of the same articles). The project's own notes are marked "unverified,
pending source check" in their publisher field, exactly as the README says, so any answer that cites them shows that.

Safe to run again: documents are identified by the SHA-256 of their text and passages by deterministic ids.

    python -m backend.knowledge.ingest_existing [--limit N] [--only wikipedia|notes|files]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

from backend.knowledge.ingest_wikipedia import LICENCE as WIKI_LICENCE
from backend.knowledge.ingest_wikipedia import chunk, ingest_article
from backend.knowledge.store import store_document

DATA = Path(__file__).resolve().parent.parent / "data"
NOTES_PUBLISHER = "ELVARN project domain notes (unverified, pending source check)"
FILES_PUBLISHER = "ELVARN project documents (the owner's own material)"
BLOCK = re.compile(r"<(\w+)>\n(.*?)\n</\1>", re.S)
OWN_FILES = ("case1.txt", "collegium_statements_and_constitution.txt", "jay.txt")


def wikipedia(kb, rag, cache, limit=None):
    done = 0
    for path in sorted((DATA / "cache_wikipedia").glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not rec.get("text") or not rec.get("title") or re.match(r"^(List of|Index of|Outline of)", rec["title"]):
            continue
        ingest_article(kb, rec["title"], rec["text"], require_relevant=False, rag=rag, cache=cache)
        done += 1
        if done % 200 == 0:
            print(f"  wikipedia: {done} articles", flush=True)
        if limit and done >= limit:
            break
    return done


def notes(kb, rag, cache, limit=None):
    path = DATA / "raw_v2" / "domain_knowledge.txt"
    if not path.is_file():
        return 0
    done = 0
    for tag, body in BLOCK.findall(path.read_text(encoding="utf-8", errors="replace")):
        if tag not in ("DOMAIN_KNOWLEDGE", "GENERAL_ENGLISH"):
            continue
        topic = re.search(r"^Topic: (.+)$", body, re.M)
        q = re.search(r"^Question: (.+?)$", body, re.M)
        a = re.search(r"^Answer: (.*)\Z", body, re.M | re.S)
        text = (f"{q.group(1).strip()}\n{a.group(1).strip()}" if q and a else body).strip()
        if len(text) < 40:
            continue
        title = (topic.group(1).strip() if topic else (q.group(1).strip() if q else text[:60])) + " (ELVARN project notes)"
        store_document(kb, rag, title=title[:240], doc_type="reference", text=text, passages=[{"locator": "note", "text": text}],
                       publisher=NOTES_PUBLISHER, licence="Project's own material", cache=cache)
        done += 1
        if limit and done >= limit:
            break
    return done


def own_files(kb, rag, cache, limit=None):
    done = 0
    for name in OWN_FILES:
        path = DATA / "raw_v2" / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        parts = chunk(text)
        if not parts:
            continue
        store_document(kb, rag, title=f"{path.stem.replace('_', ' ')} (project document)", doc_type="reference", text=text,
                       passages=[{"locator": f"chunk {i + 1}", "text": p} for i, p in enumerate(parts)],
                       publisher=FILES_PUBLISHER, licence="Owner's own material", cache=cache)
        done += 1
        if limit and done >= limit:
            break
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop each source after this many documents (0 = all)")
    ap.add_argument("--only", choices=["wikipedia", "notes", "files"], default=None)
    a = ap.parse_args()
    from backend.knowledge.kb import KnowledgeBase
    from backend.knowledge.vectorstore import RagStore
    from backend.usermemory.store import Memory
    kb, rag, mem = KnowledgeBase(), RagStore(), Memory.from_env()
    t0 = time.time()
    for name, fn in (("notes", notes), ("files", own_files), ("wikipedia", wikipedia)):
        if a.only and a.only != name:
            continue
        n = fn(kb, rag, None, a.limit or None)
        if mem is not None and n:
            mem.bump_version()                                   # once per source, not once per document
        print(f"{name}: {n} documents processed ({time.time() - t0:.0f}s)", flush=True)
    print(f"Chroma now holds {rag.count()} passages", flush=True)
    kb.close()


if __name__ == "__main__":
    sys.exit(main())
