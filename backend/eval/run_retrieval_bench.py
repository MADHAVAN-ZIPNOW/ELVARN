"""Score retrieval on the fixed benchmark (backend/eval/retrieval_bench.json): does the correct evidence reach the top 1 / 3 / 5?

    python backend/eval/run_retrieval_bench.py [--save backend/eval/results/NAME.json] [--label NAME]

For every question the real retrieval (graphrag.build_context on your Chroma + knowledge graph) returns its ranked items. Two checks:
  strict    an item among the top k contains the gold sentence (at least 75 % of its words);
  relevant  an item among the top k names the exact section AND most words of the Act (section questions), or contains the topic words (reference).
Also reported: nothing_found (no item at all) and MRR of the first strict hit. Results go to a NEW file (an existing file is never overwritten).
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR.parent))
sys.stdout.reconfigure(encoding="utf-8")

BENCH = HERE / "retrieval_bench.json"
STOP = {"the", "and", "for", "that", "with", "this", "from", "have", "has", "was", "were", "which", "shall", "their", "there", "not", "are"}


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP}


def covers(gold, text, share=0.75):
    g = words(gold)
    return bool(g) and len(g & words(text)) / len(g) >= share


def names_section(case, text):
    flat = re.sub(r"\s+", "", text.lower())
    num = re.sub(r"\s+", "", case["section"].lower())
    if not re.search(r"(?<![\d(])" + re.escape(num) + r"(?![\d(a-z])", flat):
        return False
    act = {w for w in re.findall(r"[a-z]{4,}", case["act"].lower()) if w not in ("act", "code", "india", "constitution")}
    return len(act & {w for w in re.findall(r"[a-z]{4,}", text.lower())}) >= math.ceil(0.5 * max(1, len(act)))


def relevant(case, text):
    if case["kind"] == "section":
        return names_section(case, text)
    topic = {w for w in re.findall(r"[a-z]{4,}", case["topic"].lower())}
    return bool(topic) and topic <= {w for w in re.findall(r"[a-z]{4,}", text.lower())}


def run(retrieve, cases, ks=(1, 3, 5)):
    """retrieve(question) -> ranked list of item texts. Returns {metrics, rows}."""
    rows = []
    for c in cases:
        texts = retrieve(c["question"])
        first_strict = next((i for i, t in enumerate(texts, 1) if covers(c["gold"], t)), None)
        first_rel = next((i for i, t in enumerate(texts, 1) if relevant(c, t)), None)
        rows.append({"id": c["id"], "kind": c["kind"], "n_items": len(texts), "first_strict": first_strict, "first_relevant": first_rel})
    out = {}
    for kind in ("all", "section", "reference"):
        part = [r for r in rows if kind == "all" or r["kind"] == kind]
        m = {"cases": len(part), "nothing_found_pct": round(100 * sum(r["n_items"] == 0 for r in part) / len(part), 1),
             "mrr_strict": round(sum(1 / r["first_strict"] for r in part if r["first_strict"]) / len(part), 3)}
        for k in ks:
            m[f"strict@{k}"] = round(100 * sum(1 for r in part if r["first_strict"] and r["first_strict"] <= k) / len(part), 1)
            m[f"relevant@{k}"] = round(100 * sum(1 for r in part if r["first_relevant"] and r["first_relevant"] <= k) / len(part), 1)
        out[kind] = m
    return {"metrics": out, "rows": rows}


def real_retrieve(rag, kb=None):
    from knowledge import graphrag, groundcheck

    def retrieve(question):
        ctx = graphrag.build_context(kb, rag, question)
        return [i["text"] for i in ctx.items if i["kind"] == "fact" or groundcheck.usable_note(i.get("text", ""))]
    return retrieve


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="retrieval")
    ap.add_argument("--bench", type=Path, default=BENCH)
    ap.add_argument("--mode", choices=["dense", "rrf", "rrf_exact", "rerank", "rerank_exact"], default="rerank_exact", help="dense = vectors only; rrf = + BM25; rrf_exact = + exact section boost; rerank = BM25 fusion + cross-encoder; rerank_exact = all of it")
    ap.add_argument("--save", type=Path, default=None)
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR.parent / ".env")
    from knowledge.vectorstore import RagStore
    rag = RagStore.from_env()
    if a.mode == "dense":
        rag.lexical = None
    os.environ["ELVARN_EXACT_BOOST"] = "1" if a.mode in ("rrf_exact", "rerank_exact") else "0"
    os.environ["ELVARN_RERANK"] = "1" if a.mode.startswith("rerank") else "0"
    cases = json.loads(a.bench.read_text(encoding="utf-8"))["cases"]
    t = time.time()
    res = run(real_retrieve(rag), cases)
    res["label"], res["seconds"] = a.label, round(time.time() - t, 1)
    print(f"[{a.label}] {len(cases)} questions in {res['seconds']}s")
    for kind, m in res["metrics"].items():
        print(f"  {kind:9s} strict@1/3/5 = {m['strict@1']}/{m['strict@3']}/{m['strict@5']}%   relevant@1/3/5 = {m['relevant@1']}/{m['relevant@3']}/{m['relevant@5']}%   "
              f"MRR {m['mrr_strict']}   nothing found {m['nothing_found_pct']}%")
    if a.save:
        if a.save.exists():
            raise SystemExit(f"{a.save} already exists; refusing to overwrite.")
        a.save.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print("saved", a.save)


if __name__ == "__main__":
    main()
