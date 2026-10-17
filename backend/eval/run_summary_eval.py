"""Answer evaluation: how well does the MODEL read the retrieved notes? Same 100 questions (retrieval_bench_v2.json), old retrieval (vectors only) vs the new one.
Reports: the right sentence is among the notes it reads, refusals, summaries that pass the grounding check, and how often the summary contains the right sentence.

    python backend/eval/run_summary_eval.py        (needs the knowledge base and Chroma running)
"""
import json, os, re, sys, time
from types import SimpleNamespace
sys.path.insert(0, "."); sys.path.insert(0, "backend"); sys.path.insert(0, "backend/eval"); sys.path.insert(0, "backend/training")
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv(".env")
import torch
torch.set_num_threads(4)
from backend.knowledge.vectorstore import RagStore
from knowledge import graphrag, groundcheck
from search.prompt import KB_HEADER, build_evidence_text
from model.generation import generate
from chat_eval import load
from run_retrieval_bench import covers

model, tok = load("deploy/models/24m_legal_rag")[:2]
cases = json.load(open("backend/eval/retrieval_bench_v2.json", encoding="utf-8"))["cases"]
rag = RagStore.from_env()
count = lambda t: len(tok.encode(t, add_bos=False))


def run(mode):
    if mode == "old":
        rag.lexical = None; os.environ["ELVARN_EXACT_BOOST"] = "0"; os.environ["ELVARN_RERANK"] = "0"
    else:
        from backend.knowledge.lexical import Lexical
        rag.lexical = Lexical.from_env(); os.environ["ELVARN_EXACT_BOOST"] = "1"; os.environ["ELVARN_RERANK"] = "1"
    rows = []
    for c in cases:
        ctx = graphrag.build_context(None, rag, c["question"])
        items = [i for i in ctx.items if i["kind"] == "fact" or groundcheck.usable_note(i.get("text", ""))]
        srcs = [SimpleNamespace(snippet=i["text"], title=i["source"]["title"]) for i in items]
        text, used = build_evidence_text(c["question"], srcs, count, 512, 80, header=KB_HEADER)
        if not used:
            rows.append({"none": True}); continue
        notes = groundcheck.parse_notes(text)
        gold_visible = any(covers(c["gold"], n) for n in notes.values())
        out = generate(model, tok, text, mode="chat", max_new_tokens=80, greedy=True, repetition_penalty=1.15, no_repeat_ngram_size=4)["text"].strip()
        refused = bool(groundcheck.REFUSAL.match(out))
        _, grounding, _ = groundcheck.guard(out, text, c["question"], "the stored records")
        rows.append({"none": False, "gold_visible": gold_visible, "refused": refused, "grounded": (not refused) and grounding == "model",
                     "has_gold": (not refused) and covers(c["gold"], out, 0.6)})
    return rows


def report(name, rows):
    n = len(rows); shown = [r for r in rows if not r["none"]]
    vis = [r for r in shown if r["gold_visible"]]
    pct = lambda a, b: round(100 * a / b, 1) if b else 0.0
    print(f"{name}: {n} questions | records found for {len(shown)}")
    print(f"   the right sentence is among the notes the model reads: {len(vis)} of {len(shown)} ({pct(len(vis), len(shown))}%)")
    print(f"   model refused: {sum(r['refused'] for r in shown)} of {len(shown)} ({pct(sum(r['refused'] for r in shown), len(shown))}%)")
    print(f"   summary passes the grounding check (not a refusal): {sum(r['grounded'] for r in shown)} ({pct(sum(r['grounded'] for r in shown), len(shown))}%)")
    print(f"   when the right sentence IS in the notes: summary contains it: {sum(r['has_gold'] for r in vis)} of {len(vis)} ({pct(sum(r['has_gold'] for r in vis), len(vis))}%), refused: {sum(r['refused'] for r in vis)} of {len(vis)}")


for mode in ("old", "new"):
    t = time.time()
    rows = run(mode)
    report("OLD retrieval (vectors only)" if mode == "old" else "NEW retrieval (keyword + exact + reranker)", rows)
    print("   (%.0fs)\n" % (time.time() - t), flush=True)
