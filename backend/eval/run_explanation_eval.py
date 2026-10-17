"""Benchmark of the grounded explanation layer on the fixed 100-question benchmark (backend/eval/retrieval_bench_v2.json).

For every question: the real retrieval (records + the quotes the cross-encoder chose), then the explanation layer (api/explain.py) with the 24M model,
for each prompt style. Reported (retrieval metrics are NOT changed: they are in run_retrieval_bench.py):
  records-only baseline   the right sentence is among the records shown (this is what users get without an explanation)
  explanation generated   the model produced text that parsed
  passed validation       status ANSWERED or PARTIAL (every citation valid, every claim supported, no refusal)
  every citation valid    of the generated ones, the share that passed (citations are validated, so a pass means all were valid)
  contains the right fact the validated explanation contains the expected sentence (60 % of its words)
  wrong refusals          the model refused although a quote it was given contains the right sentence
  rejected, by reason     why explanations were dropped
Compare "records only" with "records + explanation": the explanation is shown only when it passed validation, and the records never change.

    python backend/eval/run_explanation_eval.py [--save backend/eval/results/NAME.json]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
for p in (str(BACKEND_DIR.parent), str(BACKEND_DIR), str(HERE), str(BACKEND_DIR / "training")):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding="utf-8")

BENCH = HERE / "retrieval_bench_v2.json"
STYLES = ("instructions", "native")


class Service:
    """The model as api.explain expects it: .ready and .generate(text, **kw)."""
    ready = True

    def __init__(self, model_dir):
        import torch
        from chat_eval import load
        torch.set_num_threads(4)
        self.model, self.tok = load(model_dir)[:2]

    def generate(self, text, **kw):
        from model.generation import generate
        return generate(self.model, self.tok, text, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deploy/models/24m_legal_rag")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR.parent / ".env")
    os.environ["ELVARN_EXPLAIN"] = "1"
    from api import explain
    from knowledge import graphrag, groundcheck
    from knowledge.vectorstore import RagStore
    from run_retrieval_bench import covers
    rag = RagStore.from_env()
    service = Service(a.model)
    cases = json.loads(BENCH.read_text(encoding="utf-8"))["cases"]
    if a.limit:
        cases = cases[:a.limit]
    rows = {s: [] for s in STYLES}
    records_hit = 0
    started = time.time()
    for c in cases:
        ctx = graphrag.build_context(None, rag, c["question"])
        items = [i for i in ctx.items if i["kind"] == "fact" or groundcheck.usable_note(i.get("text", ""))]
        records = [{"n": n, "text": i["text"].strip()} for n, i in enumerate(items, 1)]
        hit = any(covers(c["gold"], r["text"]) for r in records)
        records_hit += hit
        number = {r["text"]: r["n"] for r in records}
        quotes = None if ctx.quotes is None else [{"record": number[q["item_text"].strip()], "quote": q["quote"], "score": q["score"]}
                                                  for q in ctx.quotes if q["item_text"].strip() in number]
        for style in STYLES:
            out = explain.explain(service, c["question"], records, quotes, style=style)
            given = {q["record"]: q["quote"] for q in explain.choose_quotes(quotes or [])}
            exp = out["explanation"]
            rows[style].append({"records_hit": hit, "status": out["status"], "reason": out["reason"],
                                "has_fact": bool(exp) and covers(c["gold"], exp["answer"], 0.6),
                                "gold_in_given": any(covers(c["gold"], t) for t in given.values()),
                                "wrong_refusal": out["reason"].startswith("refusal") and any(covers(c["gold"], t) for t in given.values())})
    n = len(cases)
    pct = lambda x, d=n: round(100 * x / d, 1) if d else 0.0
    print(f"{n} questions, {round(time.time() - started)}s | records-only baseline: the right sentence is among the records for {records_hit} ({pct(records_hit)}%)")
    result = {"questions": n, "records_only_hit": records_hit, "styles": {}}
    for style, rs in rows.items():
        shown = [r for r in rs if r["status"] in ("ANSWERED", "PARTIAL")]
        invalid = [r for r in rs if r["status"] == "INVALID"]
        generated = len(shown) + len(invalid)
        reasons = Counter(r["reason"].split(":")[0] for r in invalid)
        m = {"generated": generated, "passed_validation": len(shown), "answered": sum(r["status"] == "ANSWERED" for r in shown),
             "partial": sum(r["status"] == "PARTIAL" for r in shown), "not_enough_evidence_no_quote": sum(r["status"] == "NOT_ENOUGH_EVIDENCE" for r in rs),
             "disabled": sum(r["status"] == "DISABLED" for r in rs), "contains_right_fact": sum(r["has_fact"] for r in rs),
             "wrong_refusals": sum(r["wrong_refusal"] for r in rs), "right_sentence_among_quotes_given": sum(r["gold_in_given"] for r in rs),
             "rejected_by_reason": dict(reasons)}
        result["styles"][style] = m
        print(f"\nstyle '{style}':")
        print(f"   generated {generated} | passed validation {len(shown)} ({pct(len(shown))}%): answered {m['answered']}, partial {m['partial']} | "
              f"every citation valid: {len(shown)} of {generated} generated ({pct(len(shown), generated)}%)")
        print(f"   explanation contains the right fact: {m['contains_right_fact']} ({pct(m['contains_right_fact'])}%) | right sentence was among the quotes given to the model: "
              f"{m['right_sentence_among_quotes_given']} | wrong refusals: {m['wrong_refusals']}")
        print(f"   no relevant quote (model not asked): {m['not_enough_evidence_no_quote']} | rejected by reason: {dict(reasons)}")
    if a.save:
        if a.save.exists():
            raise SystemExit(f"{a.save} already exists; refusing to overwrite.")
        a.save.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("\nsaved", a.save)


if __name__ == "__main__":
    main()
