"""Score ELVARN models on the 40-question benchmark (greedy decoding, so results are repeatable).

Usage (project root):
    python backend/eval/run_benchmark.py <model_dir> [<model_dir> ...]
        [--raw-dir backend/data/raw --raw-dir backend/data/raw_v2]   # data to check which questions were seen in training
Each <model_dir> holds latest.json, the checkpoint and tokenizer.json (e.g. deploy/models/17m_rag).

For every question the reply must contain, for each keyword group, at least one of its keywords (whole word/phrase,
case-insensitive). Score = groups satisfied / groups. 'PASS' = every group satisfied. Questions that appear word for word
in the training data are marked "seen": a good score there can be memorisation. 'unseen' questions test what the
model really learned. Also reported: how repetitive the replies are (share of repeated 3-word sequences).
"""
import argparse
import json
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

BENCHMARK = Path(__file__).resolve().parent / "benchmark.json"


def load_benchmark(path=BENCHMARK):
    return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]


def keyword_found(keyword, text):
    return re.search(r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)", text.lower()) is not None


def score_answer(groups, answer):
    """Fraction of keyword groups that the answer satisfies (0.0 - 1.0)."""
    if not groups:
        raise ValueError("A question needs at least one keyword group.")
    hit = sum(any(keyword_found(k, answer) for k in group) for group in groups)
    return hit / len(groups)


def repetition(answer):
    """Share of 3-word sequences that are repeats (0 = none repeated, 1 = all the same). Short answers count as 0."""
    words = re.findall(r"\w+", answer.lower())
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if len(grams) < 3:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def normalise(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def seen_in_training(questions, raw_dirs):
    """Which benchmark questions appear word for word in the raw training files."""
    corpus = ""
    for d in raw_dirs:
        for f in sorted(Path(d).glob("*.txt")):
            corpus += " " + normalise(f.read_text(encoding="utf-8", errors="replace"))
    return {q["id"]: (" " + normalise(q["q"]) + " ") in (" " + corpus + " ") for q in questions}


def run(model_dir, questions, seen, max_new_tokens=60):
    from api.model_service import ModelService
    svc = ModelService(model_dir)
    info = svc.load()
    rows = []
    for q in questions:
        out = svc.generate(q["q"], mode=q["mode"], max_new_tokens=max_new_tokens, greedy=True)
        ans = out["text"]
        s = score_answer(q["groups"], ans)
        rows.append({"id": q["id"], "section": q["section"], "q": q["q"], "answer": ans, "score": s,
                     "pass": s == 1.0, "repetition": repetition(ans), "seen_in_training": bool(seen.get(q["id"]))})
    return info, rows


def summarise(rows):
    def pct(xs):
        return 100 * sum(xs) / len(xs) if xs else float("nan")
    out = {"overall_score_pct": pct([r["score"] for r in rows]), "passed": sum(r["pass"] for r in rows), "total": len(rows),
           "mean_repetition_pct": pct([r["repetition"] for r in rows])}
    for sec in ("domain", "english", "chat"):
        out[f"{sec}_score_pct"] = pct([r["score"] for r in rows if r["section"] == sec])
    out["seen_score_pct"] = pct([r["score"] for r in rows if r["seen_in_training"]])
    out["unseen_score_pct"] = pct([r["score"] for r in rows if not r["seen_in_training"]])
    out["n_seen"] = sum(r["seen_in_training"] for r in rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dirs", nargs="+", type=Path)
    ap.add_argument("--raw-dir", action="append", type=Path, default=None, help="raw data folder(s) used to detect seen questions")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--verbose", action="store_true", help="print every question and reply")
    ap.add_argument("--save", type=Path, default=None, help="write full results to this JSON file")
    args = ap.parse_args()
    questions = load_benchmark()
    raw_dirs = args.raw_dir or [BACKEND_DIR / "data" / "raw"]
    seen = seen_in_training(questions, raw_dirs)
    results = {}
    print(f"{'model':34}{'score':>8}{'passed':>9}{'domain':>8}{'english':>9}{'chat':>7}{'seen':>7}{'unseen':>8}{'repeat':>8}")
    for d in args.model_dirs:
        info, rows = run(d, questions, seen, args.max_new_tokens)
        s = summarise(rows)
        results[str(d)] = {"info": {k: info[k] for k in ("parameters", "vocab_size", "weights_from_step", "best_val_loss")}, "summary": s, "rows": rows}
        print(f"{str(d)[-33:]:34}{s['overall_score_pct']:>7.1f}%{s['passed']:>6}/{s['total']:<2}{s['domain_score_pct']:>7.1f}%{s['english_score_pct']:>8.1f}%{s['chat_score_pct']:>6.1f}%"
              f"{s['seen_score_pct']:>6.1f}%{s['unseen_score_pct']:>7.1f}%{s['mean_repetition_pct']:>7.1f}%")
        if args.verbose:
            for r in rows:
                print(f"   [{r['section'][:3]}{' seen' if r['seen_in_training'] else '     '}] {r['q']}\n        -> {r['answer'][:110]!r}  score {r['score']:.2f}")
    print(f"\n('seen' = {sum(seen.values())} of {len(questions)} questions appear word for word in the training data; 'repeat' = share of repeated 3-word sequences)")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
