"""Score how well a model answers FROM web notes (the live web-search prompt format).

Usage (project root):
    python backend/eval/run_rag_eval.py <checkpoint folder or .pt file> [<more> ...] [--save backend/eval/results/NAME.json]
A folder needs tokenizer.json next to the checkpoint; the checkpoint is latest.json's file, else best.pt / final.pt / the
newest *.pt. Each case in backend/eval/rag_eval.json (made from VALIDATION topics only) is: notes + question (+ key facts,
or "not in the notes"). The prompt is built exactly like the live server does (search.prompt HEADER/FOOTER, mode template).

Scores (simple counts, no language-model judge):
  * key_fact_coverage  share of key-fact keywords (from the notes that hold the answer) found in the answer.
  * refusal_correct    on "not in the notes" cases: the answer is an honest refusal. false_refusal: on answerable
                       cases the model refused although the notes had the answer.
  * repetition         share of repeated 3-word sequences (same measure as run_benchmark.py).
  * invented_rate      share of answers containing a number or a Capitalised word that is in neither the notes nor the
                       question (rough hallucination signal; a correct paraphrase can trigger it a little).
  * citation_rate      share of answerable answers that cite a note like [1] with a number that exists.
Greedy decoding with the repetition controls (repetition_penalty 1.15, no_repeat_ngram_size 4 by default), 2 threads.
Results go to a NEW file under backend/eval/results/ (an existing file is never overwritten).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_benchmark import keyword_found, repetition  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "rag_eval.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
REFUSAL_RE = re.compile(r"(?:do(?:es)? not|don't|doesn't|cannot|can't)\s+(?:say|contain|answer|find|mention|know|tell|help)|"
                        r"\bnot\s+(?:mentioned|stated|in the notes|given)|\bnotes (?:do not|don't)\b", re.I)
COMMON = {"according", "from", "the", "notes", "note", "i", "in", "on", "it", "this", "that", "there", "these", "those", "he", "she",
          "they", "his", "her", "its", "a", "an", "and", "but", "for", "as", "at", "by", "of", "to", "with", "not", "no", "yes",
          "my", "question", "answer", "what", "who", "when", "where", "how", "which", "is", "was", "are", "were", "also", "however",
          "after", "before", "during", "while", "if", "so", "then", "one", "two", "some", "many", "most", "other", "both", "each"}


def load_cases(path=CASES_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]


def is_refusal(answer):
    return REFUSAL_RE.search(answer) is not None


def _words(text):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'.,-]*[A-Za-z0-9]|[A-Za-z0-9]", text)


def invented_tokens(answer, notes, question):
    """Numbers and Capitalised words (not at a sentence start) in the answer found in neither the notes nor the question."""
    known = {w.lower().strip(".,") for w in _words(" ".join(notes) + " " + question)}
    bad = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        for i, w in enumerate(_words(sentence)):
            core = w.strip(".,'-")
            if not core or core.lower() in known or core.lower() in COMMON or re.fullmatch(r"\d", core):
                continue                                           # single digits are mostly citation markers like [1]
            if core[0].isdigit() or (core[0].isupper() and i > 0):
                bad.append(core)
    return bad


def citations_ok(answer, n_notes):
    found = re.findall(r"\[(\d+)\]", answer)
    return bool(found) and all(1 <= int(m) <= n_notes for m in found)


def score_case(case, answer):
    answer = answer.strip()
    refused = is_refusal(answer)
    invented = invented_tokens(answer, case["notes"], case["question"])
    row = {"id": case["id"], "type": case["type"], "answerable": case["answerable"], "answer": answer,
           "refused": refused, "repetition": repetition(answer), "invented": invented, "invented_any": bool(invented)}
    if case["answerable"]:
        groups = case["groups"]
        row["key_fact_coverage"] = sum(any(keyword_found(k, answer) for k in g) for g in groups) / len(groups) if groups else None
        row["cites_a_note"] = citations_ok(answer, len(case["notes"]))
    else:
        row["refusal_correct"] = refused
        row["invented_any"] = bool(invented) and not refused
    return row


def summarise(rows):
    def pct(xs):
        xs = [x for x in xs if x is not None]
        return round(100 * sum(xs) / len(xs), 1) if xs else None
    ans = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    return {
        "cases": len(rows), "answerable_cases": len(ans), "not_in_notes_cases": len(neg),
        "key_fact_coverage_pct": pct([r["key_fact_coverage"] for r in ans]),
        "full_coverage_pct": pct([r["key_fact_coverage"] == 1.0 for r in ans if r["key_fact_coverage"] is not None]),
        "refusal_correct_pct": pct([r["refusal_correct"] for r in neg]),
        "false_refusal_pct": pct([r["refused"] for r in ans]),
        "mean_repetition_pct": pct([r["repetition"] for r in rows]),
        "invented_rate_pct": pct([r["invented_any"] for r in rows]),
        "citation_rate_pct": pct([r["cites_a_note"] for r in ans]),
    }


def run_cases(cases, generate_fn):
    """generate_fn(case) -> answer text. Separate from the model so the scoring can be tested with a fake generator."""
    return [score_case(c, generate_fn(c)) for c in cases]


# ----------------------------------------------------------------- real model
def find_checkpoint(path):
    path = Path(path)
    if path.is_file():
        return path
    latest = path / "latest.json"
    if latest.is_file():
        name = json.loads(latest.read_text(encoding="utf-8")).get("checkpoint")
        if name and (path / name).is_file():
            return path / name
    for name in ("best.pt", "final.pt"):
        if (path / name).is_file():
            return path / name
    found = sorted(path.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if found:
        return found[-1]
    raise SystemExit(f"No checkpoint (*.pt) found in {path}.")


def load_model(path):
    import torch
    from model.transformer import model_from_checkpoint
    from tokenizer.bpe_tokenizer import BPETokenizer
    ckpt = find_checkpoint(path)
    tok_file = ckpt.parent / "tokenizer.json"
    if not tok_file.is_file():
        raise SystemExit(f"No tokenizer.json next to {ckpt}.")
    tok = BPETokenizer(tok_file)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck.get("tokenizer_sha256") != tok.sha256:
        raise SystemExit("The checkpoint does not match the tokenizer saved next to it.")
    model, _ = model_from_checkpoint(ck)
    model.eval()
    return model, tok, ckpt


def case_body(case):
    from search.prompt import FOOTER, HEADER
    lines = [f"[{i + 1}] {n}" for i, n in enumerate(case["notes"])]
    return f"{HEADER}\n" + "\n".join(lines) + f"\n{FOOTER} {case['question']}"


def evaluate_model(path, cases, max_new_tokens=80, repetition_penalty=1.15, no_repeat_ngram_size=4):
    import torch
    from model.generation import build_prompt, generate
    torch.set_num_threads(2)
    model, tok, ckpt = load_model(path)
    stats = {"context_too_short": 0}

    def gen(case):
        body = case_body(case)
        if len(tok.encode(build_prompt(body, case["mode"]), add_bos=True)) > model.sequence_length - 1:
            stats["context_too_short"] += 1              # the start of the prompt (the notes) gets cut off
        return generate(model, tok, body, mode=case["mode"], max_new_tokens=max_new_tokens, greedy=True,
                        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size)["text"]

    rows = run_cases(cases, gen)
    info = {"checkpoint": str(ckpt), "sequence_length": model.sequence_length, "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty, "no_repeat_ngram_size": no_repeat_ngram_size, **stats}
    return info, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path, help="checkpoint folder(s) or .pt file(s) with tokenizer.json next to them")
    ap.add_argument("--cases", type=Path, default=CASES_FILE)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=4)
    ap.add_argument("--save", type=Path, default=None, help="NEW json file (default: backend/eval/results/rag_<timestamp>.json)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    save = args.save or RESULTS_DIR / time.strftime("rag_%Y%m%d_%H%M%S.json")
    if save.exists():
        raise SystemExit(f"{save} already exists; refusing to overwrite.")
    cases = load_cases(args.cases)
    results = {}
    fmt = lambda v: "    n/a" if v is None else f"{v:>6.1f}%"
    print(f"{'model':36}{'facts':>7}{'full':>7}{'refuse':>7}{'falseref':>9}{'repeat':>7}{'invent':>7}{'cite':>7}")
    for m in args.models:
        info, rows = evaluate_model(m, cases, args.max_new_tokens, args.repetition_penalty, args.no_repeat_ngram_size)
        s = summarise(rows)
        results[str(m)] = {"info": info, "summary": s, "rows": rows}
        print(f"{str(m)[-35:]:36}" + "".join(fmt(s[k]) for k in (
            "key_fact_coverage_pct", "full_coverage_pct", "refusal_correct_pct", "false_refusal_pct",
            "mean_repetition_pct", "invented_rate_pct", "citation_rate_pct")))
        if info["context_too_short"]:
            print(f"   note: {info['context_too_short']} of {len(cases)} prompts are longer than the model's context "
                  f"({info['sequence_length']}); their notes were cut off. Extend the context first.")
        if args.verbose:
            for r in rows:
                print(f"   [{r['id']:>2} {r['type'][:8]:8}] {r['answer'][:110]!r}")
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {save}")
    return results


if __name__ == "__main__":
    main()
