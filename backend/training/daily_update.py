"""Keep the model up to date with the knowledge base - safely.

    1. export the knowledge base to training text (backend/knowledge/export_training.py)
    2. skip everything if there is not enough NEW material since the last training (default 200 new blocks)
    3. build RAG fine-tuning data in the live prompt format (build_rag_dataset.py, into a NEW dated folder)
    4. fine-tune a CANDIDATE from the live model (continue_train.py, into a NEW dated folder)
    5. evaluate live and candidate on the SAME fixed hold-out (backend/eval/rag_eval.json, 60 cases) with run_rag_eval.py
    6. gate: the candidate must be no worse than the live model on every measure (see gate())
    7. write a report; ONLY with --promote and a passing gate, export a new deploy folder next to the live one.

The live model is never overwritten and the server is never switched automatically: to use the new folder, point
ELVARN_CHECKPOINT_DIR at it. Everything lands in new, dated folders, so any step can be inspected or deleted.

    python backend/training/daily_update.py                 # export + (if enough new data) train + evaluate + report
    python backend/training/daily_update.py --promote       # also create deploy/models/17m_rag_kb_<date> if the gate passes
    python backend/training/daily_update.py --force         # train even if little new data arrived
    python backend/training/daily_update.py --dry-run       # export and decide, but do not build or train anything
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

STATE = BACKEND_DIR / "data" / "kb_state" / "training.json"
LIVE = PROJECT_DIR / "deploy" / "models" / "17m_rag"
LIVE_TOKENIZER_SOURCE = BACKEND_DIR / "data" / "v2" / "tokenizer" / "tokenizer.json"
MIN_NEW_BLOCKS = 200

# measure -> (direction, allowed slack in percentage points). "up": higher is better; "down": lower is better.
GATE = {
    "key_fact_coverage_pct": ("up", 1.0),      # facts copied from the notes
    "citation_rate_pct": ("up", 5.0),          # cites a real note
    "refusal_correct_pct": ("up", 5.0),        # says "not in the notes" when it is not
    "invented_rate_pct": ("down", 2.0),        # invented names / numbers
    "false_refusal_pct": ("down", 5.0),        # refuses although the notes had the answer
    "mean_repetition_pct": ("down", 2.0),
}


def blocks_in(info):
    info = info or {}
    return info.get("passage_blocks", 0) + info.get("reference_blocks", 0) + info.get("fact_blocks", 0)


def decide(state, export_info, min_new=MIN_NEW_BLOCKS, force=False):
    """(train?, reason). New material = blocks exported now minus blocks present at the last training."""
    new = blocks_in(export_info) - blocks_in((state.get("last_training") or {}).get("export"))
    if blocks_in(export_info) == 0:
        return False, "nothing to train on: the knowledge base has no official documents yet"
    if force:
        return True, f"forced ({new} new blocks)"
    if new < min_new:
        return False, f"only {new} new blocks since the last training (need {min_new})"
    return True, f"{new} new blocks since the last training"


def gate(live, candidate):
    """(passed, reasons). The candidate must be no worse than the live model on every measure, within a small slack."""
    reasons = []
    for key, (direction, slack) in GATE.items():
        a, b = live[key], candidate[key]
        worse = (b < a - slack) if direction == "up" else (b > a + slack)
        if worse:
            reasons.append(f"{key}: live {a} -> candidate {b} ({'lower' if direction == 'up' else 'higher'} is worse; slack {slack})")
    return (not reasons), reasons


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd, log):
    log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    log.flush()
    result = subprocess.run([str(c) for c in cmd], cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"step failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-new-blocks", type=int, default=MIN_NEW_BLOCKS)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--live", type=Path, default=LIVE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--promote", action="store_true", help="create a new deploy folder if the gate passes (never replaces the live one)")
    a = ap.parse_args(argv)

    from backend.knowledge.export_training import export
    from backend.knowledge.kb import KnowledgeBase
    kb = KnowledgeBase()
    try:
        info = export(kb)
    finally:
        kb.close()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    train, reason = decide(state, info, a.min_new_blocks, a.force)
    print(f"export: {info['passage_blocks']} official-document blocks, {info['reference_blocks']} new-Wikipedia blocks, {info['fact_blocks']} fact blocks -> {'TRAIN' if train else 'skip'}: {reason}")
    if not train or a.dry_run:
        return 0

    if sha256(a.live / "tokenizer.json") != sha256(LIVE_TOKENIZER_SOURCE):
        print("ABORT: the live model's tokenizer differs from the tokenizer the RAG builder uses; a candidate would not match it.")
        return 2
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    rag_dir = BACKEND_DIR / "data" / f"rag_kb_{stamp}"
    cand_dir = BACKEND_DIR / "training" / "checkpoints_bpe" / f"kb_{stamp}"
    results = BACKEND_DIR / "eval" / "results" / f"daily_{stamp}.json"
    log_path = BACKEND_DIR / "data" / "train_logs" / f"daily_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        run([py, BACKEND_DIR / "data_tools" / "build_rag_dataset.py", "--out-dir", rag_dir, "--seed", 1], log)
        run([py, BACKEND_DIR / "training" / "continue_train.py", "--init-from", a.live / "best.pt", "--data-dir", rag_dir, "--out-dir", cand_dir,
             "--epochs", a.epochs, "--batch-size", 8, "--optimizer", "fresh", "--learning-rate", "5e-5", "--warmup-steps", 50,
             "--patience", 2, "--seed", 42, "--threads", a.threads], log)
        run([py, BACKEND_DIR / "eval" / "run_rag_eval.py", a.live, cand_dir, "--save", results], log)
    scores = list(json.loads(results.read_text(encoding="utf-8")).values())          # same order as the models on the command line
    live_s, cand_s = scores[0]["summary"], scores[1]["summary"]
    passed, reasons = gate(live_s, cand_s)
    report = {"stamp": stamp, "reason": reason, "minutes": round((time.time() - t0) / 60, 1), "export": info, "live": live_s, "candidate": cand_s,
              "gate_passed": passed, "gate_failures": reasons, "candidate_dir": str(cand_dir), "rag_data": str(rag_dir), "log": str(log_path),
              "promoted_to": None}
    if passed and a.promote:
        target = PROJECT_DIR / "deploy" / "models" / f"17m_rag_kb_{stamp}"
        run([py, BACKEND_DIR / "training" / "export_deploy.py", cand_dir, target], open(log_path, "a", encoding="utf-8"))
        report["promoted_to"] = str(target)
    state["last_training"] = {"at": stamp, "export": info, "gate_passed": passed, "candidate": str(cand_dir)}
    state["last_export"] = info
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    (BACKEND_DIR / "data" / "train_logs" / f"daily_{stamp}.report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("stamp", "gate_passed", "gate_failures", "promoted_to", "candidate_dir")}, indent=2))
    print("live:", live_s)
    print("cand:", cand_s)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
