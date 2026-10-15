"""Train ELVARN (BPE tokenizer, document-level validation split).

Usage (from the project root):
    python backend/training/train_batch.py                 # full run
    python backend/training/train_batch.py --smoke         # tiny test run
    python backend/training/train_batch.py --resume        # continue last.pt

The run is one job: it trains for --max-steps optimizer steps, evaluates on
the validation split every --eval-every steps, and keeps three files in
backend/training/checkpoints_bpe/:
    last.pt   newest state, used for --resume (status: in_progress)
    best.pt   lowest validation loss so far
    final.pt  written only when the run completes (best-validation weights)
    latest.json  points to final.pt; the chat server loads this file.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from model.transformer import PRESETS, build_model, count_parameters, preset_config  # noqa: E402
from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402

PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
CHECKPOINT_DIR = BACKEND_DIR / "training" / "checkpoints_bpe"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run (30 steps) written to a separate smoke directory")
    p.add_argument("--size", choices=list(PRESETS), default="tiny",
                   help="model size preset (tiny ~0.8M, 7m ~6.7M, 1b ~1.0 billion)")
    p.add_argument("--ram-gb", type=float, default=16.0,
                   help="RAM of this machine, used for the memory safety check")
    p.add_argument("--force", action="store_true",
                   help="start even if the memory check says the model will not fit")
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None, help="folder with train.bin / val.bin / meta.json (default: backend/data/processed)")
    p.add_argument("--tokenizer-file", type=Path, default=None, help="tokenizer.json matching --data-dir (default: the project tokenizer)")
    return p.parse_args()


def load_split(name, data_dir=None):
    path = Path(data_dir or PROCESSED_DIR) / f"{name}.bin"
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run build_dataset.py first.")
    return np.memmap(path, dtype=np.uint16, mode="r")  # not loaded into RAM


def get_batch(data, batch_size, seq_len, rng):
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[s:s + seq_len] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + seq_len + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def lm_loss(logits, targets, pad_id):
    # Each position predicts the next token; PAD targets are ignored.
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id
    )


@torch.no_grad()
def evaluate(model, data, seq_len, batch_size, pad_id):
    """Validation loss over the whole split, in non-overlapping windows."""
    model.eval()
    device = next(model.parameters()).device
    n_windows = (len(data) - 1) // seq_len
    if n_windows < 1:
        raise SystemExit("Validation split is shorter than one sequence.")
    total, count = 0.0, 0
    for i in range(0, n_windows, batch_size):
        idx = range(i, min(i + batch_size, n_windows))
        x = torch.from_numpy(np.stack([data[j * seq_len:j * seq_len + seq_len] for j in idx]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([data[j * seq_len + 1:j * seq_len + seq_len + 1] for j in idx]).astype(np.int64)).to(device)
        total += lm_loss(model(x), y, pad_id).item() * len(idx)
        count += len(idx)
    model.train()
    return total / count


def lr_at(step, args):
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return 0.1 * args.learning_rate + 0.9 * args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, model, optimizer, step, best_val, config, tok, history, rng_state, status):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val,
        "config": config,
        "vocab_size": tok.vocab_size,
        "tokenizer_sha256": tok.sha256,
        "history": history,
        "rng_state": rng_state,
        "status": status,
    }, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint


def main():
    args = parse_args()
    if args.smoke:
        args.max_steps, args.eval_every, args.warmup_steps, args.batch_size = 30, 15, 5, 8
    size_dir = "" if args.size == "tiny" else args.size
    ckpt_dir = args.checkpoint_dir or (CHECKPOINT_DIR / ("smoke" if args.smoke else size_dir))

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")

    tok = BPETokenizer(args.tokenizer_file) if args.tokenizer_file else BPETokenizer()
    meta_path = Path(args.data_dir or PROCESSED_DIR) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if meta.get("tokenizer_sha256") not in (None, tok.sha256):
        raise SystemExit("Processed data was built with a different tokenizer. Re-run build_dataset.py.")

    train_data, val_data = load_split("train", args.data_dir), load_split("val", args.data_dir)
    config = preset_config(args.size)
    seq_len = config["sequence_length"]
    if len(train_data) <= seq_len + 1 or len(val_data) <= seq_len + 1:
        raise SystemExit("Dataset too small for the configured sequence length.")
    for name, data in (("train", train_data), ("val", val_data)):
        if int(data.max()) >= tok.vocab_size:
            raise SystemExit(f"{name}.bin contains token ids outside the vocabulary.")

    # Memory check BEFORE allocating: weights + gradients + AdamW (2 states)
    # is about 16 bytes per parameter, plus activations and the OS.
    with torch.device("meta"):
        n_params = count_parameters(build_model(tok.vocab_size, config))
    need_gb = n_params * 16 / 1e9
    if need_gb > args.ram_gb * 0.75 and not args.force:
        raise SystemExit(
            f"Model '{args.size}' has {n_params:,} parameters and needs about "
            f"{need_gb:.1f} GB for training state, but only {args.ram_gb:.0f} GB RAM "
            f"is available (limit {args.ram_gb * 0.75:.0f} GB). Not starting, to avoid "
            "freezing the machine. Use a smaller --size, a bigger machine, "
            "or --ram-gb / --force if you know better."
        )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Each run keeps its own copy of the tokenizer, so a checkpoint can never
    # be loaded with a different vocabulary later.
    shutil.copyfile(tok.path, ckpt_dir / "tokenizer.json")
    model = build_model(tok.vocab_size, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    step, best_val, history = 0, float("inf"), []
    last_path, best_path, final_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt", ckpt_dir / "final.pt"
    if args.resume:
        if not last_path.is_file():
            raise SystemExit(f"Nothing to resume: {last_path} not found.")
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck["tokenizer_sha256"] != tok.sha256 or ck["vocab_size"] != tok.vocab_size:
            raise SystemExit("Checkpoint was trained with a different tokenizer. Cannot resume.")
        if ck["config"] != config:
            raise SystemExit("Checkpoint model config differs from the current config. Cannot resume.")
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        step, best_val, history = ck["step"], ck["best_val_loss"], ck["history"]
        rng.bit_generator.state = ck["rng_state"]
        print(f"Resumed from step {step}")

    print("=" * 60)
    print("ELVARN TRAINING")
    print("=" * 60)
    print("Dataset files      :", meta.get("files"))
    print("Train documents    :", meta.get("train", {}).get("documents"))
    print("Val documents      :", meta.get("val", {}).get("documents"))
    print("Train tokens       :", len(train_data))
    print("Val tokens         :", len(val_data))
    print("Unknown tokens     : train", meta.get("train", {}).get("unk_percent"), "% | val", meta.get("val", {}).get("unk_percent"), "%")
    print("Vocabulary size    :", tok.vocab_size)
    print("Sequence length    :", seq_len)
    print("Model config       :", config)
    print(f"Parameters         : {count_parameters(model):,}")
    print("Batch size         :", args.batch_size)
    print("Learning rate      :", args.learning_rate)
    print("Max steps          :", args.max_steps)
    print("Device             :", device)
    print("Checkpoint dir     :", ckpt_dir)
    print("=" * 60, flush=True)

    model.train()
    started = time.time()
    while step < args.max_steps:
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        x, y = get_batch(train_data, args.batch_size, seq_len, rng)
        loss = lm_loss(model(x.to(device)), y.to(device), tok.pad_id)
        if not torch.isfinite(loss):
            raise SystemExit(f"Loss became {loss.item()} at step {step}. Stopping.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % args.eval_every == 0 or step == args.max_steps:
            val_loss = evaluate(model, val_data, seq_len, args.batch_size, tok.pad_id)
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss})
            print(f"step {step:5d} | train {loss.item():.4f} | val {val_loss:.4f} | {time.time() - started:.0f}s", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best_path, model, optimizer, step, best_val, config, tok, history, rng.bit_generator.state, "best")
            save_checkpoint(last_path, model, optimizer, step, best_val, config, tok, history, rng.bit_generator.state, "in_progress")

    # Completed: final.pt holds the best-validation weights.
    best = torch.load(best_path, map_location=device, weights_only=False)
    best["status"] = "completed"
    best["history"] = history
    best["final_step"] = step
    tmp = final_path.with_suffix(".tmp")
    torch.save(best, tmp)
    os.replace(tmp, final_path)
    save_checkpoint(last_path, model, optimizer, step, best_val, config, tok, history, rng.bit_generator.state, "completed")

    (ckpt_dir / "latest.json").write_text(json.dumps({
        "checkpoint": final_path.name,
        "status": "completed",
        "weights_from_step": best["step"],
        "final_step": step,
        "best_val_loss": best_val,
        "tokenizer_sha256": tok.sha256,
        "vocab_size": tok.vocab_size,
        "config": config,
        "smoke_run": bool(args.smoke),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"TRAINING COMPLETE. Final weights (step {best['step']}, val {best_val:.4f}) -> {final_path}")


if __name__ == "__main__":
    main()
