"""Continue training a saved ELVARN checkpoint for N TRUE epochs.

Unlike train_batch.py (which draws random windows for a fixed number of steps), this script walks the
training data epoch by epoch:

  * the training text is one stream of N tokens. Token i is trained to predict token i+1, so there are
    N-1 next-token pairs. A window (start s, length L <= sequence_length) has inputs data[s:s+L] and targets
    data[s+1:s+L+1] (the same text shifted left by one), and covers the L pairs s .. s+L-1;
  * at the start of every epoch a random offset o is drawn and the stream is cut into: a short HEAD window
    (pairs 0..o-1), full windows of sequence_length, and a short TAIL window for the leftover pairs.
    Together they cover ALL N-1 pairs EXACTLY ONCE per epoch (none skipped, none duplicated, none favoured);
  * short windows are padded with PAD; padded targets are ignored by the loss;
  * the windows are SHUFFLED every epoch and grouped into batches of --batch-size (the last, smaller batch is
    kept);
  * ONE optimizer update per batch. There is NO gradient accumulation.

Optimizer / learning rate (--optimizer):
  fresh    a brand-new AdamW; learning rate --learning-rate with warm-up and a cosine decay over this run.
  restore  a TRUE CONTINUATION of the run that produced the checkpoint: the saved AdamW state is restored
           and the ORIGINAL step-based schedule (--original-lr / --original-warmup / --original-total-steps)
           carries on from the checkpoint's step. Refused if the checkpoint has no optimizer state or its
           step counter does not match the optimizer's.

After EVERY epoch a resumable checkpoint epoch_<n>.pt is written (model + optimizer + epoch + step +
configuration + tokenizer fingerprint + random-number states + loss history). The last epoch also writes
final.pt. --resume continues an interrupted run from its newest epoch_<n>.pt and gives the same result as an
uninterrupted run. Early stopping is optional: --patience 0 switches it off.

Safety: the checkpoint's architecture must load strictly and its sequence length, vocabulary and tokenizer
hash must match the tokenizer saved with it AND the dataset in data/processed. The validation split is never
trained on. Nothing already on disk is modified: output goes to a NEW folder.

Usage (project root):
    python backend/training/continue_train.py --init-from backend/training/checkpoints_bpe/4m/best.pt \\
        --out-dir backend/training/checkpoints_bpe/my_run --epochs 5 --patience 0 --optimizer restore
    python backend/training/continue_train.py ... same arguments ... --resume      # after an interruption
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "training"))

from model.transformer import count_parameters, model_from_checkpoint  # noqa: E402
from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402
from train_batch import evaluate, lm_loss  # noqa: E402  (same loss and validation as the original run)
from train_batch import lr_at as original_lr_at  # noqa: E402  (the ORIGINAL run's learning-rate schedule)

PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
DEFAULT_INIT = BACKEND_DIR / "training" / "checkpoints_bpe" / "4m" / "best.pt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Continue training from a saved checkpoint for true epochs.")
    p.add_argument("--init-from", type=Path, default=DEFAULT_INIT, help="checkpoint to start from")
    p.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--out-dir", type=Path, default=None, help="NEW folder for outputs (must not exist, unless --resume)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2, help="early stopping: stop after this many epochs without val improvement; 0 = OFF")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--optimizer", choices=["fresh", "restore"], default="fresh",
                   help="fresh = new AdamW; restore = continue the checkpoint's own optimizer and ORIGINAL learning-rate schedule")
    p.add_argument("--learning-rate", type=float, default=1.5e-4, help="[fresh] peak LR of the new optimizer")
    p.add_argument("--warmup-steps", type=int, default=20, help="[fresh] warm-up steps")
    p.add_argument("--original-lr", type=float, default=6e-4, help="[restore] peak LR of the original run")
    p.add_argument("--original-warmup", type=int, default=100, help="[restore] warm-up steps of the original run")
    p.add_argument("--original-total-steps", type=int, default=2000, help="[restore] total steps the original schedule was defined over")
    p.add_argument("--order", choices=["shuffled", "sequential"], default="shuffled",
                   help="shuffled = random offset + shuffled windows each epoch; sequential = first token to last, no shuffle, no random offset")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run in --out-dir from its newest epoch_<n>.pt")
    p.add_argument("--max-epochs-this-run", type=int, default=0, help="stop after this many epochs in THIS invocation (0 = no limit); resume later")
    p.add_argument("--fp16", action="store_true", help="half-precision maths on the GPU (about 1.9x faster, less memory); the loss and the optimizer stay in full precision")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--device", default="auto", help="auto = the GPU when PyTorch can see one, else the CPU; or cpu / cuda")
    p.add_argument("--log-every", type=int, default=20, help="print a progress line every N batches")
    return p.parse_args(argv)


def epoch_windows(n_tokens, seq_len, rng, order="shuffled"):
    """Cut the stream of n_tokens tokens into windows covering every next-token pair exactly once.

    Returns (starts, lengths). Window (s, L) has inputs data[s:s+L] and targets data[s+1:s+L+1]; it covers
    the pairs s .. s+L-1. All pairs 0 .. n_tokens-2 are covered once:
        head   (0, o)                          o = offset, 0 <= o < seq_len  (omitted if o == 0)
        full   (o + k*seq_len, seq_len)        k = 0 .. K-1,  K = (n_tokens-1-o) // seq_len
        tail   (o + K*seq_len, leftover)       leftover = n_tokens-1-o-K*seq_len     (omitted if 0)

    order="shuffled"   : o is drawn at random and the windows are shuffled (new every epoch).
    order="sequential" : o = 0, NO shuffle, NO random numbers are used: the windows run from the first token
                         to the last, in the same order in every epoch.
    """
    if order not in ("shuffled", "sequential"):
        raise ValueError(f"order must be 'shuffled' or 'sequential', got {order!r}.")
    pairs = n_tokens - 1
    if pairs < 1:
        raise ValueError("Need at least 2 tokens to make a next-token pair.")
    if order == "sequential":
        offset = 0
    else:
        offset = int(rng.integers(0, min(seq_len, pairs + 1)))
        offset = min(offset, pairs)
    k = (pairs - offset) // seq_len
    leftover = pairs - offset - k * seq_len
    starts, lengths = [], []
    if offset > 0:
        starts.append(0); lengths.append(offset)
    starts.extend(offset + np.arange(k) * seq_len); lengths.extend([seq_len] * k)
    if leftover > 0:
        starts.append(offset + k * seq_len); lengths.append(leftover)
    starts, lengths = np.asarray(starts, dtype=np.int64), np.asarray(lengths, dtype=np.int64)
    if order == "sequential":
        return starts, lengths                     # already in stream order; nothing random
    perm = rng.permutation(len(starts))
    return starts[perm], lengths[perm]


def build_batch(data, starts, lengths, seq_len, pad_id):
    """Inputs and next-token targets for some windows. Short windows are padded with pad_id on BOTH sides
    (the loss ignores padded targets; causal attention means padding after the text cannot affect it)."""
    x = np.full((len(starts), seq_len), pad_id, dtype=np.int64)
    y = np.full((len(starts), seq_len), pad_id, dtype=np.int64)
    for row, (s0, n) in enumerate(zip(starts, lengths)):
        x[row, :n] = data[s0:s0 + n]
        y[row, :n] = data[s0 + 1:s0 + n + 1]
    return torch.from_numpy(x), torch.from_numpy(y)


def lr_at(step, total_steps, args):
    """Learning rate of a FRESH optimizer: warm-up, then cosine decay to 10% over this run."""
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / args.warmup_steps
    progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
    return 0.1 * args.learning_rate + 0.9 * args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


MAIN_TOKENIZER = BACKEND_DIR / "data" / "v2" / "tokenizer" / "tokenizer.json"      # the vocabulary every dataset builder uses; it only ever grows (vocab_sync.py)


def check_checkpoint(ck, ck_path, data_meta):
    """Refuse to train if the checkpoint does not match the tokenizer and dataset. Returns the tokenizer."""
    for key in ("model_state_dict", "config", "vocab_size", "tokenizer_sha256"):
        if key not in ck:
            raise SystemExit(f"Checkpoint {ck_path} is missing '{key}'.")
    tok_file = Path(ck_path).parent / "tokenizer.json"
    if not tok_file.is_file():
        raise SystemExit(f"No tokenizer.json next to the checkpoint ({tok_file}).")
    tok = BPETokenizer(tok_file)
    if ck["tokenizer_sha256"] != tok.sha256 or ck["vocab_size"] != tok.vocab_size:
        raise SystemExit("The checkpoint does not match the tokenizer saved next to it.")
    if data_meta.get("tokenizer_sha256") != tok.sha256 and MAIN_TOKENIZER.is_file() and BPETokenizer(MAIN_TOKENIZER).sha256 == data_meta.get("tokenizer_sha256"):
        # the dataset was built with the main vocabulary, which has GROWN since this checkpoint was trained: add the new tokens to the model
        # (old tokens keep their rows and ids; each new token starts as the average of the two pieces it glues) and train with the main vocabulary
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import extend_vocab
        try:
            added = extend_vocab.adapt_checkpoint(ck, tok_file, MAIN_TOKENIZER)
        except ValueError as exc:
            raise SystemExit(f"The dataset uses a different vocabulary and the checkpoint cannot be adapted to it: {exc}")
        print(f"The main vocabulary grew since this checkpoint: {added:,} tokens added to the model (now {ck['vocab_size']:,}).", flush=True)
        tok = BPETokenizer(MAIN_TOKENIZER)
    if data_meta.get("tokenizer_sha256") != tok.sha256:
        raise SystemExit(
            "The dataset in data/processed was built with a DIFFERENT tokenizer than this checkpoint "
            f"(dataset {str(data_meta.get('tokenizer_sha256'))[:12]}, checkpoint {tok.sha256[:12]}). "
            "Rebuild the dataset with this checkpoint's tokenizer, or pick the matching checkpoint.")
    return tok


def load_split(data_dir, name):
    path = Path(data_dir) / f"{name}.bin"
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run build_dataset.py first.")
    return np.memmap(path, dtype=np.uint16, mode="r")


def settings_of(args, seq_len, init_path):
    """Everything that must be identical for a resume to be a true continuation."""
    return {"epochs": args.epochs, "batch_size": args.batch_size, "sequence_length": seq_len, "optimizer": args.optimizer,
            "learning_rate": args.learning_rate, "warmup_steps": args.warmup_steps, "original_lr": args.original_lr,
            "original_warmup": args.original_warmup, "original_total_steps": args.original_total_steps,
            "patience": args.patience, "seed": args.seed, "init_from": str(Path(init_path).resolve()), "order": args.order}


def to_cpu(obj, _memo=None):
    """Copy tensors (also inside dicts / lists) to the CPU: a checkpoint written on the GPU loads anywhere.
    Tensors that share memory on the GPU (the tied embedding / output weights) still share one copy, so the file does not grow."""
    memo = {} if _memo is None else _memo
    if torch.is_tensor(obj):
        key = (obj.device, obj.data_ptr(), tuple(obj.shape), obj.dtype) if obj.numel() else None
        if key is not None and key in memo:
            return memo[key]
        out = obj.detach().cpu()
        if key is not None:
            memo[key] = out
        return out
    if isinstance(obj, dict):
        return {k: to_cpu(v, memo) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v, memo) for v in obj)
    return obj


def make_state(model, optimizer, config, tok, rng, *, epoch, step, history, best_val, best_path, bad_epochs, start_val,
               settings, status, init_from):
    """A complete, resumable training state (also loadable by the chat server)."""
    last = history[-1] if history else {}
    return {
        "model_state_dict": to_cpu(model.state_dict()), "optimizer_state_dict": to_cpu(optimizer.state_dict()),
        "config": config, "vocab_size": tok.vocab_size, "tokenizer_sha256": tok.sha256,
        "step": step, "final_step": step, "epoch": epoch, "best_val_loss": best_val,
        "last_train_loss": last.get("train_loss"), "last_val_loss": last.get("val_loss"),
        "history": history, "status": status, "init_from": str(init_from), "settings": settings,
        "rng_state": rng.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
        "bad_epochs": bad_epochs, "start_val": start_val, "best_path": str(best_path) if best_path else None,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_atomic(state, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # never leave a half-written file


def newest_epoch_checkpoint(out_dir):
    found = sorted(out_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return found[-1] if found else None


def run(args):
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    init_path = Path(args.init_from)
    if not init_path.is_file():
        raise SystemExit(f"Checkpoint not found: {init_path}")
    out_dir = Path(args.out_dir) if args.out_dir else (
        BACKEND_DIR / "training" / "checkpoints_bpe" / time.strftime("continue_%Y%m%d_%H%M%S"))
    resume_ck_path = None
    if args.resume:
        if not out_dir.is_dir():
            raise SystemExit(f"Nothing to resume: {out_dir} does not exist.")
        resume_ck_path = newest_epoch_checkpoint(out_dir)
        if resume_ck_path is None:
            raise SystemExit(f"Nothing to resume: no epoch_<n>.pt in {out_dir}.")
    elif out_dir.exists():
        raise SystemExit(f"{out_dir} already exists. Choose a new --out-dir (existing files are never overwritten), or use --resume.")

    meta_path = Path(args.data_dir) / "meta.json"
    data_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    ck = torch.load(init_path, map_location="cpu", weights_only=False)
    tok = check_checkpoint(ck, init_path, data_meta)
    try:
        model, config = model_from_checkpoint(ck)               # explicit head count; weights loaded strictly
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"The checkpoint's model does not match the current code: {exc}")
    seq_len = config["sequence_length"]
    settings = settings_of(args, seq_len, init_path)

    train_data, val_data = load_split(args.data_dir, "train"), load_split(args.data_dir, "val")
    for name, data in (("train", train_data), ("val", val_data)):
        if len(data) <= seq_len + 1:
            raise SystemExit(f"{name} split is shorter than one sequence.")
        if int(data.max()) >= tok.vocab_size:
            raise SystemExit(f"{name}.bin has token ids outside the vocabulary.")
    if (np.asarray(train_data) == tok.pad_id).any():
        raise SystemExit("The training data contains the PAD token id, so padded targets could not be told apart. Refusing to train.")

    pairs_per_epoch = len(train_data) - 1                       # every next-token pair, once per epoch
    max_windows = pairs_per_epoch // seq_len + 2               # full windows + head + tail (upper bound)
    steps_per_epoch = math.ceil(max_windows / args.batch_size)
    total_steps = steps_per_epoch * args.epochs                # only used by the FRESH schedule
    original_ns = SimpleNamespace(learning_rate=args.original_lr, warmup_steps=args.original_warmup,
                                  max_steps=args.original_total_steps)

    def lr_for(step_):
        return original_lr_at(step_, original_ns) if args.optimizer == "restore" else lr_at(step_, total_steps, args)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model.to(device)
    amp = bool(args.fp16 and device.type == "cuda")                    # half precision only makes sense on the GPU
    scaler = torch.amp.GradScaler("cuda", enabled=amp)                 # no-op when amp is off: the loop below is then exactly the full-precision one
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    history, best_path, bad_epochs = [], None, 0
    start_epoch = 1

    if resume_ck_path is not None:
        rk = torch.load(resume_ck_path, map_location="cpu", weights_only=False)
        if rk["tokenizer_sha256"] != tok.sha256:
            raise SystemExit("The resumable checkpoint was made with a different tokenizer. Refusing to resume.")
        if rk["config"] != config:
            raise SystemExit("The resumable checkpoint's model configuration differs from the starting checkpoint's. Refusing to resume.")
        saved_settings = {"order": "shuffled", **rk["settings"]}      # runs saved before --order existed were shuffled
        if saved_settings != settings:
            diff = {k: (saved_settings.get(k), settings.get(k)) for k in settings if saved_settings.get(k) != settings.get(k)}
            raise SystemExit(f"Settings differ from the interrupted run (saved vs now): {diff}. Use exactly the same arguments to resume.")
        model.load_state_dict(rk["model_state_dict"], strict=True)
        optimizer.load_state_dict(rk["optimizer_state_dict"])
        rng.bit_generator.state = rk["rng_state"]
        torch.set_rng_state(rk["torch_rng_state"])
        step, history, bad_epochs = rk["step"], rk["history"], rk["bad_epochs"]
        best_val, start_val = rk["best_val_loss"], rk["start_val"]
        best_path = Path(rk["best_path"]) if rk["best_path"] else None
        start_epoch = rk["epoch"] + 1
    else:
        if args.optimizer == "restore":
            state = ck.get("optimizer_state_dict")
            if not state:
                raise SystemExit("--optimizer restore needs the checkpoint to contain optimizer state, and it does not.")
            adam_steps = {int(v["step"]) for v in state["state"].values() if "step" in v}
            if adam_steps != {ck.get("step")}:
                raise SystemExit(f"The checkpoint's step counter ({ck.get('step')}) does not match its optimizer's update count "
                                 f"({sorted(adam_steps)}), so this is not a clean continuation. Use --optimizer fresh instead.")
            optimizer.load_state_dict(state)
            step = int(ck["step"])
        else:
            step = 0
        out_dir.mkdir(parents=True)
        shutil.copyfile(tok.path, out_dir / "tokenizer.json")
        start_val = None
        best_val = None
    log_file = out_dir / "train_log.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if resume_ck_path is None:
        start_val = evaluate(model, val_data, seq_len, args.batch_size, tok.pad_id)
        best_val = start_val
    log("=" * 70)
    log(("RESUMING" if resume_ck_path else "CONTINUE TRAINING") + " (true epochs, one optimizer update per batch, no gradient accumulation)")
    log("=" * 70)
    log(f"Starting from      : {init_path}  (step {ck.get('step')}, saved val loss {ck.get('best_val_loss')})")
    if resume_ck_path:
        log(f"Resuming from      : {resume_ck_path.name} (epochs 1-{start_epoch - 1} already done, global step {step})")
    log(f"Output folder      : {out_dir}")
    log(f"Model              : {count_parameters(model):,} parameters | vocab {tok.vocab_size} | seq len {seq_len} | "
        f"{config['num_heads']} attention heads x {config['embedding_size'] // config['num_heads']} dims")
    log(f"Tokenizer          : fingerprint {tok.sha256[:12]} (matches checkpoint and dataset)")
    log(f"Train tokens       : {len(train_data):,} | validation tokens {len(val_data):,} (never trained on)")
    if args.order == "sequential":
        log(f"Training pairs     : {pairs_per_epoch:,} next-token pairs, ALL used exactly once per epoch, SEQUENTIALLY from the first token "
            f"to the last (no shuffle, no random offset: identical window order in every epoch)")
    else:
        log(f"Training pairs     : {pairs_per_epoch:,} next-token pairs, ALL used exactly once per epoch (random offset + shuffle each epoch)")
    log(f"Batch size         : {args.batch_size} windows per batch -> about {steps_per_epoch} batches (= optimizer steps) per epoch")
    log(f"Epochs             : {args.epochs}")
    log(f"Early stopping     : " + (f"patience {args.patience}" if args.patience > 0 else "OFF (all epochs will be run)"))
    if args.optimizer == "restore":
        log(f"Optimizer          : AdamW state RESTORED from the checkpoint (Adam step {step}); ORIGINAL schedule continues: "
            f"peak lr {args.original_lr}, warm-up {args.original_warmup}, cosine to 10% over {args.original_total_steps} steps; "
            f"lr for the first update = {lr_for(step):.6f}")
    else:
        log(f"Optimizer          : fresh AdamW, peak lr {args.learning_rate}, warm-up {args.warmup_steps}, cosine to 10%")
    log(f"Validation loss of the starting point (measured now): {start_val:.4f}" if resume_ck_path is None
        else f"Validation loss of the starting checkpoint (from the first start): {start_val:.4f}")
    log("")

    stopped_early, epochs_this_run = False, 0
    last_epoch_done = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        starts, lengths = epoch_windows(len(train_data), seq_len, rng, args.order)
        loss_sum, targets_done, batches = 0.0, 0, 0
        t0 = time.time()
        for b0 in range(0, len(starts), args.batch_size):
            x, y = build_batch(train_data, starts[b0:b0 + args.batch_size], lengths[b0:b0 + args.batch_size], seq_len, tok.pad_id)
            x, y = x.to(device), y.to(device)
            n_targets = int((y != tok.pad_id).sum())
            lr_now = lr_for(step)
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = lm_loss(logits.float() if amp else logits, y, tok.pad_id)
            if not torch.isfinite(loss):
                raise SystemExit(f"Loss became {loss.item()} at step {step}. Stopping (nothing was overwritten).")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)                 # ONE update per batch (the scaler skips it only when half precision overflowed)
            scaler.update()
            step += 1; batches += 1
            loss_sum += loss.item() * n_targets; targets_done += n_targets   # per-TOKEN average
            if args.log_every and batches % args.log_every == 0:
                log(f"  epoch {epoch} | batch {batches}/{math.ceil(len(starts) / args.batch_size)} | step {step} | lr {lr_now:.6f} | running train loss {loss_sum / targets_done:.4f}")
        if targets_done != pairs_per_epoch:        # every pair exactly once per epoch, no more, no less
            raise SystemExit(f"Internal error: {targets_done} pairs trained, expected {pairs_per_epoch}.")

        train_loss = loss_sum / targets_done
        val_loss = evaluate(model, val_data, seq_len, args.batch_size, tok.pad_id)
        improved = val_loss < best_val
        history.append({"epoch": epoch, "batches": batches, "pairs_trained": targets_done, "steps_total": step, "lr_last_update": lr_now,
                        "train_loss": train_loss, "val_loss": val_loss, "improved": improved})
        epochs_done_after = epoch
        is_last = epoch == args.epochs
        common = dict(epoch=epoch, step=step, history=history, settings=settings, init_from=init_path, start_val=start_val)
        if improved:
            best_val, bad_epochs, best_path = val_loss, 0, out_dir / "best.pt"
            save_atomic(make_state(model, optimizer, config, tok, rng, best_val=best_val, best_path=best_path, bad_epochs=bad_epochs,
                                   status="completed", **common), best_path)
            (out_dir / "latest.json").write_text(json.dumps({
                "checkpoint": "best.pt", "status": "completed", "weights_from_step": step, "final_step": step,
                "weights_from_epoch": epoch, "best_val_loss": val_loss, "tokenizer_sha256": tok.sha256,
                "vocab_size": tok.vocab_size, "config": config, "init_from": str(init_path),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
        else:
            bad_epochs += 1
        early_stop = args.patience > 0 and bad_epochs >= args.patience
        finished = is_last or early_stop
        state = make_state(model, optimizer, config, tok, rng, best_val=best_val, best_path=best_path, bad_epochs=bad_epochs,
                           status="completed" if finished else "in_progress", **common)
        epoch_path = out_dir / f"epoch_{epoch}.pt"
        save_atomic(state, epoch_path)                  # resumable: model + optimizer + epoch + config + tokenizer fingerprint + RNG
        if finished:
            save_atomic(state, out_dir / "final.pt")
        verdict = "NEW BEST" if improved else (f"no improvement ({bad_epochs}" + (f"/{args.patience})" if args.patience > 0 else " in a row; early stopping is off)"))
        log(f"EPOCH {epoch}/{args.epochs} | batches this epoch {batches} (steps so far {step}) | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | {verdict} | saved {epoch_path.name}"
            f"{' + final.pt' if finished else ''} | best checkpoint: {best_path if best_path else '(none yet - starting checkpoint is still best)'} | {time.time() - t0:.0f}s")
        (out_dir / "epochs.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        last_epoch_done = epoch
        epochs_this_run += 1
        if early_stop:
            log(f"Early stop: validation loss did not improve for {args.patience} epochs in a row.")
            stopped_early = True
            break
        if args.max_epochs_this_run and epochs_this_run >= args.max_epochs_this_run and not is_last:
            log(f"Paused after {epochs_this_run} epoch(s) as requested. Epoch {epoch} is saved; continue with the same command plus --resume.")
            return {"out_dir": out_dir, "best_path": best_path, "history": history, "start_val": start_val, "best_val": best_val,
                    "steps": step, "paused": True}

    log("")
    log(f"FINISHED after {len(history)} epoch(s), global step {step}{' (early stop)' if stopped_early else ''}.")
    if best_path:
        log(f"Lowest validation loss {best_val:.4f} (start {start_val:.4f}) -> best.pt (epoch {min(history, key=lambda h: h['val_loss'])['epoch']}); "
            f"final checkpoint: {out_dir / 'final.pt'}")
    else:
        log(f"No epoch beat the starting checkpoint ({start_val:.4f}). The final checkpoint is still saved: {out_dir / 'final.pt'}; "
            f"your starting checkpoint {init_path} remains the lowest-loss model.")
    return {"out_dir": out_dir, "best_path": best_path, "history": history, "start_val": start_val, "best_val": best_val,
            "steps": step, "paused": False}


if __name__ == "__main__":
    run(parse_args())
