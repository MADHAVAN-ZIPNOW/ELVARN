"""Give a trained ELVARN model a LONGER context window (e.g. 128 -> 512 tokens) without losing anything it learned.

Only one thing in the model depends on the context length: the learned position table (one row per position).
Attention itself has no length limit built in (it uses a causal mask made on the fly), and every other weight
has the same shape for any length. So the extension is:

  * copy EVERY weight unchanged;
  * make the position table longer. Rows 0..old_length-1 are copied bit for bit, so for any input of at most
    old_length tokens the logits are EXACTLY the same as before (tested);
  * fill the new rows (old_length .. new_length-1) with a starting guess, then fine-tune on long examples.

Starting guess for the new rows (--init):
  tile    (default) row p = trained row (p mod old_length) + small noise. The new rows are made of vectors the
          network already knows how to read, so the model does not see strange numbers at positions 128+.
          Trade-off: position 130 looks a bit like position 2, so the model cannot tell them apart until
          fine-tuning teaches it (the noise lets training pull them apart). Good start, needs training.
  random  new rows get small random numbers (std 0.02, like a fresh model). Trade-off: these positions are a clean
          slate, but the first steps of fine-tuning see unfamiliar vectors there; usually slower to settle.
Linear interpolation ("stretching" the table) was NOT chosen because it would change rows 0..127 and break the
"short inputs unchanged" guarantee.

Cost of 512 (back-of-envelope ESTIMATES, not measured): the position table grows by 384 x 384 = 147,456 numbers
(+0.9% parameters). Per token the layers need about 2 x 17M = 34M multiply-adds of matrix work, and attention adds
about 4 x T x 384 per layer: ~1.2M x 6 layers at T=128 (about 3% extra) versus ~4.7M x 6 at T=512 (about 14%
extra). So one token costs roughly 10% more at 512 and a whole 512 window about 4.4x one 128 window; a batch of
8 x 512 tokens is about as much work as 32 x 128 plus ~10%. Expect roughly 900-1,000 tokens/second instead of
1,000-1,100. Memory: if attention scores are kept for the backward pass they take 512 x 512 x 12 heads x 4 bytes
x 6 layers = about 75 MB per window (about 0.6 GB for batch 8), so use batch 8 or less on CPU.
Generation here recomputes the whole window for every new token (no cache), so a 512-token prompt makes every
answer token about 4-5x slower than a 128-token prompt.

Usage (project root):
    python backend/training/extend_context.py --init-from <checkpoint>.pt --out-dir <NEW folder> --sequence-length 512

Never overwrites: the output folder must not exist. tokenizer.json (next to the checkpoint) is copied unchanged.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from model.transformer import build_model, config_from_checkpoint, count_parameters  # noqa: E402
from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402

INIT_METHODS = ("tile", "random")


def extend_positions(table, new_length, init="tile", noise=0.01, seed=0):
    """A (new_length x E) position table whose first rows are exactly `table`."""
    old_length, width = table.shape
    if new_length < old_length:
        raise ValueError(f"The context can only grow: {old_length} -> {new_length} is smaller.")
    if init not in INIT_METHODS:
        raise ValueError(f"init must be one of {INIT_METHODS}.")
    gen = torch.Generator().manual_seed(seed)
    out = torch.empty(new_length, width, dtype=table.dtype)
    out[:old_length] = table                                   # bit-identical
    extra = new_length - old_length
    if extra:
        if init == "tile":
            rows = torch.arange(old_length, new_length) % old_length
            out[old_length:] = table[rows] + torch.randn(extra, width, generator=gen) * noise
        else:
            out[old_length:] = torch.randn(extra, width, generator=gen) * 0.02
    return out


def extend_context(checkpoint, new_length, init="tile", noise=0.01, seed=0):
    """Return (model, config) with a context of new_length tokens and every old weight copied."""
    config = config_from_checkpoint(checkpoint)
    state = {k: v.clone() for k, v in checkpoint["model_state_dict"].items()}
    state["position_embedding.weight"] = extend_positions(state["position_embedding.weight"], new_length, init, noise, seed)
    new_config = {**config, "sequence_length": new_length}
    model = build_model(checkpoint["vocab_size"], new_config)
    model.load_state_dict(state, strict=True)
    return model, new_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--init-from", type=Path, required=True, help="the trained checkpoint (tokenizer.json must be next to it)")
    ap.add_argument("--out-dir", type=Path, required=True, help="NEW folder (must not exist)")
    ap.add_argument("--sequence-length", type=int, default=512, help="new context length in tokens (must be >= the current one)")
    ap.add_argument("--init", choices=INIT_METHODS, default="tile", help="how the new position rows start (see the top of this file)")
    ap.add_argument("--noise", type=float, default=0.01, help="[tile] size of the random noise added to the copied rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} already exists; refusing to overwrite. Choose a new folder.")
    tok_file = args.init_from.parent / "tokenizer.json"
    if not tok_file.is_file():
        raise SystemExit(f"No tokenizer.json next to {args.init_from}.")
    ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
    tok = BPETokenizer(tok_file)
    if ck.get("tokenizer_sha256") != tok.sha256 or ck.get("vocab_size") != tok.vocab_size:
        raise SystemExit("The checkpoint does not match the tokenizer saved next to it.")
    old_config = config_from_checkpoint(ck)
    try:
        model, config = extend_context(ck, args.sequence_length, args.init, args.noise, args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc))
    args.out_dir.mkdir(parents=True)
    shutil.copyfile(tok_file, args.out_dir / "tokenizer.json")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "vocab_size": tok.vocab_size,
                "tokenizer_sha256": tok.sha256, "step": 0, "final_step": 0, "best_val_loss": None,
                "status": "context_extended", "extended_from": str(args.init_from),
                "position_init": args.init}, args.out_dir / "extended.pt")
    info = {"checkpoint": "extended.pt", "config": config, "vocab_size": tok.vocab_size, "tokenizer_sha256": tok.sha256,
            "sequence_length_before": old_config["sequence_length"], "sequence_length_after": config["sequence_length"],
            "position_init": args.init, "parameters_after": count_parameters(model), "extended_from": str(args.init_from)}
    (args.out_dir / "latest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
