"""Grow a trained ELVARN model into a bigger one WITHOUT starting from scratch.

The old weights are copied into a wider and/or deeper network. Everything new starts as "do nothing", so the grown
model gives (almost) the same answers as the old one, and training then continues from there.

  * Deeper: extra layers are added at the end; their attention / MLP outputs start at zero, so the layer passes the
    signal through unchanged. (Exact: same logits.)
  * Wider: the old numbers fill the first part of every matrix; the new channels start at zero or small noise and
    the new attention heads output zero. LayerNorm weights are rescaled because it now averages over more channels.
    (Very close: the loss differs by about 0.0005, not exactly equal.)

Usage (from the project root):
    python backend/training/grow_model.py --init-from <trained checkpoint>/best.pt --out-dir backend/training/checkpoints_bpe/grown_17m \
        --embedding-size 384 --hidden-size 1536 --num-heads 12 --num-layers 7

Never overwrites: the output folder must not already exist. The tokenizer is copied unchanged.
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from model.transformer import build_model, config_from_checkpoint, count_parameters  # noqa: E402
from tokenizer.bpe_tokenizer import SPECIAL_TOKENS, BPETokenizer  # noqa: E402


def _is_layer_norm(key):
    return "layer_norm" in key


def grow_width(state, config, vocab_size, embedding_size, hidden_size, num_heads, noise=0.02, seed=0):
    old_e, old_h = config["embedding_size"], config["hidden_size"]
    if embedding_size < old_e or hidden_size < old_h:
        raise ValueError("Growing can only make the model bigger, not smaller.")
    new_config = {**config, "embedding_size": embedding_size, "hidden_size": hidden_size, "num_heads": num_heads}
    gen = torch.Generator().manual_seed(seed)
    shapes = build_model(vocab_size, new_config).state_dict()
    scale = math.sqrt(old_e / embedding_size)             # LayerNorm now spreads over more (empty) channels
    new = {}
    for key, old in state.items():
        target = shapes[key]
        if key == "lm_head.weight":
            continue                                        # tied to the token embedding, set at the end
        if key in ("token_embedding.weight", "position_embedding.weight"):
            t = torch.zeros_like(target)
            t[:, :old_e] = old
        elif _is_layer_norm(key):
            t = torch.zeros_like(target)
            t[:old_e] = old * scale if key.endswith("weight") else old
        elif key == "lm_head.bias" or target.dim() == 1 and key.endswith(".bias"):
            t = torch.zeros_like(target)
            t[:old.shape[0]] = old
        elif ".attention.W_O.weight" in key or ".fc2.weight" in key:
            t = torch.zeros_like(target)                    # new heads / hidden units add nothing at first
            t[:old.shape[0], :old.shape[1]] = old
        else:                                               # W_Q W_K W_V fc1: new rows get small noise, new input columns zero
            t = torch.randn(target.shape, generator=gen) * noise
            t[:, old.shape[1]:] = 0
            t[:old.shape[0], :old.shape[1]] = old
        new[key] = t
    new["lm_head.weight"] = new["token_embedding.weight"]
    return new, new_config


def grow_depth(state, config, vocab_size, num_layers):
    old = config["num_layers"]
    if num_layers < old:
        raise ValueError("Growing can only make the model bigger, not smaller.")
    new_config = {**config, "num_layers": num_layers}
    shapes = build_model(vocab_size, new_config).state_dict()
    new = {k: v.clone() for k, v in state.items() if not k.startswith("decoder_blocks.")}
    for key, value in state.items():
        if key.startswith("decoder_blocks."):
            new[key] = value.clone()
    for slot in range(old, num_layers):                     # extra layers: copy the last layer, then silence its outputs
        src = f"decoder_blocks.{old - 1}."
        for key, value in state.items():
            if not key.startswith(src):
                continue
            t = value.clone()
            if ".attention.W_O." in key or ".fc2." in key:
                t.zero_()
            elif _is_layer_norm(key):
                t = torch.ones_like(t) if key.endswith("weight") else torch.zeros_like(t)
            new[key.replace(src, f"decoder_blocks.{slot}.")] = t
    assert set(new) == set(shapes), "grown state does not match the new architecture"
    return new, new_config


def remap_vocabulary(state, old_tok, new_tok):
    """Move the model to a NEW tokenizer. Each new piece starts as the average of the old embeddings of the old pieces
    that spell the same bytes (e.g. new "Government" = mean of old "Gov" and "ernment"). Special tokens are copied by name.
    Everything that is not tied to the vocabulary (layers, positions) is unchanged."""
    emb, bias = state["token_embedding.weight"], state.get("lm_head.bias")
    new_vocab = new_tok.vocab_size
    new_emb = torch.zeros(new_vocab, emb.shape[1])
    new_bias = torch.zeros(new_vocab) if bias is not None else None
    vocab = new_tok._tok.get_vocab()                         # piece text (in byte-level letters) -> new id
    for piece, new_id in vocab.items():
        if piece in SPECIAL_TOKENS:
            old_ids = [old_tok.token_id(piece)]
        else:
            old_ids = [t.id for t in old_tok._tok.model.tokenize(piece)]   # old BPE on the same byte letters
        old_ids = [i for i in old_ids if i is not None] or [old_tok.unk_id]
        new_emb[new_id] = emb[old_ids].mean(0)
        if new_bias is not None:
            new_bias[new_id] = bias[old_ids].mean()
    out = {k: v.clone() for k, v in state.items()}
    out["token_embedding.weight"] = new_emb
    out["lm_head.weight"] = new_emb.clone()
    if new_bias is not None:
        out["lm_head.bias"] = new_bias
    return out


def grow(checkpoint, embedding_size, hidden_size, num_heads, num_layers, seed=0, old_tok=None, new_tok=None):
    config = config_from_checkpoint(checkpoint)
    if not config.get("tie_weights"):
        raise SystemExit("Growing is only written for models with tied embeddings.")
    vocab_size = checkpoint["vocab_size"]
    state = {k: v.clone() for k, v in checkpoint["model_state_dict"].items()}
    if new_tok is not None:
        state = remap_vocabulary(state, old_tok, new_tok)
        vocab_size = new_tok.vocab_size
    state, config = grow_width(state, config, vocab_size, embedding_size, hidden_size, num_heads, seed=seed)
    state, config = grow_depth(state, config, vocab_size, num_layers)
    model = build_model(vocab_size, config)
    model.load_state_dict(state, strict=True)
    return model, config, vocab_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", type=Path, required=True, help="the trained checkpoint to grow (tokenizer.json must be next to it)")
    ap.add_argument("--out-dir", type=Path, required=True, help="new folder (must not exist)")
    ap.add_argument("--embedding-size", type=int, default=384)
    ap.add_argument("--hidden-size", type=int, default=1536)
    ap.add_argument("--num-heads", type=int, default=12)
    ap.add_argument("--num-layers", type=int, default=7)
    ap.add_argument("--new-tokenizer", type=Path, default=None, help="move to this NEW tokenizer.json (embeddings are averaged from the old pieces)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} already exists; refusing to overwrite. Choose a new folder.")
    ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
    tok_file = args.init_from.parent / "tokenizer.json"
    if not tok_file.is_file():
        raise SystemExit(f"No tokenizer.json next to {args.init_from}.")
    old_tok = BPETokenizer(tok_file)
    new_tok = BPETokenizer(args.new_tokenizer) if args.new_tokenizer else None
    old_model = build_model(ck["vocab_size"], config_from_checkpoint(ck))
    model, config, vocab_size = grow(ck, args.embedding_size, args.hidden_size, args.num_heads, args.num_layers, args.seed, old_tok, new_tok)
    out_tok = new_tok or old_tok
    args.out_dir.mkdir(parents=True)
    shutil.copyfile(args.new_tokenizer or tok_file, args.out_dir / "tokenizer.json")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "vocab_size": vocab_size,
                "tokenizer_sha256": out_tok.sha256, "step": 0, "final_step": 0, "best_val_loss": None,
                "status": "grown", "grown_from": str(args.init_from)}, args.out_dir / "grown.pt")
    info = {"checkpoint": "grown.pt", "config": config, "vocab_size": vocab_size, "tokenizer_sha256": out_tok.sha256,
            "parameters_before": count_parameters(old_model), "parameters_after": count_parameters(model), "grown_from": str(args.init_from)}
    (args.out_dir / "latest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
