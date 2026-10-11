"""Compare trained ELVARN checkpoints fairly: bits per character on the SAME validation text.

A loss per token is not comparable between models with different vocabularies,
so this measures total negative log-likelihood divided by the number of
characters in the validation documents (lower is better).

Usage (project root):
    python backend/training/compare_models.py <checkpoint_dir> [<checkpoint_dir> ...]
Each directory must hold final.pt and the tokenizer.json saved with it.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "training"))

import corpus  # noqa: E402
from model.transformer import build_model, count_parameters  # noqa: E402
from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402


@torch.inference_mode()
def bits_per_char(ckpt_dir, val_docs):
    ckpt_dir = Path(ckpt_dir)
    tok = BPETokenizer(ckpt_dir / "tokenizer.json")
    ck = torch.load(ckpt_dir / "final.pt", map_location="cpu", weights_only=False)
    if ck["tokenizer_sha256"] != tok.sha256:
        raise SystemExit(f"{ckpt_dir}: tokenizer does not match the checkpoint")
    model = build_model(ck["vocab_size"], ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    T = model.sequence_length

    ids = []
    for d in val_docs:
        ids += tok.encode(d["text"], add_bos=True, add_eos=True)
    data = torch.tensor(ids, dtype=torch.long)
    total_nll, predicted = 0.0, 0
    for start in range(0, len(data) - 1, T):
        x = data[start:start + T].unsqueeze(0)
        y = data[start + 1:start + T + 1]
        x = x[:, :len(y)]
        if x.size(1) == 0:
            break
        logits = model(x)[0]
        total_nll += F.cross_entropy(logits, y, reduction="sum").item()
        predicted += len(y)
    chars = sum(len(d["text"]) for d in val_docs)
    return {
        "dir": str(ckpt_dir.name or ckpt_dir), "params": count_parameters(model), "vocab": tok.vocab_size,
        "tokens": predicted, "chars": chars, "loss_per_token": total_nll / predicted,
        "bits_per_char": total_nll / math.log(2) / chars,
        "step": ck.get("step"), "val_loss_at_training": ck.get("best_val_loss"),
    }


def main():
    dirs = sys.argv[1:]
    if not dirs:
        raise SystemExit(__doc__)
    _, val_docs, _ = corpus.split_documents(corpus.load_documents())
    print(f"validation documents: {len(val_docs)} | characters: {sum(len(d['text']) for d in val_docs):,}\n")
    print(f"{'model':8} {'params':>11} {'vocab':>7} {'step':>6} {'loss/token':>11} {'bits/char':>10}")
    for d in dirs:
        r = bits_per_char(d, val_docs)
        print(f"{r['dir']:8} {r['params']:>11,} {r['vocab']:>7} {r['step']:>6} {r['loss_per_token']:>11.3f} {r['bits_per_char']:>10.3f}")


if __name__ == "__main__":
    main()
