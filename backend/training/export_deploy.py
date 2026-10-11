"""Export a trained checkpoint as a small weights-only folder that can be committed to git and served.

A training checkpoint also holds the optimizer state and RNG state (about 3x the size of the weights).
The server only needs the weights, the model config, the tokenizer and latest.json.

Usage (project root):
    python backend/training/export_deploy.py <checkpoint_dir> <output_dir>
Example:
    python backend/training/export_deploy.py backend/training/checkpoints_bpe/4m deploy/models/7m
"""
import json
import shutil
import sys
import time
from pathlib import Path

import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402


def export(src, dst):
    src, dst = Path(src), Path(dst)
    latest = json.loads((src / "latest.json").read_text(encoding="utf-8"))
    if latest.get("status") != "completed":
        raise SystemExit(f"{src} is not a completed run.")
    ck = torch.load(src / latest["checkpoint"], map_location="cpu", weights_only=False)
    tok_file = src / "tokenizer.json"
    tok = BPETokenizer(tok_file)
    if ck["tokenizer_sha256"] != tok.sha256:
        raise SystemExit("Tokenizer does not match the checkpoint. Refusing to export.")

    dst.mkdir(parents=True, exist_ok=True)
    slim = {k: ck[k] for k in ("model_state_dict", "config", "vocab_size", "tokenizer_sha256", "step",
                               "best_val_loss", "final_step", "status") if k in ck}
    torch.save(slim, dst / latest["checkpoint"])
    shutil.copyfile(tok_file, dst / "tokenizer.json")
    latest["exported_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    latest["weights_only"] = True
    (dst / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in dst.iterdir()) / 1e6
    print(f"Exported {src} -> {dst}  ({size:.1f} MB: weights, tokenizer, latest.json)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    export(sys.argv[1], sys.argv[2])
