"""Loads the latest completed ELVARN checkpoint and its matching tokenizer."""
import json
import os
import sys
import threading
from pathlib import Path

import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from model.generation import generate  # noqa: E402
from model.transformer import build_model, count_parameters  # noqa: E402
from tokenizer.bpe_tokenizer import BPETokenizer, TOKENIZER_FILE  # noqa: E402

DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get("ELVARN_CHECKPOINT_DIR") or BACKEND_DIR / "training" / "checkpoints_bpe"
)


class ModelNotReady(Exception):
    """Raised when no compatible completed checkpoint can be served."""


class ModelService:
    def __init__(self, checkpoint_dir=None, tokenizer_file=TOKENIZER_FILE):
        self.checkpoint_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
        self.tokenizer_file = Path(tokenizer_file)
        self._lock = threading.Lock()
        self._state = None       # dict with model, tokenizer, info
        self.last_error = None

    # ---------------------------------------------------------------- load
    def _load(self):
        latest = self.checkpoint_dir / "latest.json"
        if not latest.is_file():
            raise ModelNotReady(
                "No trained model found. Train ELVARN first: "
                "python backend/training/train_batch.py"
            )
        meta = json.loads(latest.read_text(encoding="utf-8"))
        if meta.get("status") != "completed":
            raise ModelNotReady("The latest training run is not marked completed.")
        ckpt_path = self.checkpoint_dir / meta["checkpoint"]
        if not ckpt_path.is_file():
            raise ModelNotReady(f"Checkpoint file missing: {ckpt_path.name}")

        try:
            tokenizer = BPETokenizer(self.tokenizer_file)
        except FileNotFoundError as exc:
            raise ModelNotReady(str(exc)) from exc

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("tokenizer_sha256") != tokenizer.sha256:
            raise ModelNotReady(
                "Tokenizer does not match this checkpoint (different vocabulary). "
                "Refusing to load, to avoid garbage output."
            )
        if ck.get("vocab_size") != tokenizer.vocab_size:
            raise ModelNotReady("Checkpoint vocabulary size does not match the tokenizer.")

        model = build_model(tokenizer.vocab_size, ck["config"])
        model.load_state_dict(ck["model_state_dict"])
        model.eval()

        info = {
            "checkpoint": ckpt_path.name,
            "checkpoint_path": str(ckpt_path),
            "weights_from_step": ck.get("step"),
            "final_step": ck.get("final_step", ck.get("step")),
            "best_val_loss": ck.get("best_val_loss"),
            "status": ck.get("status"),
            "smoke_run": bool(meta.get("smoke_run")),
            "completed_at": meta.get("completed_at"),
            "vocab_size": tokenizer.vocab_size,
            "parameters": count_parameters(model),
            "sequence_length": model.sequence_length,
            "tokenizer_sha256": tokenizer.sha256[:12],
        }
        return {"model": model, "tokenizer": tokenizer, "info": info}

    def load(self):
        """Load or reload. The old model is replaced only if the new one loads."""
        try:
            new_state = self._load()
        except ModelNotReady as exc:
            self.last_error = str(exc)
            raise
        with self._lock:
            self._state = new_state
            self.last_error = None
        return new_state["info"]

    # --------------------------------------------------------------- status
    @property
    def ready(self):
        return self._state is not None

    def info(self):
        return dict(self._state["info"]) if self._state else None

    def health(self):
        return {
            "server": "ok",
            "model_ready": self.ready,
            "tokenizer_ready": self.tokenizer_file.is_file(),
            "checkpoint": self._state["info"]["checkpoint"] if self._state else None,
            "error": self.last_error,
        }

    # ------------------------------------------------------------- generate
    def generate(self, text, **kwargs):
        if not self._state:
            raise ModelNotReady(
                self.last_error or "Model is not loaded. Train ELVARN first."
            )
        with self._lock:  # one generation at a time on CPU
            state = self._state
            result = generate(state["model"], state["tokenizer"], text, **kwargs)
        result["checkpoint"] = state["info"]["checkpoint"]
        result["weights_from_step"] = state["info"]["weights_from_step"]
        return result
