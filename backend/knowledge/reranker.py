"""Cross-encoder reranker: scores (question, passage) PAIRS together, so it can tell that a passage about Section 83(1)(a) answers a question
about Section 83(1)(a) better than a nearby passage about Section 80. Vector and keyword search score the question and each passage separately.

The model is cross-encoder/ms-marco-MiniLM-L-6-v2 as a quantized ONNX file (about 23 MB, run with onnxruntime and tokenizers, which Chroma already
needs: no torch, no new package). The files live in knowledge_service/models/reranker (or the folder named by RERANKER_DIR).
Without the files nothing changes: retrieval stays as it was.
"""
import math
import os
import threading
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_service" / "models" / "reranker"          # committed: the hosted image carries it
MAX_TOKENS = 192                                     # measured: longer inputs and bigger batches pushed the 512 MB hosted service out of memory


class Reranker:
    def __init__(self, directory):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = Path(directory)
        self._np = np
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.enable_cpu_mem_arena = False                 # no growing memory pool: this runs inside a 512 MB container
        options.enable_mem_pattern = False
        self._lock = threading.Lock()                        # one rerank at a time: concurrent questions must not each hold their own buffers
        self.session = ort.InferenceSession(str(d / "model_quantized.onnx"), options, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    @classmethod
    def from_env(cls):
        """The reranker when its files exist, else None."""
        d = Path(os.environ.get("RERANKER_DIR") or DEFAULT_DIR)
        if not ((d / "model_quantized.onnx").is_file() and (d / "tokenizer.json").is_file()):
            return None
        try:
            return cls(d)
        except Exception:  # noqa: BLE001 - a broken or missing runtime must never break retrieval
            return None

    def scores(self, question, texts, batch=4):
        """One relevance score per text (higher is better; a logit, 0 is neutral)."""
        out = []
        with self._lock:
            for start in range(0, len(texts), batch):
                enc = self.tokenizer.encode_batch([(question, t) for t in texts[start:start + batch]])
                feed = {"input_ids": self._np.array([e.ids for e in enc], dtype="int64"),
                        "attention_mask": self._np.array([e.attention_mask for e in enc], dtype="int64")}
                if "token_type_ids" in self.inputs:
                    feed["token_type_ids"] = self._np.array([e.type_ids for e in enc], dtype="int64")
                logits = self.session.run(None, feed)[0]
                out.extend(float(row[0]) for row in logits)
        return out


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))
