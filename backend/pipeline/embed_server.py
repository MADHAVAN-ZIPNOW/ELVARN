"""The GPU embedding service for Chroma (see backend/knowledge/gpu_embed.py). Run it with the GPU environment:

    .venv-gpu\\Scripts\\python.exe -m backend.pipeline.embed_server          (or scripts\\start_embed_gpu.ps1)

all-MiniLM-L6-v2 in half precision on the GPU, mean pooled over the attention mask and normalised: the same recipe as Chroma's own ONNX copy of
the model. Only listens on 127.0.0.1. Needs torch (CUDA) and transformers; the model files download once from Hugging Face.
    POST /embed  {"texts": [...]}  ->  {"vectors": [[...384 numbers...], ...]}        GET /health -> {"ok": true, "device": "cuda"}
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 256                     # the model's own limit: longer passages are embedded from their first 256 word pieces, as Chroma's default does
BATCH = 128
PORT = 8099


class Embedder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.lock = Lock()                                                   # one batch on the GPU at a time

    def embed(self, texts):
        torch = self.torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))        # similar lengths together: less padding
        out = [None] * len(texts)
        with self.lock, torch.no_grad():
            for s in range(0, len(order), BATCH):
                idx = order[s:s + BATCH]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vec = torch.nn.functional.normalize(vec.float(), dim=1).cpu().tolist()
                for i, v in zip(idx, vec):
                    out[i] = v
        return out


def make_handler(embedder):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200 if self.path == "/health" else 404, {"ok": self.path == "/health", "device": embedder.device})

        def do_POST(self):
            if self.path != "/embed":
                return self._send(404, {"error": "not found"})
            try:
                size = int(self.headers.get("Content-Length") or 0)
                texts = json.loads(self.rfile.read(size))["texts"]
                self._send(200, {"vectors": embedder.embed([str(t) for t in texts])})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": str(exc)[:200]})

        def log_message(self, *args):
            pass
    return Handler


def main():
    embedder = Embedder()
    print(f"embedding service on 127.0.0.1:{PORT} ({embedder.device})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(embedder)).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
