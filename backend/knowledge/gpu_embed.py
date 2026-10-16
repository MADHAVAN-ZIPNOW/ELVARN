"""Chroma embeddings from the GPU, with the CPU as the fallback.

Chroma's default embedding is the all-MiniLM-L6-v2 model on the CPU: about 22 passages a second on this computer, which makes a billion tokens
(about 1.5 million passages) a 16 hour job. `backend/pipeline/embed_server.py` runs the SAME model on the GPU (about 230 passages a second) as a small
local service; this module is the client side. The vectors are the same kind (same model, mean pooling, normalised), so passages embedded by either
side live in one collection and the stored ones need no re-embedding.

    GPU_EMBED_URL   address of the service (default http://127.0.0.1:8099); set GPU_EMBED_URL=off to always use the CPU

If the service is not running, or fails half way, a call falls back to the CPU model: nothing breaks, it is only slower.
"""
import json
import os
import urllib.request

URL = os.environ.get("GPU_EMBED_URL", "http://127.0.0.1:8099")
CHUNK = 256


def alive(url=None, timeout=0.6):
    url = url or URL
    if url == "off":
        return False
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return json.load(r).get("ok") is True
    except Exception:  # noqa: BLE001 - not running, or not reachable: use the CPU
        return False


def _remote(texts, url, timeout=120):
    vectors = []
    for i in range(0, len(texts), CHUNK):
        req = urllib.request.Request(url.rstrip("/") + "/embed", data=json.dumps({"texts": texts[i:i + CHUNK]}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            vectors += json.load(r)["vectors"]
    return vectors


def best_embedding_function():
    """The default Chroma embedding function, answered by the GPU service while it is running.
    The GPU is tried FIRST on every call (a separate 'are you alive?' check timed out whenever the service was busy and silently sent the work to the
    CPU); only a real failure moves a call to the CPU, and then the GPU is left alone for 30 seconds before it is tried again."""
    import time
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    class GpuOrCpu(DefaultEmbeddingFunction):
        down_until = 0.0

        def __call__(self, input):  # noqa: A002 - Chroma's parameter name
            texts = [str(t) for t in input]
            if URL != "off" and time.time() >= GpuOrCpu.down_until:
                try:
                    return _remote(texts, URL)
                except Exception:  # noqa: BLE001 - the service is not running or failed: this call is done on the CPU
                    GpuOrCpu.down_until = time.time() + 30
            return super().__call__(texts)
    return GpuOrCpu()
