"""ELVARN knowledge service: the GraphRAG pipeline (Neo4j graph + Chroma vectors + Redis memory) behind a secret token.

    ELVARN site (Render)  --POST /v1/context-->  this service  -->  Neo4j (facts)   Chroma (passages)   Redis (memory + cache)
         Authorization: Bearer <token>

The heavy parts (graph driver, vector client, the embedding model) live HERE, so the chat site stays small. The site only
receives the ranked, cited context (facts + passages) and writes its answer from it, exactly as it does with local stores.

Endpoints
    GET    /healthz        no token; answers "ok" without touching any store (Render health check)
    GET    /readyz         token; says which of the three stores are reachable
    POST   /v1/context     token; {"question", "session"} -> the merged, ranked context (see knowledge/graphrag.py)
    POST   /v1/remember    token; stores one finished turn in the user's memory
    DELETE /v1/memory      token; deletes one user's memory (?session=...)
Anything else: 404 / 405. A missing or wrong token gives the same 401 body. Logs hold only status codes and timings,
never the token, the question text or client addresses.

Two ways to run (the service picks one at start-up)
    snapshot (free hosting)   the graph and vectors baked into the image (knowledge_service/snapshot/); Redis runs inside the container
    live                      real Neo4j + Chroma + Redis, configured with the variables below
    It runs from the snapshot when one is bundled and NEO4J_URI is not set (or when SNAPSHOT_DIR is set).

Environment
    KNOWLEDGE_TOKEN                 required, at least 24 characters; without it the service refuses to start
    NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD      the graph (neo4j+s://... for Aura)
    CHROMA_API_KEY / CHROMA_TENANT / CHROMA_DATABASE   or CHROMA_HOST / CHROMA_PORT / CHROMA_SSL / CHROMA_TOKEN
    REDIS_URL                       memory and cache (rediss://... for a hosted Redis)
    KNOWLEDGE_RATE_PER_MINUTE       global requests per minute, default 120
    KNOWLEDGE_MAX_CONCURRENT        requests handled at the same time, default 4
Any store that is not configured or not reachable is simply left out; the answer is built from the ones that work.
"""
import hmac
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from backend.knowledge import graphrag

log = logging.getLogger("knowledge_service")
MIN_TOKEN_CHARS = 24
SESSION_RE = r"^[A-Za-z0-9_-]{1,64}$"


class ConfigError(Exception):
    pass


class TokenBucket:
    """A global requests-per-minute limit (the same idea as the search gatekeeper)."""

    def __init__(self, per_minute):
        self.capacity, self.tokens, self.rate = float(per_minute), float(per_minute), per_minute / 60.0
        self.stamp, self.lock = time.monotonic(), threading.Lock()

    def allow(self):
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.stamp) * self.rate)
            self.stamp = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class ContextRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    session: Optional[str] = Field(default=None, pattern=SESSION_RE)


class Entity(BaseModel):
    id: str = Field(max_length=300)
    type: str = Field(max_length=40)
    name: str = Field(max_length=300)


class RememberRequest(BaseModel):
    session: str = Field(pattern=SESSION_RE)
    question: str = Field(max_length=1000)
    answer: str = Field(default="", max_length=4000)
    entities: List[Entity] = Field(default_factory=list, max_length=6)


def create_app(token, kb=None, rag=None, memory=None, rate_per_minute=120, max_concurrent=4, queue_seconds=0.0):
    """Build the service. Tests pass fake stores; production uses create_from_env()."""
    if not isinstance(token, str) or len(token) < MIN_TOKEN_CHARS:
        raise ConfigError(f"KNOWLEDGE_TOKEN is missing or shorter than {MIN_TOKEN_CHARS} characters")
    secret = token.encode("utf-8")
    bucket, slots = TokenBucket(max(1, int(rate_per_minute))), threading.BoundedSemaphore(max(1, int(max_concurrent)))
    app = FastAPI(title="ELVARN knowledge service", docs_url=None, redoc_url=None, openapi_url=None)

    def check(authorization):
        given = (authorization or "")[7:].encode("utf-8") if (authorization or "").lower().startswith("bearer ") else b""
        if not hmac.compare_digest(given, secret):                       # constant time; same answer for missing and wrong
            raise HTTPException(401, "unauthorized")
        if not bucket.allow():
            raise HTTPException(429, "too many requests")

    def busy():
        # a free slot at once, or (queue_seconds > 0) wait that long for one: on the 512 MB plan only ONE retrieval runs at a time, so a second
        # visitor waits a moment instead of getting an error
        if not (slots.acquire(timeout=queue_seconds) if queue_seconds > 0 else slots.acquire(blocking=False)):
            raise HTTPException(503, "busy")

    def populated():
        try:
            return bool(kb.populated()) if kb is not None else bool(rag is not None and rag.count() > 0)
        except Exception:
            return False

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(authorization: Optional[str] = Header(default=None)):
        check(authorization)
        out = {}
        for name, probe in (("graph", lambda: kb.populated()), ("vector", lambda: rag.count() >= 0), ("memory", lambda: memory.ping())):
            try:
                out[name] = bool(probe())
            except Exception:
                out[name] = False
        return out

    @app.post("/v1/context")
    def context(req: ContextRequest, authorization: Optional[str] = Header(default=None)):
        check(authorization)
        if kb is None and rag is None:
            raise HTTPException(503, "no store is available")
        busy()
        started = time.perf_counter()
        try:
            ctx = graphrag.build_context(kb, rag, req.question, memory, req.session)
            out = {**ctx.to_dict(), "populated": populated()}
        except Exception as exc:  # noqa: BLE001
            log.warning("context failed: %s", exc.__class__.__name__)
            raise HTTPException(502, "retrieval failed")
        finally:
            slots.release()
        log.info("context items=%d notes=%d ms=%d", len(out["items"]), len(out["notes"]), int((time.perf_counter() - started) * 1000))
        return out

    @app.post("/v1/remember")
    def remember(req: RememberRequest, authorization: Optional[str] = Header(default=None)):
        check(authorization)
        if memory is not None:
            memory.remember(req.session, req.question, req.answer, [e.model_dump() for e in req.entities])
        return {"ok": True}

    @app.delete("/v1/memory")
    def forget(session: str = Query(pattern=SESSION_RE), authorization: Optional[str] = Header(default=None)):
        check(authorization)
        if memory is not None:
            memory.forget(session)
        return {"forgotten": True}

    return app


SNAPSHOT_DEFAULT = Path(__file__).resolve().parent / "snapshot"


def snapshot_dir():
    """The folder of the baked-in snapshot when this service should run from it: SNAPSHOT_DIR if set, else the bundled one when no
    Neo4j is configured. None means 'use the configured stores' (Neo4j + Chroma + Redis)."""
    explicit = os.environ.get("SNAPSHOT_DIR")
    if explicit:
        return Path(explicit)
    return SNAPSHOT_DEFAULT if (SNAPSHOT_DEFAULT / "manifest.json").is_file() and not os.environ.get("NEO4J_URI") else None


def stores_from_snapshot(directory):
    """Free hosted mode: the graph is served from memory, the vectors from an embedded Chroma that was filled from the snapshot
    (at image build time, so start-up is instant; refilled here only if the folder is empty or from another snapshot)."""
    import chromadb
    from backend.knowledge import snapshot
    from backend.knowledge.snapshot_graph import SnapshotGraph
    from backend.knowledge.vectorstore import RagStore
    manifest = snapshot.check_snapshot(directory)
    chroma_path = Path(os.environ.get("CHROMA_PATH", "/tmp/elvarn_chroma"))
    chroma_path.mkdir(parents=True, exist_ok=True)
    rag = RagStore(client=chromadb.PersistentClient(path=str(chroma_path)))
    loaded = snapshot.load_vectors(directory, rag, marker_file=chroma_path / "snapshot.marker")
    rag.lexical = snapshot.ensure_lexical(rag, chroma_path / "lexical.db", loaded)          # BM25 next to the vectors (built at image build time)
    from backend.knowledge.reranker import Reranker
    rag.reranker = Reranker.from_env()                                                      # the 23 MB cross-encoder bundled in knowledge_service/models
    graph = SnapshotGraph(Path(directory) / snapshot.GRAPH)
    log.info("snapshot %s: graph %s, %d passages (%d loaded now)", manifest["created_at"], manifest["graph"], rag.count(), loaded)
    return graph, rag


def create_from_env():
    """uvicorn --factory knowledge_service.app:create_from_env"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from backend.knowledge.kb import KnowledgeBase
    from backend.knowledge.vectorstore import RagStore
    from backend.usermemory.store import Memory

    def attempt(name, make):
        try:
            return make()
        except Exception as exc:  # noqa: BLE001 - a store that is down is left out, the others still work
            log.warning("%s unavailable at start-up: %s", name, exc.__class__.__name__)
            return None
    directory = snapshot_dir()
    if directory is not None:
        graph_rag = attempt("snapshot", lambda: stores_from_snapshot(directory))
        kb, rag = graph_rag if graph_rag else (None, None)
    else:
        kb = attempt("graph", KnowledgeBase.from_env)
        rag = attempt("vector store", RagStore.from_env)
    memory = attempt("memory", Memory.from_env)
    log.info("mode=%s stores: graph=%s vector=%s memory=%s", "snapshot" if directory is not None else "live", kb is not None, rag is not None, memory is not None)
    return create_app(os.environ.get("KNOWLEDGE_TOKEN", ""), kb, rag, memory,
                      rate_per_minute=int(os.environ.get("KNOWLEDGE_RATE_PER_MINUTE", "120")),
                      max_concurrent=int(os.environ.get("KNOWLEDGE_MAX_CONCURRENT", "4")),
                      queue_seconds=float(os.environ.get("KNOWLEDGE_QUEUE_SECONDS", "0")))
