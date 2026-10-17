"""RAG store on Chroma: every document passage with its embedding, for similarity search.

The ids are the same as the Passage ids in Neo4j (kb.passage_id), so a Chroma hit can be followed into the graph and a
graph fact can point back to its text. Writes are upserts, so reloading a document is harmless.

Connection (all optional; with none of them set the store is disabled):
  local / self-hosted server   CHROMA_HOST, CHROMA_PORT (default 8001), CHROMA_SSL=1 for https, CHROMA_TOKEN for a bearer token
  Chroma Cloud                 CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE
Embeddings are computed on the client side with Chroma's default model (all-MiniLM-L6-v2, ONNX, runs on the CPU), so a process
that queries needs roughly 300 MB extra memory for the model. The same variables with a prefix (TARGET_CHROMA_HOST ...) describe a
second store for migrate_stores.py.
"""
import os

COLLECTION = "elvarn_passages"
BATCH = 64
META_KEYS = ("source_id", "title", "url", "doc_type", "publisher", "published_date", "licence", "locator", "position")


def clean_meta(meta):
    """Chroma metadata may not contain None; keep only the known keys with real values."""
    return {k: meta[k] for k in META_KEYS if meta.get(k) is not None}


def configured(prefix=""):
    """True when the environment describes a Chroma store (self-hosted or cloud)."""
    return bool(os.environ.get(prefix + "CHROMA_HOST") or os.environ.get(prefix + "CHROMA_API_KEY"))


def make_client(prefix=""):
    """A Chroma client built from the environment (see the module docstring). `prefix` selects a second set, e.g. "TARGET_"."""
    import chromadb
    env = lambda name, default=None: os.environ.get(prefix + name, default)          # noqa: E731
    if env("CHROMA_API_KEY"):
        return chromadb.CloudClient(tenant=env("CHROMA_TENANT"), database=env("CHROMA_DATABASE"), api_key=env("CHROMA_API_KEY"))
    ssl = str(env("CHROMA_SSL", "")).strip().lower() in ("1", "true", "yes", "on")
    headers = {"Authorization": "Bearer " + env("CHROMA_TOKEN")} if env("CHROMA_TOKEN") else None
    return chromadb.HttpClient(host=env("CHROMA_HOST", "localhost"), port=int(env("CHROMA_PORT", "443" if ssl else "8001")), ssl=ssl, headers=headers)


class RagStore:
    def __init__(self, client=None, embedding_function=None, collection=COLLECTION, prefix=""):
        if client is None:
            client = make_client(prefix)
        if embedding_function is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            if os.environ.get("ELVARN_PUBLIC") == "1":
                embedding_function = DefaultEmbeddingFunction()
            else:                                                      # this computer: the GPU service while it runs (same model, ten times faster), else the CPU
                from backend.knowledge.gpu_embed import best_embedding_function
                embedding_function = best_embedding_function()
        self.col = client.get_or_create_collection(collection, embedding_function=embedding_function, metadata={"hnsw:space": "cosine"})
        self.lexical = None                                          # the BM25 keyword index (backend/knowledge/lexical.py), when there is one
        self.reranker = None                                         # the cross-encoder (backend/knowledge/reranker.py), when its files exist

    @classmethod
    def from_env(cls, prefix=""):
        """The configured store, or None when no Chroma variables are set. The keyword index is attached when its file exists."""
        if not configured(prefix):
            return None
        store = cls(prefix=prefix)
        if not prefix:
            from backend.knowledge.lexical import Lexical
            from backend.knowledge.reranker import Reranker
            store.lexical = Lexical.from_env()
            store.reranker = Reranker.from_env()
        return store

    def add(self, items):
        """Upsert [{id, text, meta}] (any size; sent in batches). Returns how many were written."""
        items = [i for i in items if i["text"].strip()]
        for start in range(0, len(items), BATCH):
            batch = items[start:start + BATCH]
            self.col.upsert(ids=[i["id"] for i in batch], documents=[i["text"] for i in batch], metadatas=[clean_meta(i["meta"]) for i in batch])
        return len(items)

    def missing(self, ids):
        """The subset of `ids` not yet stored (so a load can skip passages that are already indexed)."""
        have = set()
        ids = list(ids)
        for start in range(0, len(ids), 500):
            have.update(self.col.get(ids=ids[start:start + 500], include=[])["ids"])
        return [i for i in ids if i not in have]

    def query(self, text, k=8, where=None):
        """Nearest passages: [{id, text, meta, distance}] best first (cosine distance: 0 = identical, 1 = unrelated)."""
        if not text.strip() or self.col.count() == 0:
            return []
        r = self.col.query(query_texts=[text], n_results=min(k, self.col.count()), where=where, include=["documents", "metadatas", "distances"])
        return [{"id": i, "text": d, "meta": m or {}, "distance": dist}
                for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def hybrid_query(self, text, k=8, pool=30, rrf_k=60):
        """Vector search AND keyword (BM25) search, fused by reciprocal rank fusion: [{id, text, meta, distance, rrf}] best first. A passage found
        only by keywords has distance None (no vector distance was computed for it). Without a keyword index this is the plain vector query."""
        dense = self.query(text, k=pool)
        lex = self.lexical.search(text, k=pool * 3) if self.lexical is not None else []
        if not lex:
            return dense[:k]
        score = {}
        for rank, hit in enumerate(dense):
            score[hit["id"]] = score.get(hit["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (pid, _) in enumerate(lex):
            score[pid] = score.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
        by_id = {h["id"]: h for h in dense}
        absent = [pid for pid, _ in lex if pid not in by_id]
        if absent:
            got = self.col.get(ids=absent, include=["documents", "metadatas"])
            for pid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                by_id[pid] = {"id": pid, "text": doc, "meta": meta or {}, "distance": None}
        from backend.knowledge.lexical import exact_matcher
        exact = exact_matcher(text) if os.environ.get("ELVARN_EXACT_BOOST", "1") == "1" else None                                  # a question that names "Section 83(1)(a) of <Act>": passages that have it come first
        flags = {i: bool(exact and exact(by_id[i]["text"])) for i in score if i in by_id}
        order = sorted(flags, key=lambda i: (not flags[i], -score[i]))[:k]
        return [dict(by_id[i], rrf=score[i], exact=flags[i]) for i in order]

    def count(self):
        return self.col.count()

    def delete_source(self, source_id):
        self.col.delete(where={"source_id": source_id})
