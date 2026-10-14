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
            embedding_function = DefaultEmbeddingFunction()
        self.col = client.get_or_create_collection(collection, embedding_function=embedding_function, metadata={"hnsw:space": "cosine"})

    @classmethod
    def from_env(cls, prefix=""):
        """The configured store, or None when no Chroma variables are set."""
        return cls(prefix=prefix) if configured(prefix) else None

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

    def count(self):
        return self.col.count()

    def delete_source(self, source_id):
        self.col.delete(where={"source_id": source_id})
