import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "training"))

import train_batch  # noqa: E402  (the real training code)
from model.transformer import build_model  # noqa: E402
from tokenizer import train_bpe  # noqa: E402
from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402

TINY_CONFIG = {
    "embedding_size": 32, "hidden_size": 64, "sequence_length": 32,
    "num_layers": 2, "num_heads": 2, "dropout": 0.0, "tie_weights": True,
}

CORPUS = [
    "<CASUAL_CONVERSATION>\nUser: hii\nAssistant: Hey! How is your day going?",
    "<CASUAL_CONVERSATION>\nUser: what does btw mean?\nAssistant: BTW means by the way.",
    "<DOMAIN_KNOWLEDGE>\nTopic: Audit\nQuestion: What is the CAG?\nAnswer: The Comptroller and Auditor General audits government accounts.",
    "<GENERAL_ENGLISH>\nQuestion: What does transparent mean?\nAnswer: Transparent means open and clear.",
    "<HUMOUR>\nUser: I am bored.\nAssistant: Your brain opened too many tabs.",
] * 20


@pytest.fixture(scope="session")
def tok_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_bpe.train(CORPUS, 300, out_file=path)
    return path


@pytest.fixture(scope="session")
def tok(tok_file):
    return BPETokenizer(tok_file)


def write_checkpoint(directory, tokenizer, seed=0, steps=15):
    """Train a tiny model with the real training helpers and save it in the
    same layout train_batch.py produces (final.pt + latest.json)."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    ids = []
    for text in CORPUS:
        ids += tokenizer.encode(text, add_bos=True, add_eos=True)
    data = np.asarray(ids, dtype=np.uint16)
    model = build_model(tokenizer.vocab_size, TINY_CONFIG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(steps):
        x, y = train_batch.get_batch(data, 8, TINY_CONFIG["sequence_length"], rng)
        loss = train_batch.lm_loss(model(x), y, tokenizer.pad_id)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    train_batch.save_checkpoint(
        directory / "final.pt", model, opt, steps, losses[-1], TINY_CONFIG,
        tokenizer, [], rng.bit_generator.state, "completed")
    (directory / "latest.json").write_text(json.dumps({
        "checkpoint": "final.pt", "status": "completed", "final_step": steps,
        "best_val_loss": losses[-1], "smoke_run": False,
        "completed_at": "test", "config": TINY_CONFIG}))
    return model, losses


@pytest.fixture()
def ckpt_dir(tmp_path, tok):
    write_checkpoint(tmp_path / "ckpt", tok)
    return tmp_path / "ckpt"


@pytest.fixture
def kb():
    """A clean knowledge graph for one test. Needs a DISPOSABLE Neo4j: set NEO4J_TEST_URI (e.g. bolt://localhost:7688).
    Community Neo4j has a single database, so the fixture wipes everything in it and refuses to touch the real instance."""
    import os
    uri = os.environ.get("NEO4J_TEST_URI")
    if not uri:
        pytest.skip("set NEO4J_TEST_URI to a disposable Neo4j to run knowledge-graph tests")
    if uri.rstrip("/") == (os.environ.get("NEO4J_URI") or "").rstrip("/"):
        pytest.fail("NEO4J_TEST_URI is the real database; refusing to wipe it")
    from knowledge.kb import KnowledgeBase
    k = KnowledgeBase(uri, password=os.environ.get("NEO4J_TEST_PASSWORD", "testpassword123"))
    k._run("MATCH (n) DETACH DELETE n")
    yield k
    k._run("MATCH (n) DETACH DELETE n")
    k.close()


def _hash_embedding_class():
    """A deterministic bag-of-words embedder for tests: texts that share words are close (no model download needed)."""
    import hashlib
    from chromadb import EmbeddingFunction

    class HashEmbedding(EmbeddingFunction):
        DIM = 256

        def __init__(self):
            pass

        def __call__(self, input):
            out = []
            for text in input:
                v = [0.0] * self.DIM
                for w in __import__("re").findall(r"[a-z0-9]+", text.lower()):
                    if len(w) >= 3:
                        v[int(hashlib.md5(w[:5].encode()).hexdigest(), 16) % self.DIM] += 1.0
                norm = sum(x * x for x in v) ** 0.5 or 1.0
                out.append([x / norm for x in v])
            return out

        @staticmethod
        def name():
            return "hash-embedding-test"

        def get_config(self):
            return {}

        @staticmethod
        def build_from_config(config):
            return HashEmbedding()

        def default_space(self):
            return "cosine"

    return HashEmbedding


@pytest.fixture
def rag_store():
    """An in-memory Chroma collection with the fake embedder (a fresh collection per test)."""
    import uuid
    import chromadb
    from knowledge.vectorstore import RagStore
    return RagStore(client=chromadb.EphemeralClient(), embedding_function=_hash_embedding_class()(), collection="t" + uuid.uuid4().hex[:12])


@pytest.fixture
def memory():
    import fakeredis
    from usermemory.store import Memory
    return Memory(fakeredis.FakeRedis(decode_responses=True))
