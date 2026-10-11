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
