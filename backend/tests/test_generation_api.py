import json

import pytest
import torch
from fastapi.testclient import TestClient

from api.model_service import ModelNotReady, ModelService
from api.server import create_app
from conftest import write_checkpoint
from model.generation import build_prompt, generate


def load(ckpt_dir, tok_file):
    svc = ModelService(ckpt_dir, tok_file)
    svc.load()
    return svc


def test_prompt_formats():
    assert build_prompt("hii", "chat") == "<CASUAL_CONVERSATION>\nUser: hii\nAssistant:"
    assert "Question: q\nAnswer:" in build_prompt("q", "domain")
    assert build_prompt("x", "chat", [("a", "b")]).count("User:") == 2
    with pytest.raises(ValueError):
        build_prompt("x", "nope")


def test_generation_valid_ids_and_limits(ckpt_dir, tok_file):
    svc = load(ckpt_dir, tok_file)
    out = svc.generate("hii", mode="chat", max_new_tokens=12, greedy=True)
    assert 0 <= out["generated_tokens"] <= 12
    assert out["prompt_tokens"] > 0 and out["seconds"] >= 0
    assert out["checkpoint"] == "final.pt"


def test_greedy_is_deterministic_and_sampling_respects_seed(ckpt_dir, tok_file):
    svc = load(ckpt_dir, tok_file)
    a = svc.generate("hii", max_new_tokens=15, greedy=True)["text"]
    b = svc.generate("hii", max_new_tokens=15, greedy=True)["text"]
    assert a == b
    torch.manual_seed(1)
    s1 = svc.generate("hii", max_new_tokens=15, temperature=1.0)["text"]
    torch.manual_seed(1)
    s2 = svc.generate("hii", max_new_tokens=15, temperature=1.0)["text"]
    assert s1 == s2


def test_long_prompt_is_truncated_to_context(ckpt_dir, tok_file):
    svc = load(ckpt_dir, tok_file)
    out = svc.generate("word " * 400, max_new_tokens=3, greedy=True)
    assert out["prompt_tokens"] <= 31          # context 32 minus room for new token


def test_service_matches_direct_model_call(ckpt_dir, tok_file):
    svc = load(ckpt_dir, tok_file)
    state = svc._state
    direct = generate(state["model"], state["tokenizer"], "hii", mode="chat",
                      max_new_tokens=10, greedy=True)
    assert svc.generate("hii", mode="chat", max_new_tokens=10, greedy=True)["text"] == direct["text"]


# ------------------------------------------------------------------ API
@pytest.fixture()
def client(ckpt_dir, tok_file):
    app = create_app(ckpt_dir, load_on_start=False)
    app.state.service.tokenizer_file = tok_file
    app.state.service.load()
    return TestClient(app)


def test_health_ready(client):
    h = client.get("/api/health").json()
    assert h["model_ready"] is True and h["checkpoint"] == "final.pt" and h["server"] == "ok"
    m = client.get("/api/model").json()
    assert m["parameters"] > 0 and m["vocab_size"] > 0 and "chat" in m["modes"]


def test_chat_returns_text_from_real_checkpoint(client, ckpt_dir, tok_file):
    r = client.post("/api/chat", json={"text": "hii", "greedy": True, "max_new_tokens": 10})
    assert r.status_code == 200
    body = r.json()
    direct = load(ckpt_dir, tok_file).generate("hii", mode="chat", max_new_tokens=10, greedy=True)
    assert body["text"] == direct["text"]
    assert body["generated_tokens"] == direct["generated_tokens"]
    assert body["checkpoint"] == "final.pt"


def test_generate_endpoint_with_params(client):
    r = client.post("/api/generate", json={"text": "What is the CAG?", "mode": "domain",
                                           "temperature": 0.5, "top_k": 5, "max_new_tokens": 8})
    assert r.status_code == 200 and r.json()["generated_tokens"] <= 8


@pytest.mark.parametrize("payload", [
    {"text": ""}, {"text": "   "}, {"text": "x" * 1001}, {"text": "hi", "mode": "bogus"},
    {"text": "hi", "temperature": 5}, {"text": "hi", "max_new_tokens": 0},
    {"text": "hi", "max_new_tokens": 9999}, {"text": "hi", "top_k": -1}, {},
])
def test_invalid_inputs_rejected(client, payload):
    assert client.post("/api/chat", json=payload).status_code == 422


def test_missing_checkpoint_gives_clear_error(tmp_path):
    client = TestClient(create_app(tmp_path / "empty"))
    h = client.get("/api/health").json()
    assert h["model_ready"] is False and "train" in h["error"].lower()
    r = client.post("/api/chat", json={"text": "hii"})
    assert r.status_code == 503 and "train" in r.json()["detail"].lower()
    assert client.post("/api/reload").status_code == 503


def test_incomplete_run_is_not_served(ckpt_dir, tok_file):
    meta = json.loads((ckpt_dir / "latest.json").read_text())
    meta["status"] = "in_progress"
    (ckpt_dir / "latest.json").write_text(json.dumps(meta))
    with pytest.raises(ModelNotReady):
        ModelService(ckpt_dir, tok_file).load()


def test_mismatched_tokenizer_is_refused(ckpt_dir, tmp_path):
    from tokenizer import train_bpe
    other = tmp_path / "other.json"
    train_bpe.train(["completely different text for another vocabulary " * 5], 280, out_file=other)
    with pytest.raises(ModelNotReady, match="Tokenizer does not match"):
        ModelService(ckpt_dir, other).load()


def test_reload_switches_to_new_checkpoint(client, ckpt_dir, tok):
    before = client.post("/api/chat", json={"text": "hii", "greedy": True, "max_new_tokens": 12}).json()
    write_checkpoint(ckpt_dir, tok, seed=123, steps=40)          # newer, different weights
    r = client.post("/api/reload")
    assert r.status_code == 200 and r.json()["model"]["final_step"] == 40
    after = client.post("/api/chat", json={"text": "hii", "greedy": True, "max_new_tokens": 12}).json()
    assert after["weights_from_step"] == 40 and before["weights_from_step"] == 15


def test_failed_reload_keeps_old_model_and_reports_error(client, ckpt_dir):
    (ckpt_dir / "final.pt").unlink()
    r = client.post("/api/reload")
    assert r.status_code == 503
    assert client.get("/api/health").json()["error"]             # error is reported, not hidden
    assert client.post("/api/chat", json={"text": "hii", "max_new_tokens": 3}).status_code == 200


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ELVARN" in r.text
