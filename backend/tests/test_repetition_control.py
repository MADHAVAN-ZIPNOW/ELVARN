"""Repetition controls (repetition_penalty, no_repeat_ngram_size) in generation and in the API."""
import json

import pytest
import torch
from fastapi.testclient import TestClient

import api.server as server
from api.server import create_app, repetition_settings, GenerateRequest
from model import generation
from model.generation import (apply_repetition_penalty, banned_by_ngram, generate, generate_stream)
from search.orchestrate import SearchPlan, SearchResult
from api.model_service import ModelService


def load(ckpt_dir, tok_file):
    svc = ModelService(ckpt_dir, tok_file)
    svc.load()
    return svc._state["model"], svc._state["tokenizer"]


# ------------------------------------------------------------ pure helpers
def test_penalty_lowers_seen_tokens_for_positive_and_negative_logits():
    logits = torch.tensor([2.0, -2.0, 1.0, 0.5])
    out = apply_repetition_penalty(logits, [0, 1], 2.0)
    assert out[0].item() == pytest.approx(1.0)      # positive: divided
    assert out[1].item() == pytest.approx(-4.0)     # negative: multiplied
    assert out[2].item() == 1.0 and out[3].item() == 0.5
    assert logits[0].item() == 2.0                  # input not modified
    # probability of the repeated token goes down
    assert torch.softmax(out, -1)[0] < torch.softmax(logits, -1)[0]


def test_penalty_one_or_no_tokens_is_identity():
    logits = torch.randn(10)
    assert apply_repetition_penalty(logits, [1, 2], 1.0) is logits
    assert apply_repetition_penalty(logits, [], 1.5) is logits


def test_ngram_blocking_helper():
    assert banned_by_ngram([1, 2, 3, 1, 2], 3) == {3}           # "1 2" was followed by 3 before
    assert banned_by_ngram([1, 2, 3, 4], 3) == set()
    assert banned_by_ngram([5, 5, 5], 2) == {5}
    assert banned_by_ngram([1, 2], 0) == set()
    assert banned_by_ngram([1], 3) == set()
    assert banned_by_ngram([7, 8, 7], 1) == {7, 8}                # n = 1: no token may repeat


# ----------------------------------------------------- generation behaviour
def test_defaults_unchanged(ckpt_dir, tok_file):
    model, tok = load(ckpt_dir, tok_file)
    a = generate(model, tok, "hii", max_new_tokens=15, greedy=True)
    b = generate(model, tok, "hii", max_new_tokens=15, greedy=True, repetition_penalty=1.0, no_repeat_ngram_size=0)
    assert a["text"] == b["text"] and a["raw_text"] == b["raw_text"]
    assert a["repetition_penalty"] == 1.0 and a["no_repeat_ngram_size"] == 0


def test_ngram_blocking_in_real_generation(ckpt_dir, tok_file, monkeypatch):
    model, tok = load(ckpt_dir, tok_file)
    captured, real_sample = [], generation.sample_next

    def spy(logits, *a, **k):
        captured.append(real_sample(logits, *a, **k))
        return captured[-1]
    monkeypatch.setattr(generation, "sample_next", spy)
    generate(model, tok, "hii", max_new_tokens=30, greedy=True, no_repeat_ngram_size=2)
    pairs = list(zip(captured, captured[1:]))
    assert len(pairs) == len(set(pairs))            # no 2-token sequence of sampled ids occurs twice


def test_penalty_changes_a_looping_model():
    """A model that always wants token 5 repeats it; with the controls it cannot."""
    class Tok:
        pad_id, unk_id, bos_id, eos_id, vocab_size = 0, 1, 2, 3, 12

        def encode(self, text, add_bos=False, add_eos=False):
            return [2, 6, 7]

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(f"w{i}" for i in ids)

    class Loop(torch.nn.Module):
        sequence_length = 64

        def forward(self, x):
            out = torch.zeros(1, x.size(1), 12)
            out[..., 5] = 3.0
            out[..., 8] = 2.5
            out[..., 9] = 2.0
            return out

    tok, model = Tok(), Loop()
    plain = generate(model, tok, "x", mode="raw", max_new_tokens=8, greedy=True)
    assert plain["raw_text"].split() == ["w5"] * 8
    pen = generate(model, tok, "x", mode="raw", max_new_tokens=8, greedy=True, repetition_penalty=1.5)
    assert pen["raw_text"].split()[:3] == ["w5", "w8", "w5"] or pen["raw_text"].split()[1] != "w5"
    assert len(set(pen["raw_text"].split())) > 1
    ng = generate(model, tok, "x", mode="raw", max_new_tokens=8, greedy=True, no_repeat_ngram_size=2)
    words = ng["raw_text"].split()
    pairs = list(zip(words, words[1:]))
    assert len(pairs) == len(set(pairs)) and len(set(words)) > 1


def test_penalty_ignores_prompt_unless_asked():
    class Tok:
        pad_id, unk_id, bos_id, eos_id, vocab_size = 0, 1, 2, 3, 12

        def encode(self, text, add_bos=False, add_eos=False):
            return [2, 5]                 # the prompt contains token 5

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(f"w{i}" for i in ids)

    class M(torch.nn.Module):
        sequence_length = 64

        def forward(self, x):
            out = torch.zeros(1, x.size(1), 12)
            out[..., 5] = 3.0
            out[..., 8] = 2.5
            return out

    first = lambda **k: generate(M(), Tok(), "x", mode="raw", max_new_tokens=1, greedy=True, **k)["raw_text"]
    assert first(repetition_penalty=1.5) == "w5"                       # copying from the notes is allowed
    assert first(repetition_penalty=1.5, penalize_prompt=True) == "w8"


@pytest.mark.parametrize("kw", [{"repetition_penalty": 0.5}, {"repetition_penalty": 3.0},
                                {"no_repeat_ngram_size": -1}, {"no_repeat_ngram_size": 11}])
def test_invalid_values_rejected(ckpt_dir, tok_file, kw):
    model, tok = load(ckpt_dir, tok_file)
    with pytest.raises(ValueError):
        generate(model, tok, "hii", max_new_tokens=2, **kw)


def test_stream_with_controls_matches_non_stream_and_respects_stop_strings(ckpt_dir, tok_file):
    model, tok = load(ckpt_dir, tok_file)
    kw = dict(max_new_tokens=25, greedy=True, repetition_penalty=1.3, no_repeat_ngram_size=3)
    events = list(generate_stream(model, tok, "hii", **kw))
    done = events[-1]
    assert events[0]["event"] == "start" and events[0]["repetition_penalty"] == 1.3
    assert done["event"] == "done" and done["no_repeat_ngram_size"] == 3
    shown = "".join(e["text"] for e in events if e["event"] == "token")
    for stop in generation.STOP_STRINGS:
        assert stop not in shown
    assert generate(model, tok, "hii", **kw)["text"] == done["text"]


# ------------------------------------------------------------------- API
def req(**kw):
    return GenerateRequest(text="q", **kw)


def test_settings_off_without_evidence_and_client_wins():
    assert repetition_settings(req(), 0) == ({"repetition_penalty": 1.0, "no_repeat_ngram_size": 0}, "off")
    s, src = repetition_settings(req(), 2)
    assert s == server.WEB_REPETITION and src == "web_default"
    s, src = repetition_settings(req(repetition_penalty=1.3), 2)
    assert s == {"repetition_penalty": 1.3, "no_repeat_ngram_size": 4} and src == "client"
    s, src = repetition_settings(req(repetition_penalty=1.2), 0)
    assert s == {"repetition_penalty": 1.2, "no_repeat_ngram_size": 0} and src == "client"
    s, src = repetition_settings(req(repetition_penalty=1.0, no_repeat_ngram_size=0), 2)
    assert s == {"repetition_penalty": 1.0, "no_repeat_ngram_size": 0}      # client can switch the web defaults off


@pytest.fixture()
def client(ckpt_dir, tok_file):
    app = create_app(ckpt_dir, load_on_start=False)
    app.state.service.tokenizer_file = tok_file
    app.state.service.load()
    return TestClient(app)


def test_api_validation(client):
    for bad in ({"repetition_penalty": 0.9}, {"repetition_penalty": 2.5}, {"no_repeat_ngram_size": -1}, {"no_repeat_ngram_size": 11}):
        assert client.post("/api/chat", json={"text": "hii", **bad}).status_code == 422


def test_api_default_response_reports_off(client):
    b = client.post("/api/chat", json={"text": "hii", "greedy": True, "max_new_tokens": 5}).json()
    assert b["repetition_penalty"] == 1.0 and b["no_repeat_ngram_size"] == 0 and b["repetition_source"] == "off"


def test_api_client_values_recorded_and_stream(client):
    body = {"text": "hii", "greedy": True, "max_new_tokens": 5, "repetition_penalty": 1.2, "no_repeat_ngram_size": 3}
    b = client.post("/api/chat", json=body).json()
    assert (b["repetition_penalty"], b["no_repeat_ngram_size"], b["repetition_source"]) == (1.2, 3, "client")
    lines = [json.loads(x) for x in client.post("/api/chat/stream", json=body).text.splitlines() if x]
    done = lines[-1]
    assert done["event"] == "done" and done["repetition_penalty"] == 1.2 and done["repetition_source"] == "client"


def test_api_web_defaults_only_when_evidence_in_prompt(client, monkeypatch):
    body = {"text": "hii", "greedy": True, "max_new_tokens": 5, "web_search": "auto"}
    # no evidence in the prompt: off
    monkeypatch.setattr(server, "run_search", lambda *a, **k: SearchResult())
    assert client.post("/api/chat", json=body).json()["repetition_source"] == "off"
    # evidence in the prompt: web defaults, in both the normal and the streaming path
    monkeypatch.setattr(server, "run_search", lambda *a, **k: SearchResult(status="used", searched=True, model_text="hii", evidence_in_prompt=2))
    monkeypatch.setattr(server, "plan_search", lambda *a, **k: SearchPlan("q", "test"))
    b = client.post("/api/chat", json=body).json()
    assert (b["repetition_penalty"], b["no_repeat_ngram_size"], b["repetition_source"]) == (1.15, 4, "web_default")
    lines = [json.loads(x) for x in client.post("/api/chat/stream", json=body).text.splitlines() if x]
    assert lines[-1]["repetition_source"] == "web_default" and lines[-1]["no_repeat_ngram_size"] == 4
