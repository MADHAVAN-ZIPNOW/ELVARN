"""KV cache: the cached incremental path must compute exactly the same function as full recompute."""
import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from model.attention import CausalSelfAttention  # noqa: E402
from model.generation import generate, generate_stream  # noqa: E402
from model.transformer import build_model  # noqa: E402

torch.set_num_threads(2)

CONFIGS = [
    dict(embedding_size=16, hidden_size=32, sequence_length=16, num_layers=1, num_heads=1),
    dict(embedding_size=32, hidden_size=64, sequence_length=32, num_layers=2, num_heads=4),
    dict(embedding_size=48, hidden_size=96, sequence_length=64, num_layers=3, num_heads=6),
]
VOCAB = 40


def make_model(cfg, seed=0, scale=8.0):
    torch.manual_seed(seed)
    model = build_model(VOCAB, {**cfg, "dropout": 0.1, "tie_weights": True})
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)  # big weights so any mistake in positions/masks would show clearly
    model.eval()
    return model


def test_float32_close_in_relative_terms():
    # float32 rounding differs between the two summation orders; compare relative to logit size.
    model = make_model(CONFIGS[1], scale=4.0)
    ids = torch.randint(0, VOCAB, (2, 32))
    with torch.no_grad():
        full = model(ids)
        past, steps = None, []
        for t in range(32):
            logits, past = model.forward_cached(ids[:, t:t + 1], past=past)
            steps.append(logits)
    got = torch.cat(steps, dim=1)
    assert (got - full).abs().max() <= 1e-5 * full.abs().max() * 10


@pytest.mark.parametrize("cfg", CONFIGS)
def test_incremental_logits_match_full_recompute(cfg):
    # float64 so the 1e-5 tolerance measures real differences, not float32 rounding noise
    model = make_model(cfg).double()
    n = cfg["sequence_length"]
    ids = torch.randint(0, VOCAB, (2, n))
    with torch.no_grad():
        full = model(ids)
        # one token at a time
        past, steps = None, []
        for t in range(n):
            logits, past = model.forward_cached(ids[:, t:t + 1], past=past)
            steps.append(logits)
        assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)
        # prefill then chunks of several tokens with a past
        split = n // 3
        logits, past = model.forward_cached(ids[:, :split])
        pieces = [logits]
        pos = split
        for size in (3, 1, 4, 2, 5):
            if pos >= n:
                break
            chunk = ids[:, pos:pos + size]
            logits, past = model.forward_cached(chunk, past=past)
            pieces.append(logits)
            pos += chunk.size(1)
        if pos < n:
            logits, past = model.forward_cached(ids[:, pos:], past=past)
            pieces.append(logits)
        assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-5)
        # last_only returns just the final position
        last, _ = model.forward_cached(ids, last_only=True)
        assert last.shape == (2, 1, VOCAB)
        assert torch.allclose(last[:, 0], full[:, -1], atol=1e-5)


def test_past_is_the_right_shape_and_small():
    cfg = CONFIGS[1]
    model = make_model(cfg)
    ids = torch.randint(0, VOCAB, (1, 10))
    with torch.no_grad():
        _, past = model.forward_cached(ids)
    assert len(past) == cfg["num_layers"]
    for k, v in past:
        assert k.shape == v.shape == (1, cfg["num_heads"], 10, cfg["embedding_size"] // cfg["num_heads"])


def test_exceeding_the_window_raises():
    model = make_model(CONFIGS[0])
    n = CONFIGS[0]["sequence_length"]
    with pytest.raises(ValueError, match="exceeds"):
        model.forward_cached(torch.zeros(1, n + 1, dtype=torch.long))
    _, past = model.forward_cached(torch.zeros(1, n, dtype=torch.long))
    with pytest.raises(ValueError, match="exceeds"):
        model.forward_cached(torch.zeros(1, 1, dtype=torch.long), past=past)
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.zeros(1, n + 1, dtype=torch.long))  # the old path still refuses too


def test_training_path_unchanged():
    """past=None, use_cache=False must equal the original attention math (copied here)."""
    torch.manual_seed(1)
    att = CausalSelfAttention(32, num_heads=4)
    x = torch.randn(2, 9, 32)

    def old_forward(m, H):
        Q, K, V = (m._split_heads(w(H)) for w in (m.W_Q, m.W_K, m.W_V))
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=True)
        b, _, s, _ = out.shape
        return m.W_O(out.transpose(1, 2).reshape(b, s, m.embedding_size))

    assert torch.equal(att(x), old_forward(att, x))
    out, present = att(x, use_cache=True)
    assert torch.allclose(out, att(x), atol=1e-6)
    assert present[0].shape == (2, 4, 9, 8)
    # the model's normal forward returns plain logits, and gradients still flow
    model = make_model(CONFIGS[1])
    model.train()
    logits = model(torch.randint(0, VOCAB, (2, 8)))
    assert isinstance(logits, torch.Tensor) and logits.shape == (2, 8, VOCAB)
    logits.sum().backward()
    assert model.token_embedding.weight.grad is not None


# ---------------------------------------------------------------- generation

PIECES = ["<pad>", "<unk>", "<bos>", "<eos>", "\n", "User", ":", " the", " a", " cat", " sat", "Source", " on",
          " mat", ".", " and", "x", "y", "Question", "z"]


class Tok:
    pad_id, unk_id, bos_id, eos_id = 0, 1, 2, 3
    vocab_size = VOCAB

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [2] if add_bos else []
        ids += [4 + (ord(c) % (VOCAB - 4)) for c in text]
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(PIECES[i] if i < len(PIECES) else f"<{i}>" for i in ids)


def gen_model(seq_len=24, scale=3.0):
    return make_model(dict(embedding_size=32, hidden_size=64, sequence_length=seq_len, num_layers=2, num_heads=4),
                      seed=3, scale=scale)


def run(model, text, use_cache, **kw):
    torch.manual_seed(1234)
    events = list(generate_stream(model, Tok(), text, mode="raw", use_cache=use_cache, **kw))
    return events


def strip_timing(events):
    out = []
    for e in events:
        e = dict(e)
        for k in ("seconds", "tokens_per_second"):
            e.pop(k, None)
        out.append(e)
    return out


@pytest.mark.parametrize("kw", [
    dict(greedy=True),
    dict(greedy=True, repetition_penalty=1.3),
    dict(greedy=True, no_repeat_ngram_size=2),
    dict(temperature=1.2, top_k=10),
    dict(temperature=1.5, top_k=0, repetition_penalty=1.2, no_repeat_ngram_size=3),
    dict(temperature=1.0, top_k=8, repetition_penalty=1.1, penalize_prompt=True),
])
@pytest.mark.parametrize("text,max_new", [
    ("hello", 10),                      # short prompt, no sliding
    ("hello there my friend", 10),      # prompt + generation just fits / nearly fits
    ("a" * 22, 12),                     # prompt fills the window (22 + bos = 23 = limit - 1), then slides
    ("a" * 60, 15),                     # prompt longer than the window: truncated, then slides
    ("hello", 50),                      # generation alone slides past the window
])
def test_generation_identical_with_and_without_cache(kw, text, max_new):
    model = gen_model()
    a = run(model, text, True, max_new_tokens=max_new, **kw)
    b = run(model, text, False, max_new_tokens=max_new, **kw)
    assert strip_timing(a) == strip_timing(b)
    assert a[-1]["event"] == "done"
    assert a[-1]["generated_tokens"] == b[-1]["generated_tokens"]


def test_stop_strings_and_holdback_identical():
    model = gen_model(seq_len=64, scale=0.5)
    hit = set()
    for seed in range(40):
        outs = []
        for cache in (True, False):
            torch.manual_seed(seed)
            outs.append(strip_timing(list(generate_stream(
                model, Tok(), "x", mode="raw", use_cache=cache, max_new_tokens=40,
                temperature=3.0, top_k=0))))
        assert outs[0] == outs[1]
        hit.add(outs[0][-1]["stop_reason"])
    assert len(hit) >= 1


def test_forced_stop_string_with_scripted_model():
    """A model script-wise producing '\\nUser:' must stop identically (cache on/off)."""
    model = gen_model(seq_len=64, scale=0.5)
    script = [9, 10, 4, 5, 6, 8]  # " cat sat \n User :  a"
    calls = {"n": 0}

    def scripted(logits_fn):
        def wrapper(*a, **k):
            out = logits_fn(*a, **k)
            logits = out[0] if isinstance(out, tuple) else out
            logits = torch.full_like(logits, -5.0)
            idx = min(calls["n"] - 1, len(script) - 1)  # count() already bumped n for this call
            logits[..., script[idx]] = 5.0
            return (logits, out[1]) if isinstance(out, tuple) else logits
        return wrapper

    results = []
    for cache in (True, False):
        calls["n"] = 0
        m = copy.deepcopy(model)
        orig_fwd, orig_fc = m.forward, m.forward_cached

        def count(f):
            def g(*a, **k):
                r = f(*a, **k)
                calls["n"] += 1
                return r
            return g
        m.forward = scripted(count(orig_fwd))
        m.forward_cached = scripted(count(orig_fc))
        results.append(strip_timing(list(generate_stream(
            m, Tok(), "x", mode="raw", use_cache=cache, max_new_tokens=20, greedy=True))))
    assert results[0] == results[1]
    assert results[0][-1]["stop_reason"] == "new_block"
    assert results[0][-1]["text"] == "cat sat"


def test_env_switch_disables_cache(monkeypatch):
    model = gen_model()
    calls = {"cached": 0}
    orig = model.forward_cached

    def spy(*a, **k):
        calls["cached"] += 1
        return orig(*a, **k)
    model.forward_cached = spy
    generate(model, Tok(), "hi", mode="raw", max_new_tokens=3, greedy=True)
    assert calls["cached"] == 3
    calls["cached"] = 0
    monkeypatch.setenv("ELVARN_NO_KV_CACHE", "1")
    generate(model, Tok(), "hi", mode="raw", max_new_tokens=3, greedy=True)
    assert calls["cached"] == 0


def test_cache_only_used_while_text_fits_window():
    """After the window is full the cache is dropped (positions would shift): forward_cached is not
    called with a past that would overflow, and the full forward is used instead."""
    model = gen_model(seq_len=16)
    lengths = []
    orig = model.forward_cached

    def spy(ids, past=None, last_only=False):
        lengths.append((0 if past is None else past[0][0].size(2)) + ids.size(1))
        return orig(ids, past=past, last_only=last_only)
    model.forward_cached = spy
    generate(model, Tok(), "hi", mode="raw", max_new_tokens=30, greedy=True, use_cache=True)
    assert lengths and max(lengths) <= 16
