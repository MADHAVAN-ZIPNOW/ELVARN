import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from api.model_service import ModelNotReady, ModelService
from model.attention import CausalSelfAttention
from model.transformer import (DEFAULT_CONFIG, PRESETS, build_model, config_from_checkpoint, count_parameters,
                               model_from_checkpoint, preset_config)

E = 64
HEADS = [1, 2, 4, 8]


def reference_attention(attn, H):
    """Slow, obvious causal multi-head attention written separately for each head, using the module's own
    weights. head h owns rows h*d:(h+1)*d of W_Q, W_K, W_V (its own projections)."""
    B, T, _ = H.shape
    d = attn.head_size
    heads = []
    for h in range(attn.num_heads):
        sl = slice(h * d, (h + 1) * d)
        q = H @ attn.W_Q.weight[sl].T + attn.W_Q.bias[sl]
        k = H @ attn.W_K.weight[sl].T + attn.W_K.bias[sl]
        v = H @ attn.W_V.weight[sl].T + attn.W_V.bias[sl]
        scores = q @ k.transpose(-2, -1) / math.sqrt(d)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)      # True = future position
        scores = scores.masked_fill(mask, float("-inf"))
        heads.append(torch.softmax(scores, dim=-1) @ v)
    return torch.cat(heads, dim=-1) @ attn.W_O.weight.T + attn.W_O.bias         # concatenate, then output projection


def make(heads, seed=0):
    torch.manual_seed(seed)
    return CausalSelfAttention(E, num_heads=heads).eval()


# ------------------------------------------------------------------ shapes + head size
@pytest.mark.parametrize("heads", HEADS)
def test_attention_output_shape_is_unchanged(heads):
    attn = make(heads)
    out = attn(torch.randn(3, 11, E))
    assert out.shape == (3, 11, E)
    assert attn.head_size == E // heads and attn.num_heads == heads


@pytest.mark.parametrize("heads", HEADS)
def test_split_heads_gives_batch_heads_seq_headsize(heads):
    attn = make(heads)
    q = attn._split_heads(attn.W_Q(torch.randn(2, 7, E)))
    assert q.shape == (2, heads, 7, E // heads)


def test_model_logits_shape_is_unchanged_for_any_head_count(tok):
    for heads in (1, 2, 4):
        cfg = {**preset_config("tiny"), "embedding_size": 32, "hidden_size": 64, "sequence_length": 16, "num_heads": heads}
        logits = build_model(tok.vocab_size, cfg)(torch.randint(0, tok.vocab_size, (3, 9)))
        assert logits.shape == (3, 9, tok.vocab_size)                       # next-token training shapes are unchanged


def test_parameter_count_does_not_depend_on_head_count(tok):
    counts = {h: count_parameters(build_model(tok.vocab_size, {"embedding_size": 32, "hidden_size": 64, "sequence_length": 16,
                                                                "num_layers": 2, "num_heads": h, "dropout": 0.0, "tie_weights": True}))
              for h in (1, 2, 4, 8)}
    assert len(set(counts.values())) == 1                                   # heads re-slice the same weights, they add none


# ------------------------------------------------------------------ divisibility error
@pytest.mark.parametrize("bad", [3, 5, 7, 128, 65])
def test_embedding_must_be_divisible_by_num_heads(bad):
    with pytest.raises(ValueError, match=rf"embedding_size \({E}\) must be divisible by num_heads \({bad}\)"):
        CausalSelfAttention(E, num_heads=bad)


@pytest.mark.parametrize("bad", [0, -2, 1.5, "4", None, True])
def test_num_heads_must_be_a_positive_whole_number(bad):
    with pytest.raises(ValueError, match="positive whole number"):
        CausalSelfAttention(E, num_heads=bad)


def test_every_shipped_preset_has_a_valid_head_split():
    for name in PRESETS:
        cfg = preset_config(name)
        assert cfg["embedding_size"] % cfg["num_heads"] == 0, name
    assert DEFAULT_CONFIG["embedding_size"] // DEFAULT_CONFIG["num_heads"] == 32          # tiny: 4 heads x 32
    assert preset_config("7m")["embedding_size"] // preset_config("7m")["num_heads"] == 32  # 7m: 8 heads x 32


# ------------------------------------------------------------------ the maths is right
@pytest.mark.parametrize("heads", HEADS)
def test_matches_a_slow_per_head_reference(heads):
    attn = make(heads)
    x = torch.randn(2, 9, E)
    with torch.no_grad():
        assert torch.allclose(attn(x), reference_attention(attn, x), atol=1e-5)


def test_one_head_equals_the_original_single_head_formula():
    """The first version of ELVARN used one head: scores = QK^T / sqrt(E), causal mask, softmax, @V, then W_O."""
    attn = make(1)
    x = torch.randn(2, 10, E)
    with torch.no_grad():
        q, k, v = attn.W_Q(x), attn.W_K(x), attn.W_V(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(E)
        mask = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
        old = attn.W_O(torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1) @ v)
        assert torch.allclose(attn(x), old, atol=1e-5)


def test_each_head_has_its_own_projections_and_heads_are_independent():
    """Changing the query weights of ONE head changes only that head's slice of the (concatenated) output."""
    attn = make(4)
    with torch.no_grad():                       # make W_O the identity so the concatenated head outputs are visible
        attn.W_O.weight.copy_(torch.eye(E)); attn.W_O.bias.zero_()
    x = torch.randn(2, 9, E)
    before = attn(x).detach().clone()
    d = attn.head_size
    with torch.no_grad():
        attn.W_Q.weight[2 * d:3 * d] += torch.randn(d, E)          # head 2's query projection only
    after = attn(x).detach()
    changed = [(before[..., h * d:(h + 1) * d] - after[..., h * d:(h + 1) * d]).abs().max().item() > 1e-6 for h in range(4)]
    assert changed == [False, False, True, False]


def test_multi_head_is_really_different_from_single_head_with_the_same_weights():
    a1, a4 = make(1), make(4)
    a4.load_state_dict(a1.state_dict())                            # identical weights, different head split
    x = torch.randn(2, 9, E)
    with torch.no_grad():
        assert not torch.allclose(a1(x), a4(x), atol=1e-4)


# ------------------------------------------------------------------ causal masking
@pytest.mark.parametrize("heads", HEADS)
def test_future_tokens_cannot_change_earlier_outputs(heads):
    attn = make(heads)
    a = torch.randn(1, 12, E)
    b = a.clone(); b[0, 7:] = torch.randn(5, E)                    # change only positions 7..11
    with torch.no_grad():
        oa, ob = attn(a), attn(b)
    assert torch.allclose(oa[:, :7], ob[:, :7], atol=1e-6)         # positions 0..6 never saw the change
    assert not torch.allclose(oa[:, 7:], ob[:, 7:], atol=1e-6)


@pytest.mark.parametrize("heads", HEADS)
def test_output_at_position_t_has_exactly_zero_gradient_to_later_positions(heads):
    attn = make(heads)
    x = torch.randn(1, 10, E, requires_grad=True)
    t = 4
    attn(x)[0, t].sum().backward()
    assert x.grad[0, t + 1:].abs().max().item() == 0.0              # nothing at all flows from the future
    assert x.grad[0, :t + 1].abs().max().item() > 0.0               # but it does depend on itself and the past


def test_first_token_attends_only_to_itself():
    attn = make(4)
    x = torch.randn(1, 6, E)
    with torch.no_grad():
        full = attn(x)[0, 0]
        alone = attn(x[:, :1])[0, 0]
    assert torch.allclose(full, alone, atol=1e-6)


# ------------------------------------------------------------------ checkpoints
def tiny_model(tok, heads, tie=True):
    cfg = {"embedding_size": 32, "hidden_size": 64, "sequence_length": 16, "num_layers": 2,
           "num_heads": heads, "dropout": 0.0, "tie_weights": tie}
    torch.manual_seed(1)
    return build_model(tok.vocab_size, cfg).eval(), cfg


def checkpoint_of(model, cfg, tok):
    return {"model_state_dict": model.state_dict(), "config": cfg, "vocab_size": tok.vocab_size,
            "tokenizer_sha256": tok.sha256, "step": 5, "status": "completed"}


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_checkpoint_round_trip_keeps_heads_and_logits(tok, heads, tmp_path):
    model, cfg = tiny_model(tok, heads)
    path = tmp_path / "c.pt"
    torch.save(checkpoint_of(model, cfg, tok), path)
    loaded, cfg2 = model_from_checkpoint(torch.load(path, weights_only=False))
    assert cfg2["num_heads"] == heads
    assert loaded.decoder_blocks[0].attention.num_heads == heads and loaded.decoder_blocks[0].attention.head_size == 32 // heads
    x = torch.randint(0, tok.vocab_size, (2, 9))
    with torch.no_grad():
        assert torch.equal(model(x), loaded.eval()(x))


def test_head_count_is_invisible_in_the_weights_so_it_must_come_from_the_config(tok):
    """Same weights load fine into a model with a different head count (no shape error) but give different answers.
    That is exactly why a checkpoint without num_heads is refused instead of guessed."""
    model, cfg = tiny_model(tok, 4)
    with torch.no_grad():                                            # realistic weights (a fresh model's are ~0, which hides the effect)
        for p_ in model.parameters():
            p_.copy_(torch.randn_like(p_) * 0.4)
    other = build_model(tok.vocab_size, {**cfg, "num_heads": 2}).eval()
    other.load_state_dict(model.state_dict(), strict=True)           # loads without complaint ...
    x = torch.randint(0, tok.vocab_size, (1, 9))
    with torch.no_grad():
        difference = (model(x) - other(x)).abs().max().item()
    assert difference > 1e-2, difference                              # ... but behaves differently


def test_checkpoint_without_num_heads_is_refused_not_guessed(tok):
    model, cfg = tiny_model(tok, 4)
    ck = checkpoint_of(model, {k: v for k, v in cfg.items() if k != "num_heads"}, tok)
    with pytest.raises(ValueError, match="num_heads"):
        config_from_checkpoint(ck)
    with pytest.raises(ValueError, match="cannot be guessed"):
        model_from_checkpoint(ck)


def test_checkpoint_with_incompatible_head_count_is_refused(tok):
    model, cfg = tiny_model(tok, 4)
    with pytest.raises(ValueError, match="must be divisible"):
        model_from_checkpoint(checkpoint_of(model, {**cfg, "num_heads": 5}, tok))


def test_mismatched_weight_shapes_are_never_loaded_silently(tok):
    model, cfg = tiny_model(tok, 4)
    with pytest.raises(RuntimeError):                                # wrong embedding size -> shape error, not a quiet load
        model_from_checkpoint(checkpoint_of(model, {**cfg, "embedding_size": 64}, tok))
    with pytest.raises(RuntimeError):                                # wrong number of layers -> missing/extra keys
        model_from_checkpoint(checkpoint_of(model, {**cfg, "num_layers": 3}, tok))


def test_old_format_checkpoint_loads_as_one_head_and_matches_the_old_model(tok):
    """Before heads existed a checkpoint stored its sizes at the top level, with no config, and used separate
    (untied) output weights. Those load as ONE head."""
    model, cfg = tiny_model(tok, 1, tie=False)
    legacy = {"model_state_dict": model.state_dict(), "vocab_size": tok.vocab_size, "epoch": 5,
              "embedding_size": 32, "hidden_size": 64, "num_layers": 2, "sequence_length": 16}
    loaded, cfg2 = model_from_checkpoint(legacy)
    assert cfg2["num_heads"] == 1 and cfg2["tie_weights"] is False
    x = torch.randint(0, tok.vocab_size, (2, 9))
    with torch.no_grad():
        assert torch.equal(model(x), loaded.eval()(x))


def test_a_file_that_is_not_a_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="does not look like"):
        config_from_checkpoint({"model_state_dict": {}})


REAL_OLD = Path(__file__).resolve().parents[1] / "training" / "checkpoints" / "checkpoint_epoch_5.pt"


@pytest.mark.skipif(not REAL_OLD.is_file(), reason="your original word-level checkpoint is not on this machine")
def test_your_real_old_checkpoint_loads_strictly_as_one_head():
    ck = torch.load(REAL_OLD, map_location="cpu", weights_only=False)      # read only: never written
    model, cfg = model_from_checkpoint(ck)
    assert cfg["num_heads"] == 1 and cfg["embedding_size"] == 128
    assert count_parameters(model) == 2_836_726
    out = model.eval()(torch.randint(0, ck["vocab_size"], (1, 12)))
    assert out.shape == (1, 12, ck["vocab_size"]) and torch.isfinite(out).all()


def test_model_service_refuses_a_checkpoint_with_no_head_count(ckpt_dir, tok_file):
    ck = torch.load(ckpt_dir / "final.pt", weights_only=False)
    del ck["config"]["num_heads"]
    torch.save(ck, ckpt_dir / "final.pt")
    with pytest.raises(ModelNotReady, match="cannot be loaded safely"):
        ModelService(ckpt_dir, tok_file).load()


def test_model_service_reports_head_count_and_size(ckpt_dir, tok_file):
    svc = ModelService(ckpt_dir, tok_file)
    info = svc.load()
    assert (info["num_heads"], info["head_size"]) == (2, 16)           # the tiny test model: 32 dims, 2 heads
