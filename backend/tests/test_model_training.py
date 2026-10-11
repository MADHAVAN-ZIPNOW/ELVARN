import math

import numpy as np
import pytest
import torch

import train_batch
from conftest import TINY_CONFIG, write_checkpoint
from model.attention import CausalSelfAttention
from model.transformer import build_model, count_parameters


def make(tok):
    return build_model(tok.vocab_size, TINY_CONFIG)


def test_forward_shape_and_vocab_dims(tok):
    model = make(tok)
    assert model.token_embedding.num_embeddings == tok.vocab_size
    assert model.lm_head.out_features == tok.vocab_size
    assert model.position_embedding.num_embeddings == TINY_CONFIG["sequence_length"]
    logits = model(torch.randint(0, tok.vocab_size, (3, 10)))
    assert logits.shape == (3, 10, tok.vocab_size)


def test_tied_weights_counted_once(tok):
    model = make(tok)
    assert model.lm_head.weight is model.token_embedding.weight
    untied = build_model(tok.vocab_size, {**TINY_CONFIG, "tie_weights": False})
    assert count_parameters(untied) - count_parameters(model) == tok.vocab_size * 32


def test_rejects_sequence_longer_than_context(tok):
    with pytest.raises(ValueError, match="exceeds"):
        make(tok)(torch.zeros(1, TINY_CONFIG["sequence_length"] + 1, dtype=torch.long))


def test_causal_masking_future_tokens_do_not_leak(tok):
    model = make(tok).eval()
    a = torch.randint(4, tok.vocab_size, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 7) % tok.vocab_size        # change only the future
    with torch.no_grad():
        la, lb = model(a), model(b)
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)      # past unaffected
    assert not torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-5)  # future differs


def test_attention_heads_must_divide_embedding():
    with pytest.raises(ValueError):
        CausalSelfAttention(30, num_heads=4)


def test_initial_loss_is_near_log_vocab(tok):
    torch.manual_seed(0)
    model = make(tok).eval()
    x = torch.randint(4, tok.vocab_size, (8, 16))
    y = torch.randint(4, tok.vocab_size, (8, 16))
    loss = train_batch.lm_loss(model(x), y, tok.pad_id).item()
    assert abs(loss - math.log(tok.vocab_size)) < 1.0


def test_pad_targets_are_ignored_in_loss(tok):
    torch.manual_seed(0)
    model = make(tok).eval()
    x = torch.randint(4, tok.vocab_size, (2, 8))
    y = torch.randint(4, tok.vocab_size, (2, 8))
    y_padded = y.clone()
    y_padded[:, 4:] = tok.pad_id
    logits = model(x)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :4].reshape(-1, logits.size(-1)), y[:, :4].reshape(-1))
    assert torch.isclose(train_batch.lm_loss(logits, y_padded, tok.pad_id), expected, atol=1e-6)


def test_one_training_batch_backward_updates_weights(tok):
    torch.manual_seed(0)
    model = make(tok).train()
    data = np.random.default_rng(0).integers(4, tok.vocab_size, 500).astype(np.uint16)
    x, y = train_batch.get_batch(data, 4, 16, np.random.default_rng(1))
    assert torch.equal(x[:, 1:], y[:, :-1])            # targets are inputs shifted by one
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = model.token_embedding.weight.detach().clone()
    loss = train_batch.lm_loss(model(x), y, tok.pad_id)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    assert not torch.equal(before, model.token_embedding.weight)


def test_training_reduces_loss(tmp_path, tok):
    _, losses = write_checkpoint(tmp_path, tok, steps=40)
    assert losses[-1] < losses[0] - 0.5


def test_checkpoint_roundtrip_gives_identical_logits(tmp_path, tok):
    model, _ = write_checkpoint(tmp_path, tok)
    ck = torch.load(tmp_path / "final.pt", weights_only=False)
    assert ck["tokenizer_sha256"] == tok.sha256 and ck["vocab_size"] == tok.vocab_size
    clone = build_model(ck["vocab_size"], ck["config"])
    clone.load_state_dict(ck["model_state_dict"])
    x = torch.randint(0, tok.vocab_size, (2, 9))
    assert torch.allclose(model.eval()(x), clone.eval()(x), atol=1e-6)


def test_validation_loss_runs_over_whole_split(tok):
    model = make(tok)
    data = np.random.default_rng(0).integers(4, tok.vocab_size, 400).astype(np.uint16)
    assert math.isfinite(train_batch.evaluate(model, data, 16, 4, tok.pad_id))
