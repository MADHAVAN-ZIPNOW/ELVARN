"""Growing a model must keep what it learned: same answers when only deeper, nearly the same when wider."""
import sys
from pathlib import Path

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "training"))

import grow_model as gm  # noqa: E402
from model.transformer import build_model, count_parameters  # noqa: E402

V = 60
CONFIG = {"embedding_size": 32, "hidden_size": 64, "sequence_length": 16, "num_layers": 2, "num_heads": 4, "dropout": 0.0, "tie_weights": True}


def trained_like_checkpoint():
    torch.manual_seed(0)
    model = build_model(V, CONFIG)
    with torch.no_grad():
        for p in model.parameters():           # a fresh model has near-zero weights: use real-sized ones so differences would show
            p.copy_(torch.randn_like(p) * 0.3)
        for name, p in model.named_parameters():
            if "layer_norm" in name and name.endswith("weight"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
    return {"model_state_dict": model.state_dict(), "config": dict(CONFIG), "vocab_size": V, "tokenizer_sha256": "x"}, model


def logits(model, ids):
    model.eval()
    with torch.no_grad():
        out = model(ids)
    return out[0] if isinstance(out, tuple) else out


IDS = torch.randint(0, V, (3, 12), generator=torch.Generator().manual_seed(1))


def test_deeper_gives_exactly_the_same_logits():
    ck, old = trained_like_checkpoint()
    grown, cfg, _ = gm.grow(ck, 32, 64, 4, 5)
    assert cfg["num_layers"] == 5 and count_parameters(grown) > count_parameters(old)
    assert torch.allclose(logits(old, IDS), logits(grown, IDS), atol=1e-5)


def test_wider_keeps_the_answers_nearly_the_same():
    ck, old = trained_like_checkpoint()
    grown, cfg, _ = gm.grow(ck, 48, 96, 6, 2)
    assert cfg["embedding_size"] == 48 and cfg["num_heads"] == 6
    a, b = logits(old, IDS), logits(grown, IDS)
    assert (a.log_softmax(-1) - b.log_softmax(-1)).abs().mean() < 0.15


def test_wider_and_deeper_has_the_right_shape_and_tied_embeddings():
    ck, _ = trained_like_checkpoint()
    grown, cfg, _ = gm.grow(ck, 48, 96, 6, 4)
    assert cfg["num_layers"] == 4 and cfg["tie_weights"] is True
    assert grown.lm_head.weight is grown.token_embedding.weight or torch.equal(grown.lm_head.weight, grown.token_embedding.weight)
    assert logits(grown, IDS).shape[-1] == V


def test_old_weights_are_copied_into_the_grown_model():
    ck, old = trained_like_checkpoint()
    grown, _, _ = gm.grow(ck, 48, 96, 6, 3)
    assert torch.equal(grown.token_embedding.weight[:, :32], old.token_embedding.weight)
    assert torch.equal(grown.token_embedding.weight[:, 32:], torch.zeros(V, 16))


def test_cannot_shrink():
    ck, _ = trained_like_checkpoint()
    with pytest.raises(ValueError):
        gm.grow(ck, 16, 32, 2, 2)
    with pytest.raises(ValueError):
        gm.grow(ck, 32, 64, 4, 1)


def test_input_checkpoint_is_not_modified():
    ck, _ = trained_like_checkpoint()
    before = {k: v.clone() for k, v in ck["model_state_dict"].items()}
    gm.grow(ck, 48, 96, 6, 4)
    assert all(torch.equal(before[k], ck["model_state_dict"][k]) for k in before)


# ------------------------------------------------------------------ moving to a new tokenizer
def _tokenizers(tmp_path):
    from tokenizer import train_bpe
    from tokenizer.bpe_tokenizer import BPETokenizer
    text = ["The Government of India audits the accounts of the Union and the States. " * 3, "Panchayati raj is local self government in villages. " * 3] * 20
    small = train_bpe.train(text, 300, out_file=tmp_path / "old" / "tokenizer.json")
    big = train_bpe.train(text, 420, out_file=tmp_path / "new" / "tokenizer.json")
    return BPETokenizer(small), BPETokenizer(big)


def test_new_vocabulary_rows_are_averages_of_the_old_pieces_they_spell(tmp_path):
    old_tok, new_tok = _tokenizers(tmp_path)
    torch.manual_seed(0)
    config = {**CONFIG}
    old = build_model(old_tok.vocab_size, config)
    with torch.no_grad():
        for p in old.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    ck = {"model_state_dict": old.state_dict(), "config": config, "vocab_size": old_tok.vocab_size, "tokenizer_sha256": "x"}
    grown, cfg, vocab = gm.grow(ck, 32, 64, 4, 2, old_tok=old_tok, new_tok=new_tok)
    assert vocab == new_tok.vocab_size and grown.token_embedding.weight.shape[0] == new_tok.vocab_size
    assert torch.equal(grown.lm_head.weight, grown.token_embedding.weight)
    emb, new_emb = old.token_embedding.weight, grown.token_embedding.weight
    for name in ("<PAD>", "<BOS>", "<EOS>", "<HUMOUR>"):                          # special tokens copied by name
        assert torch.equal(new_emb[new_tok.token_id(name)], emb[old_tok.token_id(name)])
    for piece, new_id in new_tok._tok.get_vocab().items():
        if piece.startswith("<") and piece.endswith(">") and piece in ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<DOMAIN_KNOWLEDGE>", "<GENERAL_ENGLISH>", "<CASUAL_CONVERSATION>", "<HUMOUR>"):
            continue
        ids = [t.id for t in old_tok._tok.model.tokenize(piece)]
        assert torch.allclose(new_emb[new_id], emb[ids].mean(0), atol=1e-6)
    assert logits(grown, IDS.clamp(max=new_tok.vocab_size - 1)).shape[-1] == new_tok.vocab_size
