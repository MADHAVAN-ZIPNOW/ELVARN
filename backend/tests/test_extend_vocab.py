"""extend_vocab: common words become single tokens, nothing old is lost (tiny tokenizer and tiny model, no files from the project)."""
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "training"))

import extend_vocab as ev  # noqa: E402
from model.transformer import build_model  # noqa: E402
from tokenizer.bpe_tokenizer import SPECIAL_TOKENS  # noqa: E402

TEXT = ("The petitioner filed a writ petition and the respondent denied the claim. " * 6 + "The appellant appeared before the court on appeal. " * 4 +
        "Notice was issued to the petitioner. " * 3)


def tiny_tokenizer():
    tok = Tokenizer(models.BPE(unk_token="<UNK>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([TEXT[:300], "a b c d e f", "x y z"] * 3, trainers.BpeTrainer(
        vocab_size=170, special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2, show_progress=False))
    return tok


def test_new_words_become_single_tokens_and_old_text_is_unchanged():
    base = tiny_tokenizer()
    base_json = json.loads(base.to_str())
    counter, _ = ev.count_words([TEXT] * 30)
    words = ev.pick_words(counter, base, 3, min_count=5)
    assert "petitioner" in words and all(len(base.encode(" " + w).ids) >= 2 for w in words)
    new_json, steps = ev.extend_tokenizer(base_json, words, base)
    new = Tokenizer.from_str(json.dumps(new_json))
    assert new.get_vocab_size() == base.get_vocab_size() + len(steps) and len(steps) >= 3
    for w in words:
        assert len(new.encode(" " + w).ids) == 1                                   # now ONE token
    assert len(new.encode(TEXT).ids) < len(base.encode(TEXT).ids)                   # the same text is shorter
    assert new.decode(new.encode(TEXT).ids) == base.decode(base.encode(TEXT).ids) == TEXT   # and decodes back exactly
    other = "a b c d e f x y z"                                                     # text with none of the new words: the very same ids
    assert new.encode(other).ids == base.encode(other).ids
    for k, v in base_json["model"]["vocab"].items():
        assert new_json["model"]["vocab"][k] == v                                   # no old id moved


def test_old_logits_are_unchanged_and_new_rows_are_averages():
    base = tiny_tokenizer()
    new_json, steps = ev.extend_tokenizer(json.loads(base.to_str()), ["petitioner", "respondent"], base)
    config = {"embedding_size": 16, "hidden_size": 32, "sequence_length": 12, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "tie_weights": True}
    torch.manual_seed(0)
    old_model = build_model(base.get_vocab_size(), config).eval()
    ck = {"model_state_dict": old_model.state_dict(), "config": config, "vocab_size": base.get_vocab_size()}
    new_model, _ = ev.extend_checkpoint(ck, steps, bias_offset=2.0)
    new_model.eval()
    ids = torch.tensor([[2, 5, 9, 14, 3, 7, 11, 20]])
    with torch.no_grad():
        a, b = old_model(ids), new_model(ids)
    assert b.shape[-1] == a.shape[-1] + len(steps)
    assert torch.allclose(a, b[..., :a.shape[-1]], atol=1e-6, rtol=0)                  # every old token scores the same as before (to floating-point noise)
    nid, ia, ib = steps[0]
    assert torch.allclose(new_model.token_embedding.weight[nid], (old_model.token_embedding.weight[ia] + old_model.token_embedding.weight[ib]) / 2)
    assert torch.equal(new_model.lm_head.weight, new_model.token_embedding.weight)  # still tied
    assert new_model.lm_head.bias[nid] < old_model.lm_head.bias[ia] and new_model.lm_head.bias[nid] < old_model.lm_head.bias[ib] + 0.01
