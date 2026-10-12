"""Extending the context window must keep every old weight and leave short inputs exactly unchanged."""
import json
import sys
from pathlib import Path

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "training"))

import continue_train  # noqa: E402
import extend_context as ec  # noqa: E402
from model.transformer import build_model, count_parameters, model_from_checkpoint  # noqa: E402

V = 60
CONFIG = {"embedding_size": 32, "hidden_size": 64, "sequence_length": 32, "num_layers": 2, "num_heads": 4, "dropout": 0.0, "tie_weights": True}


def checkpoint(tok_sha="x"):
    torch.manual_seed(0)
    model = build_model(V, CONFIG)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return {"model_state_dict": model.state_dict(), "config": dict(CONFIG), "vocab_size": V, "tokenizer_sha256": tok_sha}, model


def run(model, ids):
    model.eval()
    with torch.no_grad():
        return model(ids)


def test_short_inputs_are_bit_identical_and_all_weights_copied():
    ck, old = checkpoint()
    new, cfg = ec.extend_context(ck, 96)
    assert cfg["sequence_length"] == 96 and new.sequence_length == 96
    ids = torch.randint(0, V, (2, 30), generator=torch.Generator().manual_seed(1))
    assert torch.equal(run(old, ids), run(new, ids))                    # exactly equal, not just close
    full = torch.randint(0, V, (1, 32), generator=torch.Generator().manual_seed(2))
    assert torch.equal(run(old, full), run(new, full))
    old_sd, new_sd = old.state_dict(), new.state_dict()
    for key, value in old_sd.items():
        if key == "position_embedding.weight":
            assert torch.equal(new_sd[key][:32], value) and new_sd[key].shape == (96, 32)
        else:
            assert torch.equal(new_sd[key], value), key
    assert count_parameters(new) - count_parameters(old) == 64 * 32


def test_longer_inputs_now_work_and_are_finite_and_causal():
    ck, old = checkpoint()
    new, _ = ec.extend_context(ck, 96)
    ids = torch.randint(0, V, (1, 96), generator=torch.Generator().manual_seed(3))
    with pytest.raises(ValueError):
        run(old, ids)
    out = run(new, ids)
    assert out.shape == (1, 96, V) and torch.isfinite(out).all()
    changed = ids.clone()
    changed[0, 90] = (changed[0, 90] + 1) % V                           # a later token must not change earlier outputs
    assert torch.equal(run(new, changed)[0, :90], out[0, :90])


def test_tile_init_copies_rows_plus_small_noise_and_random_init_is_small():
    table = torch.randn(8, 5)
    t = ec.extend_positions(table, 20, "tile", noise=0.01)
    assert torch.equal(t[:8], table)
    assert (t[8:16] - table).abs().max() < 0.08 and (t[8:16] != table).any()
    assert (t[16:] - table[:4]).abs().max() < 0.08                       # wraps around
    r = ec.extend_positions(table, 20, "random")
    assert torch.equal(r[:8], table) and r[8:].abs().max() < 0.2
    assert torch.equal(ec.extend_positions(table, 8), table)            # same length: unchanged
    assert torch.equal(ec.extend_positions(table, 20, seed=1), ec.extend_positions(table, 20, seed=1))


def test_cannot_shrink_or_bad_init():
    ck, _ = checkpoint()
    with pytest.raises(ValueError):
        ec.extend_context(ck, 16)
    with pytest.raises(ValueError):
        ec.extend_context(ck, 64, init="stretch")


def test_script_writes_new_folder_loadable_by_trainer_and_refuses_overwrite(tmp_path, tok):
    src = tmp_path / "src"
    src.mkdir()
    ck, old = checkpoint()
    ck["vocab_size"], ck["tokenizer_sha256"] = tok.vocab_size, tok.sha256
    ck["model_state_dict"] = build_model(tok.vocab_size, CONFIG).state_dict()
    torch.save(ck, src / "best.pt")
    (src / "tokenizer.json").write_bytes(tok.path.read_bytes())
    before = (src / "best.pt").read_bytes()
    out = tmp_path / "ext"
    ec.main(["--init-from", str(src / "best.pt"), "--out-dir", str(out), "--sequence-length", "64"])
    assert (out / "tokenizer.json").read_bytes() == (src / "tokenizer.json").read_bytes()
    latest = json.loads((out / "latest.json").read_text())
    assert latest["checkpoint"] == "extended.pt" and latest["sequence_length_after"] == 64
    new_ck = torch.load(out / "extended.pt", weights_only=False)
    for key in ("model_state_dict", "config", "vocab_size", "tokenizer_sha256", "step", "status"):
        assert key in new_ck
    model, cfg = model_from_checkpoint(new_ck)
    assert cfg["sequence_length"] == 64
    meta = {"tokenizer_sha256": tok.sha256}
    assert continue_train.check_checkpoint(new_ck, out / "extended.pt", meta).sha256 == tok.sha256   # the trainer accepts it
    assert (src / "best.pt").read_bytes() == before                      # the source is untouched
    with pytest.raises(SystemExit):                                       # never overwrites
        ec.main(["--init-from", str(src / "best.pt"), "--out-dir", str(out)])
    with pytest.raises(SystemExit):                                       # cannot shrink
        ec.main(["--init-from", str(src / "best.pt"), "--out-dir", str(tmp_path / "small"), "--sequence-length", "16"])
    assert not (tmp_path / "small" / "extended.pt").exists()


def test_service_and_search_budget_follow_the_models_real_length(tmp_path, tok, tok_file):
    from api.model_service import ModelService
    from search.prompt import build_evidence_text
    from search.service import Source
    ck, _ = checkpoint()
    ck["vocab_size"], ck["tokenizer_sha256"] = tok.vocab_size, tok.sha256
    ck["model_state_dict"] = build_model(tok.vocab_size, CONFIG).state_dict()
    new, cfg = ec.extend_context(ck, 96)
    d = tmp_path / "m"
    d.mkdir()
    torch.save({**ck, "model_state_dict": new.state_dict(), "config": cfg, "step": 1}, d / "e.pt")
    (d / "latest.json").write_text(json.dumps({"checkpoint": "e.pt", "status": "completed"}))
    svc = ModelService(d, tok_file)
    info = svc.load()
    assert info["sequence_length"] == 96
    count, limit = svc.prompt_budget()
    assert limit == 96
    srcs = [Source("t", "https://a.com", "a.com", "word " * 60) for _ in range(3)]
    small, _ = build_evidence_text("What is X?", srcs, lambda s: len(s.split()), 32, 8)
    big, used = build_evidence_text("What is X?", srcs, lambda s: len(s.split()), 96, 8)
    assert len(big.split()) > len(small.split()) and len(used) == 3
    out = svc.generate("word " * 200, max_new_tokens=3, greedy=True)
    assert 32 < out["prompt_tokens"] <= 95                               # a longer prompt is kept now
