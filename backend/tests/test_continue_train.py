import hashlib
import json
import math
import shutil
from collections import Counter

import numpy as np
import pytest
import torch

import continue_train
from conftest import CORPUS

PAD = 0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream(n):
    """A stream where token i has the unique id i+1 (never 0 = PAD), so a token tells us where it came from."""
    return (np.arange(n) + 1).astype(np.uint16)


def pairs_of_epoch(n, seq_len, seed):
    """Run one epoch of the real windowing + batching code and return every (input -> target) pair it trained on,
    as stream indices i meaning 'token i predicts token i+1'."""
    data = stream(n)
    starts, lengths = continue_train.epoch_windows(n, seq_len, np.random.default_rng(seed))
    seen = []
    for b0 in range(0, len(starts), 5):                               # batches of 5 windows
        x, y = continue_train.build_batch(data, starts[b0:b0 + 5], lengths[b0:b0 + 5], seq_len, PAD)
        for xr, yr in zip(x.tolist(), y.tolist()):
            for xi, yi in zip(xr, yr):
                if yi == PAD:                                          # padded position: ignored by the loss
                    continue
                assert xi != PAD
                seen.append((xi - 1, yi - 1))                          # back to stream indices
    return seen


# ------------------------------------------------------------ alignment
def test_each_input_token_predicts_the_very_next_token_of_the_stream():
    for xi, yi in pairs_of_epoch(1000, 32, seed=1):
        assert yi == xi + 1                                           # target = the token right after the input


def test_window_inputs_and_targets_are_the_text_and_the_text_shifted_by_one():
    data = stream(500)
    x, y = continue_train.build_batch(data, np.array([100, 0]), np.array([32, 7]), 32, PAD)
    assert x[0].tolist() == list(range(101, 133)) and y[0].tolist() == list(range(102, 134))     # full window
    assert x[1, :7].tolist() == list(range(1, 8)) and y[1, :7].tolist() == list(range(2, 9))     # short head window
    assert (x[1, 7:] == PAD).all() and (y[1, 7:] == PAD).all()                                    # padding on both sides


def test_padded_targets_are_ignored_by_the_loss():
    import train_batch
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 11)
    y = torch.tensor([[3, 5, PAD, PAD]])
    full = torch.nn.functional.cross_entropy(logits[0, :2], y[0, :2])
    assert torch.isclose(train_batch.lm_loss(logits, y, PAD), full, atol=1e-6)


# ------------------------------------------------------------ coverage
@pytest.mark.parametrize("n,seq", [(1001, 32), (1000, 32), (129, 128), (130, 128), (33, 32), (50, 64), (2, 8), (65, 64)])
def test_every_pair_is_trained_exactly_once_per_epoch(n, seq):
    """No pair skipped, none duplicated - for many random offsets, including awkward sizes (N-1 divisible by the
    window length, shorter than one window, only two tokens...)."""
    for seed in range(40):
        seen = pairs_of_epoch(n, seq, seed)
        counts = Counter(i for i, _ in seen)
        assert sorted(counts) == list(range(n - 1)), f"seed {seed}: some pairs were skipped"
        assert set(counts.values()) == {1}, f"seed {seed}: some pairs were used twice"
        assert all(j == i + 1 for i, j in seen)


def test_large_stream_with_your_real_sizes_covers_all_301447_pairs_once():
    n, seq = 301448, 128                                              # your real train.bin size and window length
    covered = np.zeros(n - 1, dtype=np.int32)
    for seed in range(60):                                            # 60 different random offsets
        covered[:] = 0
        starts, lengths = continue_train.epoch_windows(n, seq, np.random.default_rng(seed))
        for s, L in zip(starts, lengths):
            assert 1 <= L <= seq and s >= 0 and s + L + 1 <= n        # inputs AND shifted targets stay inside the data
            covered[s:s + L] += 1
        assert covered.min() == 1 and covered.max() == 1, f"seed {seed}: min {covered.min()} max {covered.max()}"
        assert int(lengths.sum()) == n - 1


def test_no_pair_is_favoured_over_many_epochs():
    """The old windowing trained on the first/last pairs of the stream in only ~1 epoch in 128. Now every pair is
    covered in every epoch, so over many epochs every pair has the same count."""
    n, seq, epochs = 700, 64, 300
    total = np.zeros(n - 1, dtype=np.int64)
    for seed in range(epochs):
        starts, lengths = continue_train.epoch_windows(n, seq, np.random.default_rng(seed))
        for s, L in zip(starts, lengths):
            total[s:s + L] += 1
    assert (total == epochs).all(), f"uneven: min {total.min()} max {total.max()} (expected {epochs} for every pair)"


def test_order_and_offset_change_every_epoch():
    rng = np.random.default_rng(0)
    a = continue_train.epoch_windows(5000, 64, rng)
    b = continue_train.epoch_windows(5000, 64, rng)
    assert not np.array_equal(a[0], b[0])                              # a new order (and a new offset) every epoch
    assert not np.array_equal(a[0], np.sort(a[0]))                     # shuffled, not left in stream order


def test_too_short_a_stream_is_rejected():
    with pytest.raises(ValueError):
        continue_train.epoch_windows(1, 8, np.random.default_rng(0))


# ------------------------------------------------------------ full run on a tiny real checkpoint
@pytest.fixture()
def setup(tmp_path, tok, tok_file, ckpt_dir):
    shutil.copyfile(tok_file, ckpt_dir / "tokenizer.json")
    ids = []
    for text in CORPUS:
        ids += tok.encode(text, add_bos=True, add_eos=True)
    data = np.asarray(ids, dtype=np.uint16)
    cut = int(len(data) * 0.9)
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    data[:cut].tofile(data_dir / "train.bin")
    data[cut:].tofile(data_dir / "val.bin")
    (data_dir / "meta.json").write_text(json.dumps({"tokenizer_sha256": tok.sha256}))
    return {"ckpt": ckpt_dir / "final.pt", "data": data_dir, "train_tokens": cut, "out": tmp_path / "continued"}


def args_for(s, **over):
    a = ["--init-from", str(s["ckpt"]), "--data-dir", str(s["data"]), "--out-dir", str(s["out"]),
         "--batch-size", "4", "--epochs", "2", "--log-every", "0", "--threads", "2", "--warmup-steps", "2"]
    for k, v in over.items():
        a += [f"--{k.replace('_', '-')}", str(v)]
    return continue_train.parse_args(a)


def test_true_epochs_one_update_per_batch_no_accumulation(setup, monkeypatch):
    updates, window_counts = [], []
    real_step = torch.optim.AdamW.step
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *a, **k: (updates.append(1), real_step(self, *a, **k))[1])
    real_windows = continue_train.epoch_windows
    monkeypatch.setattr(continue_train, "epoch_windows",
                        lambda n, t, rng, order="shuffled": (lambda r: (window_counts.append(len(r[0])), r)[1])(real_windows(n, t, rng, order)))
    before = sha(setup["ckpt"])
    res = continue_train.run(args_for(setup, patience=99))

    assert [h["epoch"] for h in res["history"]] == [1, 2]
    assert [h["batches"] for h in res["history"]] == [math.ceil(w / 4) for w in window_counts]   # windows grouped by batch size
    assert all(h["pairs_trained"] == setup["train_tokens"] - 1 for h in res["history"])          # EVERY pair, every epoch
    assert res["steps"] == sum(h["batches"] for h in res["history"])
    assert len(updates) == res["steps"]                               # exactly one optimizer update per batch (no accumulation)
    assert sha(setup["ckpt"]) == before                               # the original checkpoint was not touched
    log = (setup["out"] / "train_log.txt").read_text()
    assert "EPOCH 1/2" in log and "train loss" in log and "val loss" in log and "best checkpoint" in log


def test_validation_tokens_never_reach_training(tmp_path, tok, tok_file, ckpt_dir):
    """Train on ids < 150 only; the validation file uses ids >= 150. If any validation token reached a training
    batch, the spy would see it."""
    shutil.copyfile(tok_file, ckpt_dir / "tokenizer.json")
    assert tok.vocab_size > 200
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "processed"; data_dir.mkdir()
    rng.integers(10, 150, 1500).astype(np.uint16).tofile(data_dir / "train.bin")
    rng.integers(150, 200, 400).astype(np.uint16).tofile(data_dir / "val.bin")
    (data_dir / "meta.json").write_text(json.dumps({"tokenizer_sha256": tok.sha256}))
    seen = set()
    real_build = continue_train.build_batch

    def spy(*a, **k):
        x, y = real_build(*a, **k)
        seen.update(x.flatten().tolist()); seen.update(y.flatten().tolist())
        return x, y
    import pytest as _p
    mp = _p.MonkeyPatch(); mp.setattr(continue_train, "build_batch", spy)
    try:
        continue_train.run(continue_train.parse_args(["--init-from", str(ckpt_dir / "final.pt"), "--data-dir", str(data_dir),
            "--out-dir", str(tmp_path / "o"), "--batch-size", "4", "--epochs", "2", "--log-every", "0", "--threads", "2", "--patience", "99"]))
    finally:
        mp.undo()
    assert seen and max(seen) < 150, f"a validation-only token reached training: {sorted(x for x in seen if x >= 150)[:5]}"


def test_saves_new_checkpoint_only_when_validation_improves(setup):
    res = continue_train.run(args_for(setup, patience=99, learning_rate=3e-3, epochs=3))
    assert any(h["improved"] for h in res["history"])
    assert res["best_path"].is_file() and (setup["out"] / "latest.json").is_file()
    ck = torch.load(res["best_path"], weights_only=False)
    assert ck["best_val_loss"] == pytest.approx(res["best_val"]) and ck["best_val_loss"] < res["start_val"]
    assert ck["tokenizer_sha256"] and ck["config"]["sequence_length"] == 32


def test_early_stop_after_two_epochs_without_improvement(setup):
    res = continue_train.run(args_for(setup, epochs=5, patience=2, learning_rate=0.0))
    assert len(res["history"]) == 2
    assert not any(h["improved"] for h in res["history"])
    assert res["best_path"] is None and not (setup["out"] / "best.pt").exists()
    assert "No epoch beat the starting checkpoint" in (setup["out"] / "train_log.txt").read_text()


# ------------------------------------------------------------ refusals
def test_refuses_dataset_built_with_a_different_tokenizer(setup):
    (setup["data"] / "meta.json").write_text(json.dumps({"tokenizer_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="DIFFERENT tokenizer"):
        continue_train.run(args_for(setup))
    assert not setup["out"].exists()


def test_refuses_to_overwrite_an_existing_output_folder(setup):
    setup["out"].mkdir()
    (setup["out"] / "precious.txt").write_text("keep me")
    with pytest.raises(SystemExit, match="already exists"):
        continue_train.run(args_for(setup))
    assert (setup["out"] / "precious.txt").read_text() == "keep me"


def test_refuses_checkpoint_without_its_tokenizer(setup):
    (setup["ckpt"].parent / "tokenizer.json").unlink()
    with pytest.raises(SystemExit, match="tokenizer.json"):
        continue_train.run(args_for(setup))


def test_refuses_training_data_containing_the_pad_id(setup):
    d = np.fromfile(setup["data"] / "train.bin", dtype=np.uint16)
    d[10] = PAD
    d.tofile(setup["data"] / "train.bin")
    with pytest.raises(SystemExit, match="PAD"):
        continue_train.run(args_for(setup))
