"""Tests for: all epochs with no early stopping, a resumable checkpoint after every epoch, --resume, and
--optimizer restore (true continuation of the checkpoint's own optimizer and original learning-rate schedule)."""
from types import SimpleNamespace

import pytest
import torch

import continue_train
import train_batch
from test_continue_train import args_for, setup, sha  # noqa: F401  (setup is a fixture)


def load(path):
    return torch.load(path, weights_only=False)


def same_weights(a, b, atol=1e-6):
    return a.keys() == b.keys() and all(torch.allclose(a[k], b[k], atol=atol) for k in a)


# ------------------------------------------------------------ no early stopping
def test_no_early_stopping_runs_every_epoch_and_saves_each_one(setup):
    res = continue_train.run(args_for(setup, epochs=4, patience=0, learning_rate=0.0))      # lr 0: validation can never improve
    assert [h["epoch"] for h in res["history"]] == [1, 2, 3, 4]                             # all four ran, none skipped
    assert not any(h["improved"] for h in res["history"])
    for n in range(1, 5):
        assert (setup["out"] / f"epoch_{n}.pt").is_file()
    assert (setup["out"] / "final.pt").is_file()
    log = (setup["out"] / "train_log.txt").read_text()
    assert "Early stopping     : OFF" in log and "Early stop:" not in log
    assert all(f"EPOCH {n}/4" in log for n in range(1, 5))


def test_every_epoch_logs_train_and_validation_loss(setup):
    res = continue_train.run(args_for(setup, epochs=3, patience=0))
    log = (setup["out"] / "train_log.txt").read_text()
    for h in res["history"]:
        assert f"EPOCH {h['epoch']}/3" in log and f"train loss {h['train_loss']:.4f}" in log and f"val loss {h['val_loss']:.4f}" in log
    saved = __import__("json").loads((setup["out"] / "epochs.json").read_text())
    assert [(h["epoch"], h["train_loss"], h["val_loss"]) for h in saved] == [(h["epoch"], h["train_loss"], h["val_loss"]) for h in res["history"]]


# ------------------------------------------------------------ resumable checkpoint after each epoch
def test_each_epoch_checkpoint_holds_everything_needed_to_resume(setup, tok):
    res = continue_train.run(args_for(setup, epochs=3, patience=0))
    for n in (1, 2, 3):
        ck = load(setup["out"] / f"epoch_{n}.pt")
        assert ck["epoch"] == n and ck["tokenizer_sha256"] == tok.sha256 and ck["vocab_size"] == tok.vocab_size
        assert ck["config"]["sequence_length"] == 32 and ck["config"]["num_heads"] == 2
        for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "torch_rng_state", "settings", "history", "step"):
            assert key in ck, key
        assert len(ck["history"]) == n and ck["step"] == res["history"][n - 1]["steps_total"]
        assert ck["status"] == ("completed" if n == 3 else "in_progress")
        assert ck["optimizer_state_dict"]["state"], "the optimizer's moving averages must be saved"
    final, last = load(setup["out"] / "final.pt"), load(setup["out"] / "epoch_3.pt")
    assert final["epoch"] == 3 and final["status"] == "completed" and same_weights(final["model_state_dict"], last["model_state_dict"], 0)


def test_originals_are_never_touched(setup):
    before = sha(setup["ckpt"])
    continue_train.run(args_for(setup, epochs=2, patience=0))
    assert sha(setup["ckpt"]) == before
    assert not (setup["ckpt"].parent / "epoch_1.pt").exists()


# ------------------------------------------------------------ resume
def test_resume_gives_the_same_result_as_an_uninterrupted_run(setup, tmp_path):
    a, b = dict(setup, out=tmp_path / "A"), dict(setup, out=tmp_path / "B")
    continue_train.run(args_for(a, epochs=3, patience=0, threads=1))                         # uninterrupted

    paused = continue_train.run(args_for(b, epochs=3, patience=0, threads=1, max_epochs_this_run=1))   # "interrupted" after epoch 1
    assert paused["paused"] and (b["out"] / "epoch_1.pt").is_file() and not (b["out"] / "final.pt").exists()
    resume = args_for(b, epochs=3, patience=0, threads=1)
    resume.resume = True
    continue_train.run(resume)

    fa, fb = load(a["out"] / "final.pt"), load(b["out"] / "final.pt")
    assert fb["epoch"] == 3 and fa["step"] == fb["step"]
    assert same_weights(fa["model_state_dict"], fb["model_state_dict"])                      # identical weights
    for ha, hb in zip(fa["history"], fb["history"]):
        assert ha["train_loss"] == pytest.approx(hb["train_loss"], abs=1e-6) and ha["val_loss"] == pytest.approx(hb["val_loss"], abs=1e-6)
    assert "RESUMING" in (b["out"] / "train_log.txt").read_text() and "EPOCH 1/3" in (b["out"] / "train_log.txt").read_text()


def test_resume_refuses_different_settings(setup):
    continue_train.run(args_for(setup, epochs=3, patience=0, max_epochs_this_run=1))
    changed = args_for(setup, epochs=3, patience=0, batch_size=8)
    changed.resume = True
    with pytest.raises(SystemExit, match="Settings differ"):
        continue_train.run(changed)
    assert not (setup["out"] / "epoch_2.pt").exists()                                       # nothing was trained


def test_resume_needs_an_existing_run(setup):
    nothing = args_for(setup)
    nothing.resume = True
    with pytest.raises(SystemExit, match="Nothing to resume"):
        continue_train.run(nothing)


# ------------------------------------------------------------ optimizer: restore vs fresh
def test_restore_continues_the_original_optimizer_and_original_schedule(setup):
    start = load(setup["ckpt"])
    assert start["step"] == 15
    res = continue_train.run(args_for(setup, epochs=2, patience=0, optimizer="restore",
                                      original_lr=3e-3, original_warmup=2, original_total_steps=60))
    ns = SimpleNamespace(learning_rate=3e-3, warmup_steps=2, max_steps=60)
    b1 = res["history"][0]["batches"]
    e1 = load(setup["out"] / "epoch_1.pt")
    assert e1["step"] == 15 + b1                                                              # the step counter carries on
    assert {int(v["step"]) for v in e1["optimizer_state_dict"]["state"].values()} == {15 + b1}  # Adam's moments were NOT restarted
    last_lr = train_batch.lr_at(15 + b1 - 1, ns)                                              # the original run's own schedule function
    assert res["history"][0]["lr_last_update"] == pytest.approx(last_lr)
    assert e1["optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(last_lr)
    assert f"lr for the first update = {train_batch.lr_at(15, ns):.6f}" in (setup["out"] / "train_log.txt").read_text()


def test_fresh_optimizer_starts_counting_from_zero(setup):
    res = continue_train.run(args_for(setup, epochs=1, patience=0, optimizer="fresh"))
    e1 = load(setup["out"] / "epoch_1.pt")
    assert {int(v["step"]) for v in e1["optimizer_state_dict"]["state"].values()} == {res["history"][0]["batches"]}
    assert e1["step"] == res["history"][0]["batches"]


def test_restore_is_refused_without_optimizer_state(setup):
    ck = load(setup["ckpt"]); del ck["optimizer_state_dict"]; torch.save(ck, setup["ckpt"])   # (a temp copy, not a real checkpoint)
    with pytest.raises(SystemExit, match="optimizer state"):
        continue_train.run(args_for(setup, optimizer="restore"))
    assert not setup["out"].exists()


def test_restore_is_refused_when_step_and_optimizer_disagree(setup):
    ck = load(setup["ckpt"]); ck["step"] = 999; torch.save(ck, setup["ckpt"])
    with pytest.raises(SystemExit, match="not a clean continuation"):
        continue_train.run(args_for(setup, optimizer="restore"))
    assert not setup["out"].exists()
