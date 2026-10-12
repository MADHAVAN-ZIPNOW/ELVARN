"""--order sequential: each epoch walks the training text from the first token to the last, every next-token pair
once, with NO shuffling and NO random offset."""
import numpy as np
import pytest
import torch

import continue_train
from test_continue_train import args_for, setup  # noqa: F401  (setup is a fixture)
from test_continue_train_epochs import load, same_weights

PAD = 0


def stream(n):
    return (np.arange(n) + 1).astype(np.uint16)          # token i has id i+1 (never 0 = PAD)


SIZES = [(1001, 32), (1000, 32), (129, 128), (130, 128), (33, 32), (50, 64), (2, 8), (65, 64), (301448, 128)]


# ------------------------------------------------------------ the windows
@pytest.mark.parametrize("n,seq", SIZES)
def test_windows_run_from_the_first_token_to_the_last_in_order(n, seq):
    starts, lengths = continue_train.epoch_windows(n, seq, np.random.default_rng(0), "sequential")
    assert starts[0] == 0                                              # begins at the very first token
    assert (np.diff(starts) > 0).all()                                 # strictly ascending: not shuffled
    assert (starts[1:] == starts[:-1] + lengths[:-1]).all()            # each window starts exactly where the last one ended
    assert starts[-1] + lengths[-1] == n - 1                           # the last window ends at the last valid pair
    assert int(lengths.sum()) == n - 1                                 # every pair, once
    assert (lengths[:-1] == seq).all() and 1 <= lengths[-1] <= seq     # all full windows, then at most one short tail


def test_with_your_real_sizes_the_epoch_is_2356_windows_in_74_batches():
    starts, lengths = continue_train.epoch_windows(301448, 128, np.random.default_rng(0), "sequential")
    assert len(starts) == 2356 and starts[0] == 0 and starts[-1] == 301440 and lengths[-1] == 7
    assert -(-len(starts) // 32) == 74                                 # batch size 32 -> 74 optimizer steps per epoch
    assert starts[:3].tolist() == [0, 128, 256]


def test_no_random_numbers_are_used_at_all():
    rng = np.random.default_rng(123)
    before = rng.bit_generator.state
    a = continue_train.epoch_windows(5000, 64, rng, "sequential")
    assert rng.bit_generator.state == before                           # the random generator was never touched
    b = continue_train.epoch_windows(5000, 64, np.random.default_rng(999), "sequential")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])   # any seed -> identical windows
    again = continue_train.epoch_windows(5000, 64, rng, "sequential")
    assert np.array_equal(a[0], again[0])                              # and identical in every epoch


def test_shuffled_mode_is_unchanged_and_the_default():
    rng = np.random.default_rng(0)
    a = continue_train.epoch_windows(5000, 64, rng)
    assert not np.array_equal(a[0], np.sort(a[0]))                     # default is still shuffled
    with pytest.raises(ValueError, match="order"):
        continue_train.epoch_windows(5000, 64, rng, "backwards")


# ------------------------------------------------------------ alignment + coverage through the real batch builder
@pytest.mark.parametrize("n,seq", [s for s in SIZES if s[0] < 5000])
def test_pairs_are_trained_in_exact_stream_order_each_once(n, seq):
    data = stream(n)
    starts, lengths = continue_train.epoch_windows(n, seq, np.random.default_rng(0), "sequential")
    seen = []
    for b0 in range(0, len(starts), 5):
        x, y = continue_train.build_batch(data, starts[b0:b0 + 5], lengths[b0:b0 + 5], seq, PAD)
        for xr, yr in zip(x.tolist(), y.tolist()):
            seen += [(xi - 1, yi - 1) for xi, yi in zip(xr, yr) if yi != PAD]
    assert [i for i, _ in seen] == list(range(n - 1))                  # pair 0, 1, 2 ... N-2: in order, none skipped, none repeated
    assert all(j == i + 1 for i, j in seen)                            # each input token predicts the very next token


# ------------------------------------------------------------ whole training runs
def test_every_epoch_is_identical_and_uses_no_randomness(setup):
    spy = []
    real = continue_train.epoch_windows
    continue_train.epoch_windows = lambda n, t, rng, order="shuffled": (lambda r: (spy.append(r), r)[1])(real(n, t, rng, order))
    try:
        res = continue_train.run(args_for(setup, epochs=3, patience=0, order="sequential"))
    finally:
        continue_train.epoch_windows = real
    assert len(spy) == 3 and all(np.array_equal(spy[0][0], s[0]) for s in spy)          # same windows, same order, every epoch
    assert all(h["pairs_trained"] == setup["train_tokens"] - 1 for h in res["history"])  # every pair every epoch
    e1, e3 = load(setup["out"] / "epoch_1.pt"), load(setup["out"] / "epoch_3.pt")
    assert e1["rng_state"] == e3["rng_state"]                                           # the numpy generator never advanced
    assert (setup["out"] / "final.pt").is_file() and "SEQUENTIALLY" in (setup["out"] / "train_log.txt").read_text()


def test_sequential_resume_gives_the_same_result_as_an_uninterrupted_run(setup, tmp_path):
    a, b = dict(setup, out=tmp_path / "A"), dict(setup, out=tmp_path / "B")
    continue_train.run(args_for(a, epochs=3, patience=0, threads=1, order="sequential"))
    assert continue_train.run(args_for(b, epochs=3, patience=0, threads=1, order="sequential", max_epochs_this_run=1))["paused"]
    resume = args_for(b, epochs=3, patience=0, threads=1, order="sequential")
    resume.resume = True
    continue_train.run(resume)
    assert same_weights(load(a["out"] / "final.pt")["model_state_dict"], load(b["out"] / "final.pt")["model_state_dict"])


def test_resume_refuses_a_different_window_order(setup):
    continue_train.run(args_for(setup, epochs=3, patience=0, order="sequential", max_epochs_this_run=1))
    other = args_for(setup, epochs=3, patience=0, order="shuffled")
    other.resume = True
    with pytest.raises(SystemExit, match="Settings differ"):
        continue_train.run(other)


def test_runs_saved_before_the_order_option_existed_still_resume_as_shuffled(setup):
    continue_train.run(args_for(setup, epochs=2, patience=0, max_epochs_this_run=1))               # a default (shuffled) run
    p = setup["out"] / "epoch_1.pt"
    ck = load(p); del ck["settings"]["order"]; torch.save(ck, p)                                   # make it look like an old checkpoint
    resume = args_for(setup, epochs=2, patience=0)
    resume.resume = True
    assert continue_train.run(resume)["history"][-1]["epoch"] == 2
