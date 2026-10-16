"""The store sync program: the cross-program lock, the heartbeat, and the collector's own pass stepping aside while the program lives."""
import os
import time

from backend.pipeline import store_sync


def test_two_passes_never_hold_the_lock_together(tmp_path):
    lock = tmp_path / "x.lock"
    with store_sync.pass_lock(lock) as first:
        assert first is True
        with store_sync.pass_lock(lock) as second:
            assert second is False                                   # another pass is running: this one must not start
    with store_sync.pass_lock(lock) as again:
        assert again is True                                         # released when the first pass ended


def test_heartbeat_says_whether_the_program_is_alive(tmp_path):
    beat = tmp_path / "hb"
    assert not store_sync.alive(beat)
    store_sync.beat(beat)
    assert store_sync.alive(beat)
    old = time.time() - 600
    os.utime(beat, (old, old))
    assert not store_sync.alive(beat)                                # a program that died leaves an old heartbeat: the collector takes the work back


def test_the_collectors_pass_steps_aside_while_the_program_is_alive(monkeypatch):
    from backend.pipeline.person_collector import StoreSync
    calls = []
    monkeypatch.setattr(store_sync, "do_pass", lambda registry, say: calls.append("pass"))
    monkeypatch.setattr(store_sync, "alive", lambda *a, **k: True)
    StoreSync(say=lambda m: None).once()
    assert calls == []
    monkeypatch.setattr(store_sync, "alive", lambda *a, **k: False)
    StoreSync(say=lambda m: None).once()
    assert calls == ["pass"]
