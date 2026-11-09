"""The service runs its own schedule so calls and e-mails do not depend on an outside job. No database: run_tick is replaced."""
import asyncio
import threading
import time
from types import SimpleNamespace

from app import jobs, selftick


def make_app():
    return SimpleNamespace(state=SimpleNamespace(service="svc", mailbox="box", pusher="phone"))


def test_it_ticks_every_interval_and_survives_a_failed_tick(monkeypatch):
    calls, naps = [], []

    def fake_tick(svc, mailbox, pusher=None):
        calls.append((svc, mailbox, pusher))
        if len(calls) == 1:
            raise RuntimeError("database asleep")
        return {}

    async def nap(seconds):
        naps.append(seconds)

    monkeypatch.setattr(jobs, "run_tick", fake_tick)
    asyncio.run(selftick.run(make_app(), every=300, sleep=nap, rounds=3))
    assert calls == [("svc", "box", "phone")] * 3 and naps == [30, 300, 300, 300]


def test_ticks_never_run_at_the_same_time(monkeypatch):
    """The service, the owner's PC and outside timers can all tick at once: only one may run, or a call could ring twice."""
    active, worst = [0], [0]

    def slow_body(svc, mailbox, now=None, pusher=None):
        active[0] += 1
        worst[0] = max(worst[0], active[0])
        time.sleep(0.05)
        active[0] -= 1
        return {}

    monkeypatch.setattr(jobs, "_run_tick", slow_body)
    threads = [threading.Thread(target=jobs.run_tick, args=(None, None)) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert worst[0] == 1
