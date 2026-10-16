"""Pressing Start on the DEPLOYED site: the site (public mode) passes the owner's button presses to the agent on the owner's computer and shows what the agent reports.
Offline: the agent talks to the site through the test client. People are FICTIONAL."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from backend.pipeline import control, remote_ops
from backend.pipeline.agent import Agent
from backend.pipeline.bridge import CollectBridge
from backend.pipeline.monitor import Monitor
from backend.pipeline.registry import Registry

KEY = "owner-key-with-16plus-chars"
OWNER = {"X-Owner-Key": KEY}


# ---- the bridge on its own ------------------------------------------------------------------------------------------------
def test_offline_until_an_agent_reports_then_online_then_offline_again():
    now = {"t": 1000.0}
    bridge = CollectBridge(stale_after=20, clock=lambda: now["t"])
    s = bridge.status()
    assert s["state"] == "offline" and s["agent"]["online"] is False and s["people"]["total"] is None and s["stages"]["neo4j"] is None
    bridge.poll({"state": "running", "running": True, "message": "Collecting", "people": {"total": 5}, "stages": {}, "movement": []})
    s = bridge.status()
    assert s["state"] == "running" and s["agent"]["online"] is True and s["people"]["total"] == 5
    now["t"] += 30                                                                          # the agent stopped reporting
    s = bridge.status()
    assert s["state"] == "offline" and s["running"] is False and "stopped reporting" in s["message"] and s["people"]["total"] == 5


def test_commands_wait_for_the_agent_expire_and_are_answered_once():
    now = {"t": 0.0}
    bridge = CollectBridge(clock=lambda: now["t"])
    a = bridge.enqueue("start")
    b = bridge.enqueue("stop")
    assert [c["seq"] for c in bridge.poll()] == [a, b] and bridge.poll() == []              # handed over once
    bridge.enqueue("start")
    now["t"] += 200                                                                          # nobody came for it
    assert bridge.poll() == []
    seq = bridge.enqueue("people", {"q": "x"})
    bridge.poll(results=[{"seq": seq, "result": {"items": []}}])
    assert bridge.results[seq]["result"] == {"items": []}


def test_a_question_is_answered_by_the_agent_within_the_wait_or_gives_none():
    bridge = CollectBridge()
    assert bridge.call("targeted", {"text": "x"}, wait=0.3) is None
    bridge.poll()                                                                           # the unanswered command is still waiting: clear it

    def agent():
        for _ in range(50):
            for c in bridge.poll():
                bridge.poll(results=[{"seq": c["seq"], "result": {"ok": True, "echo": c["args"]}}])
                return
            time.sleep(0.05)
    t = threading.Thread(target=agent)
    t.start()
    assert bridge.call("targeted", {"text": "hello"}, wait=5) == {"ok": True, "echo": {"text": "hello"}}
    t.join()


# ---- the hosted site (public mode) -------------------------------------------------------------------------------------------
@pytest.fixture()
def site(ckpt_dir):
    return TestClient(create_app(ckpt_dir, load_on_start=False, public=True, owner_key=KEY, max_concurrent=0))


def test_the_public_site_needs_the_owner_key_for_everything_collect(site):
    for method, path in (("get", "/api/collect/status"), ("post", "/api/collect/start"), ("post", "/api/collect/stop"), ("get", "/api/collect/people"),
                         ("post", "/api/collect/agent/poll")):
        r = getattr(site, method)(path, **({"json": {}} if method == "post" else {}))
        assert r.status_code in (403, 422, 429)
    assert site.get("/api/collect/status").status_code in (403, 429)
    assert site.get("/api/collect/status", headers=OWNER).status_code in (200, 429)


def test_without_an_agent_the_dashboard_says_offline_and_commands_are_not_queued(site):
    s = site.get("/api/collect/status", headers=OWNER).json()
    assert s["state"] == "offline" and s["agent"]["online"] is False and "agent" in s["message"].lower()
    assert site.post("/api/collect/start", headers=OWNER).json()["state"] == "offline"
    r = site.post("/api/collect/targeted", headers=OWNER, json={"text": "collect Ila Marudhan ED"}).json()
    assert r["request"]["ok"] is False and "offline" in r["request"]["message"]
    assert site.post("/api/collect/agent/poll", headers=OWNER, json={"report": None, "results": []}).json() == {"commands": []}          # nothing was queued


@pytest.fixture()
def local(tmp_path, monkeypatch):
    """The owner's computer: a registry, a Monitor, and the agent."""
    for name in ("NEO4J_URI", "CHROMA_HOST", "CHROMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    reg = Registry(tmp_path / "c.sqlite")
    for i, (name, prio) in enumerate((("Ila Marudhan", 0), ("Kavin Selvarasan", 2))):
        reg.upsert_person({"key": f"wd:Q{i}", "name": name, "party": "Fictional Party", "first_year": 1960 + i, "last_year": 1970 + i,
                           "positions": ["MLA"], "priority": prio, "en_title": name})
    monkeypatch.setattr(Monitor, "kafka", lambda self, stats: {"up": False, "topic": "kb.raw_documents", "messages": None, "consumer_lag": None})
    monkeypatch.setattr(Monitor, "spark", lambda self, stats: {"up": False, "processed_documents": 0, "skipped": 0, "failed": 0, "dead_letters": 0})
    started = []
    monkeypatch.setattr(control, "start", lambda path=None, **kw: started.append(kw) or control.write(
        {"state": "running", "pid": None, "started_at": control._now(), "message": kw.get("message", "Collecting")}, path))
    monkeypatch.setattr(control, "request_stop", lambda path=None: control.write({"state": "stopping", "stop_requested": True}, path))
    monitor = Monitor(registry=reg, raw_dir=tmp_path / "kb_raw", corpus_dir=tmp_path / "corpus", control_path=tmp_path / "control.json")
    return monitor, started


def agent_for(site, monitor):
    def post(payload):
        r = site.post("/api/collect/agent/poll", headers=OWNER, json=payload)
        assert r.status_code == 200, r.text
        return r.json()
    return Agent("https://site.example", KEY, monitor, post=post, say=lambda m: None)


def test_start_pressed_on_the_site_starts_the_collector_on_the_owners_computer(site, local):
    monitor, started = local
    agent = agent_for(site, monitor)
    agent.step()                                                                           # the agent reports: the site is online
    s = site.get("/api/collect/status", headers=OWNER).json()
    assert s["agent"]["online"] is True and s["state"] == "idle" and s["people"]["total"] == 2
    assert site.post("/api/collect/start", headers=OWNER).json()["message"].startswith("Start sent")
    assert agent.step() == 1                                                               # the agent picks the command up and runs it
    assert len(started) == 1
    agent.step()                                                                           # and reports the new state
    assert site.get("/api/collect/status", headers=OWNER).json()["state"] == "running"
    site.post("/api/collect/stop", headers=OWNER)
    assert agent.step() == 1
    agent.step()
    assert site.get("/api/collect/status", headers=OWNER).json()["state"] == "stopping"


def test_a_typed_request_and_the_roster_come_back_through_the_agent(site, local):
    monitor, started = local
    agent = agent_for(site, monitor)
    agent.step()
    stop = threading.Event()

    def run_agent():
        while not stop.is_set():
            agent.step()
            time.sleep(0.05)
    t = threading.Thread(target=run_agent)
    t.start()
    try:
        r = site.post("/api/collect/targeted", headers=OWNER, json={"text": "collect Ila Marudhan ED cases"}).json()
        assert r["request"]["ok"] is True and r["request"]["person"]["name"] == "Ila Marudhan" and r["request"]["stages"] == ["ed_records"]
        assert started[0]["extra"] == ("--person", "wd:Q0", "--sources", "ed")
        page = site.get("/api/collect/people?q=Kavin", headers=OWNER).json()
        assert page["total"] == 1 and page["items"][0]["name"] == "Kavin Selvarasan"
        assert site.get("/api/collect/person/wd:Q1", headers=OWNER).json()["person"]["name"] == "Kavin Selvarasan"
        assert site.get("/api/collect/person/wd:Q404", headers=OWNER).status_code == 404
    finally:
        stop.set()
        t.join()


# ---- the agent ---------------------------------------------------------------------------------------------------------------------
def test_the_agent_refuses_a_weak_key_and_survives_a_failing_command(local):
    monitor, _ = local
    with pytest.raises(ValueError):
        Agent("https://site.example", "short", monitor)
    assert remote_ops.run(monitor, "nonsense", {})["error"].startswith("unknown command")
    assert "error" in remote_ops.run(monitor, "people", {"limit": "not a number"})                  # a bad argument is an answer, not a crash
    sent = []
    agent = Agent("https://site.example", KEY, monitor, post=lambda p: sent.append(p) or {"commands": [{"seq": 7, "action": "stop", "args": {}}]}, say=lambda m: None)
    assert agent.step() == 1 and agent.answers == [{"seq": 7, "result": {"ok": True}}]
    agent.post = lambda p: sent.append(p) or {"commands": []}
    agent.step()
    assert sent[-1]["results"] == [{"seq": 7, "result": {"ok": True}}] and agent.answers == []     # the answer travels with the next post, once
