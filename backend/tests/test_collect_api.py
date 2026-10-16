"""The owner-only /api/collect/* routes and the monitor behind them. A stopped Kafka / Spark / Neo4j / Chroma must show up as 'unavailable', never as an error.
Everything runs against a temporary registry; the real stores are never touched. People here are FICTIONAL."""
import time

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from backend.pipeline import control, monitor as monitor_module
from backend.pipeline.monitor import Monitor
from backend.pipeline.registry import Registry

_REAL_KAFKA = Monitor.kafka                      # the fixture below replaces it; the lag test needs the real one
OWNER = {"X-Owner-Key": "owner-key-with-16plus-chars"}


@pytest.fixture()
def monitor(tmp_path, monkeypatch):
    for name in ("NEO4J_URI", "CHROMA_HOST", "CHROMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    reg = Registry(tmp_path / "c.sqlite")
    for i, (name, prio) in enumerate((("Ila Marudhan", 0), ("Kavin Selvarasan", 2), ("Nila Thirunavukkarasu", 2))):
        reg.upsert_person({"key": f"wd:Q{i}", "name": name, "party": "Fictional Party", "first_year": 1960 + i, "last_year": 1970 + i,
                           "positions": ["MLA"], "priority": prio, "en_title": name})
    reg.add_document({"doc_id": "doc_a", "person_key": "wd:Q0", "topic": "corruption", "source": "web", "url": "https://x/a", "title": "A case",
                      "raw_path": "kb_raw/web/d/doc_a.html", "sha256": "aa", "bytes": 100, "tokens": 25})
    reg.stage_seen("doc_a", "collected")
    reg.stage_seen("doc_a", "raw")
    (tmp_path / "kb_raw" / "web" / "d").mkdir(parents=True)
    (tmp_path / "kb_raw" / "web" / "d" / "doc_a.html").write_bytes(b"x" * 100)
    (tmp_path / "kb_raw" / "ab").mkdir()
    (tmp_path / "kb_raw" / "ab" / "old.pdf").write_bytes(b"y" * 10)
    mon = Monitor(registry=reg, raw_dir=tmp_path / "kb_raw", corpus_dir=tmp_path / "corpus", control_path=tmp_path / "control.json")
    monkeypatch.setattr(Monitor, "kafka", lambda self, stats: {"up": False, "topic": "kb.raw_documents", "messages": None, "consumer_lag": None})
    monkeypatch.setattr(Monitor, "spark", lambda self, stats: {"up": False, "processed_documents": 0, "skipped": 0, "failed": 0, "dead_letters": 0})
    return mon


@pytest.fixture()
def client(ckpt_dir, monitor):
    app = create_app(ckpt_dir, load_on_start=False, max_concurrent=0, owner_key=OWNER["X-Owner-Key"], collect_service=monitor)
    return TestClient(app)


def test_everything_needs_the_owner_key(client):
    for method, path in (("get", "/api/collect/status"), ("post", "/api/collect/start"), ("post", "/api/collect/stop"), ("get", "/api/collect/people"),
                         ("get", "/api/collect/person/wd:Q0")):
        assert getattr(client, method)(path).status_code in (403, 429)                           # 429 once the address has failed too often (see the throttle test)
        assert getattr(client, method)(path, headers={"X-Owner-Key": "wrong-key-wrong-key-wrong"}).status_code in (403, 429)


def test_wrong_keys_are_throttled(client):
    codes = [client.get("/api/collect/status", headers={"X-Owner-Key": "wrong-key-wrong-key-wrong"}).status_code for _ in range(10)]
    assert codes[:8] == [403] * 8 and codes[8:] == [429, 429]
    assert client.get("/api/collect/status", headers=OWNER).status_code == 429                  # even the right key waits: the address is locked for a while


def test_status_has_the_contract_shape_and_degrades_stage_by_stage(client):
    s = client.get("/api/collect/status", headers=OWNER).json()
    assert s["state"] == "idle" and s["running"] is False and s["scope"] == {"from_year": 1947, "to_year": 2026}
    assert s["people"]["total"] == 3 and s["people"]["pending"] == 3 and s["people"]["done"] == 0
    assert [q["name"] for q in s["people"]["queue"]][0] == "Ila Marudhan"                  # chief ministers first
    st = s["stages"]
    assert set(st) == {"collector", "raw", "kafka", "spark", "corpus", "neo4j", "chroma"}
    assert st["collector"]["documents"] == 1 and st["collector"]["bytes"] == 100
    assert st["raw"]["files"] == 2 and st["raw"]["by_source"]["web"] == 1 and st["raw"]["by_source"]["pdf"] == 1 and st["raw"]["free_disk_bytes"] > 0
    assert st["kafka"]["up"] is False and st["kafka"]["messages"] is None            # unavailable, not an error
    assert st["neo4j"]["up"] is False and st["neo4j"]["passages"] is None
    assert st["chroma"]["up"] is False and st["chroma"]["embedded_pct"] is None
    assert st["corpus"]["target_tokens"] == 1_000_000_000 and st["corpus"]["clean_tokens"] == 0
    assert [m["stage"] for m in s["movement"]] == ["raw", "collected"] or {m["stage"] for m in s["movement"]} == {"collected", "raw"}


def test_spark_stats_lines_feed_the_corpus_numbers_and_the_lag(monitor, tmp_path, monkeypatch):
    stats = tmp_path / "corpus" / "_stats"
    stats.mkdir(parents=True)
    (stats / "2026-10-16.jsonl").write_text(
        '{"processed": 3, "skipped": 1, "failed": 0, "passages": 40, "clean_tokens": 9000, "duplicates_removed": 2, "doc_ids": ["doc_a"], "offsets": {"0": 7}}\n'
        'not json\n'
        '{"processed": 2, "skipped": 0, "failed": 1, "passages": 10, "clean_tokens": 1000, "duplicates_removed": 0, "offsets": {"0": 9, "1": 4}}\n', encoding="utf-8")
    lines = monitor.stats_lines()
    assert len(lines) == 2
    corpus = monitor.corpus(lines)
    assert corpus["passages"] == 50 and corpus["clean_tokens"] == 10000 and corpus["duplicates_removed"] == 2
    monitor.observe(lines)
    assert {"spark", "corpus"} <= set(monitor.registry.stages_of(["doc_a"])["doc_a"])        # the document was seen moving through Spark
    marks = {0: (0, 20), 1: (0, 10)}
    monitor.cache.items["kafka"] = (time.time(), ("kb.raw_documents", marks))
    out = _REAL_KAFKA(monitor, lines)
    assert out["messages"] == 30 and out["consumer_lag"] == (20 - 9) + (10 - 4)


def test_start_and_stop_go_through_the_control_file(client, monitor, monkeypatch):
    calls = []
    monkeypatch.setattr(control, "start", lambda *a, **k: calls.append("start") or control.write({"state": "running", "pid": None, "started_at": control._now()}, monitor.control_path))
    s = client.post("/api/collect/start", headers=OWNER).json()
    assert calls == ["start"] and s["state"] == "running" and s["running"] is True and s["started_at"]
    stopped = client.post("/api/collect/stop", headers=OWNER).json()
    assert stopped["state"] == "stopping" and stopped["running"] is True               # graceful: still finishing the current document


def test_people_paging_filtering_and_detail(client):
    page = client.get("/api/collect/people?limit=2", headers=OWNER).json()
    assert page["total"] == 3 and len(page["items"]) == 2 and page["items"][0]["name"] == "Ila Marudhan"
    assert page["items"][0]["corruption_docs"] == 1 and page["items"][0]["docs"] == 1
    assert client.get("/api/collect/people?q=Kavin", headers=OWNER).json()["total"] == 1
    assert client.get("/api/collect/people?status=done", headers=OWNER).json()["total"] == 0
    detail = client.get("/api/collect/person/wd:Q0", headers=OWNER).json()
    assert detail["person"]["name"] == "Ila Marudhan" and [d["doc_id"] for d in detail["documents"]] == ["doc_a"]
    assert client.get("/api/collect/person/wd:Q404", headers=OWNER).status_code == 404


def test_targeted_request_resolves_the_person_and_starts_only_those_sources(client, monitor, monkeypatch):
    started = []
    monkeypatch.setattr(control, "start", lambda path=None, **kw: started.append(kw) or control.write(
        {"state": "running", "pid": None, "started_at": control._now(), "message": kw.get("message", "")}, monitor.control_path))
    r = client.post("/api/collect/targeted", headers=OWNER, json={"text": "collect Ila Marudhan ED cases"}).json()
    req = r["request"]                                                                       # the plan travels next to the usual status document
    assert req["ok"] is True and req["person"]["key"] == "wd:Q0" and req["stages"] == ["ed_records"] and req["sources"] == ["ed"]
    assert started[0]["extra"] == ("--person", "wd:Q0", "--sources", "ed") and "Ila Marudhan" in started[0]["message"]
    assert r["state"] == "running"                                                           # the usual status document comes with it
    again = client.post("/api/collect/targeted", headers=OWNER, json={"text": "collect Ila Marudhan ED"}).json()
    assert again["request"]["ok"] is False and "running" in again["request"]["message"] and len(started) == 1       # one collection at a time


def test_targeted_request_rejects_unknown_empty_and_unauthorised_requests(client):
    nobody = client.post("/api/collect/targeted", headers=OWNER, json={"text": "collect Selvarasan Thirunavukkarasu cbi"}).json()
    assert nobody["request"]["ok"] is False and not nobody["request"].get("needs_choice")                           # two different people, no common name: nothing matches
    empty = client.post("/api/collect/targeted", headers=OWNER, json={"text": "  "}).json()
    assert empty["request"]["ok"] is False and "name" in empty["request"]["message"].lower()
    assert client.post("/api/collect/targeted", json={"text": "x"}).status_code == 403
