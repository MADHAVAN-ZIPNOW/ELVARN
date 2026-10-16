"""The politician-by-politician collector: registry, raw archive, source relevance, the collection loop, resume, and the official-records matching.
Offline: Wikipedia / Wikidata are replaced by a small fake. Every person, party and case in here is FICTIONAL."""
import json
from pathlib import Path

import pytest

from backend.pipeline import control, official_index, person_sources
from backend.pipeline.common import FetchError, ListSink
from backend.pipeline.person_collector import STAGES, PersonCollector, build_message
from backend.pipeline.person_sources import Found, RawArchive
from backend.pipeline.registry import Registry

PERSON = {"key": "wd:Q9001", "wikidata_id": "Q9001", "name": "Arulmozhi Vendhan", "party": "Fictional Progress Party", "first_year": 1971, "last_year": 1989,
          "positions": ["Chief Minister of Tamil Nadu"], "en_title": "Arulmozhi Vendhan", "ta_title": "அருள்மொழி வேந்தன்", "priority": 0}


def page(body):
    return f"<html><head><title>x</title><script>var a=1;</script></head><body><nav>menu</nav><p>{body}</p></body></html>".encode("utf-8")


OWN_EN = page("Arulmozhi Vendhan was a politician of Tamil Nadu who served as Chief Minister. " * 30)
OWN_TA = page("அருள்மொழி வேந்தன் தமிழ்நாடு அரசியல்வாதி. " * 40)
SCAM = page("In Tamil Nadu the Vigilance case alleged bribery by Arulmozhi Vendhan; the court heard the corruption charge against Arulmozhi Vendhan. " * 12)
PARTY = page("The Fictional Progress Party of Tamil Nadu was founded by Arulmozhi Vendhan. Elsewhere the party was accused of corruption in a scam. " + "Filler text. " * 400)
UNRELATED = page("A cricket match in Chennai, Tamil Nadu. Nothing else here.")
WIKIDATA = json.dumps({"entities": {"Q9001": {"id": "Q9001", "labels": {"en": {"value": "Arulmozhi Vendhan"}},
                                                  "claims": {"P102": [{"mainsnak": {"datavalue": {"value": {"entity-type": "item", "id": "Q5551"}}}}]}}}}).encode()


class FakeWikimedia:
    """Quacks like person_sources.Wikimedia."""

    def __init__(self, fail_first=0):
        self.pages = {("en", "Arulmozhi Vendhan"): OWN_EN, ("ta", PERSON["ta_title"]): OWN_TA, ("en", "Vendhan bribery case"): SCAM,
                      ("en", "Fictional Progress Party"): PARTY, ("en", "Cricket in Chennai"): UNRELATED}
        self.calls, self.fail = [], fail_first

    def article_html(self, lang, title):
        self.calls.append(("html", lang, title))
        if self.fail:
            self.fail -= 1
            raise FetchError("temporary failure")
        data = self.pages.get((lang, title))
        return (data, f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}") if data else None

    def entity_json(self, qid):
        self.calls.append(("entity", qid))
        return WIKIDATA

    def labels(self, qids):
        self.calls.append(("labels", tuple(qids)))
        return {q: f"Label {q}" for q in qids}

    def search(self, query, limit=12):
        self.calls.append(("search", query))
        return ["Arulmozhi Vendhan", "Vendhan bribery case", "Fictional Progress Party", "Cricket in Chennai"]


@pytest.fixture()
def env(tmp_path):
    reg = Registry(tmp_path / "c.sqlite")
    reg.upsert_person(PERSON)
    sink = ListSink()
    archive = RawArchive(tmp_path / "kb_raw", today=lambda: "2026-10-16")
    return reg, sink, archive, tmp_path


def collector(env, wm=None, **kw):
    reg, sink, archive, _ = env
    return PersonCollector(reg, wm or FakeWikimedia(), archive, sink, say=lambda m: None, sleep=lambda s: None, **kw)


# ---- registry ----------------------------------------------------------------------------------------------------------------------
def test_roster_order_chief_ministers_first_and_resume_before_new(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    for key, name, prio, year in (("a", "Aa", 2, 1952), ("b", "Bb", 0, 1969), ("c", "Cc", 1, 1960), ("d", "Dd", 0, 1950)):
        reg.upsert_person({"key": key, "name": name, "priority": prio, "first_year": year})
    assert [reg.claim_next()["key"] for _ in range(1)] == ["d"]                  # priority 0, earliest year
    assert reg.claim_next()["key"] == "d"                                       # still in progress: resumed, not skipped
    reg.finish_person("d")
    assert [reg.claim_next()["key"], ][0] == "b"
    reg.finish_person("b")
    assert reg.claim_next()["key"] == "c"
    reg.finish_person("c")
    assert reg.claim_next()["key"] == "a"
    reg.finish_person("a")
    assert reg.claim_next() is None and reg.counts()["done"] == 4


def test_same_bytes_are_registered_once_and_linked_to_other_people(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    reg.upsert_person({"key": "p1", "name": "One"})
    reg.upsert_person({"key": "p2", "name": "Two"})
    d = {"doc_id": "doc_1", "person_key": "p1", "topic": "corruption", "source": "web", "url": "u", "title": "t", "raw_path": "kb_raw/web/x/doc_1.html",
         "sha256": "abc", "bytes": 5, "tokens": 10}
    assert reg.add_document(d) is True
    assert reg.add_document({**d, "doc_id": "doc_2"}) is False
    reg.link("p2", "doc_1", "corruption")
    reg.link("p2", "doc_1", "corruption")                                        # linking twice counts once
    assert reg.person("p1")["corruption_docs"] == 1 and reg.person("p2")["corruption_docs"] == 1
    assert [x["doc_id"] for x in reg.documents_of("p2")] == ["doc_1"]


# ---- raw archive -------------------------------------------------------------------------------------------------------------------
def test_raw_archive_layout_checksum_and_never_rewrites(tmp_path):
    archive = RawArchive(tmp_path / "kb_raw", today=lambda: "2026-10-16")
    doc_id, sha, raw_path, size, created = archive.save(b"hello original", "web", "html")
    assert doc_id == "doc_" + sha[:20] and created and size == 14
    assert raw_path == f"kb_raw/web/2026-10-16/{doc_id}.html"
    assert (tmp_path / "kb_raw" / "web" / "2026-10-16" / f"{doc_id}.html").read_bytes() == b"hello original"
    again = archive.save(b"hello original", "web", "html")
    assert again[0] == doc_id and again[4] is False                              # the file is not written twice
    lines = (tmp_path / "kb_raw" / "checksums.txt").read_text().splitlines()
    assert lines == [f"{sha}  web/2026-10-16/{doc_id}.html"]
    assert archive.exists(raw_path)


# ---- relevance ---------------------------------------------------------------------------------------------------------------------
def test_corruption_article_must_be_about_the_person_not_just_name_them():
    assert person_sources.near_corruption(person_sources.visible_text(SCAM), PERSON) >= 3
    assert person_sources.near_corruption(person_sources.visible_text(PARTY), PERSON) < 3
    wm = FakeWikimedia()
    assert person_sources.corruption_doc(wm, PERSON, "Vendhan bribery case") is not None
    assert person_sources.corruption_doc(wm, PERSON, "Fictional Progress Party") is None             # names him once, corruption elsewhere
    assert person_sources.corruption_doc(wm, PERSON, "Cricket in Chennai") is None
    assert person_sources.corruption_doc(wm, PERSON, "No such page") is None


# ---- the loop ----------------------------------------------------------------------------------------------------------------------
def test_one_person_end_to_end(env):
    reg, sink, archive, tmp = env
    assert collector(env).run() == "done"
    assert reg.person(PERSON["key"])["status"] == "done"
    topics = sorted(d["topic"] for d in reg.documents_of(PERSON["key"]))
    assert topics == ["corruption", "person", "person", "structured"]
    assert len(sink.messages) == 4
    for m in sink.messages:                                                      # the contract: a pointer, not the bytes
        assert m["event_type"] == "document_ingested" and m["status"] == "NEW" and m["site"] == "politician"
        assert m["raw_path"].startswith("kb_raw/") and (tmp / m["raw_path"]).is_file()
        assert m["doc_id"] == "doc_" + m["sha256"][:20] and m["file_size_bytes"] == (tmp / m["raw_path"]).stat().st_size
        assert m["person_key"] == PERSON["key"] and m["url"] == m["source_url"]
        assert "content" not in m and "data" not in m
    assert {m["language"] for m in sink.messages} == {"en", "ta"}
    structured = [m for m in sink.messages if m["topic"] == "structured"][0]                  # the names of linked items travel as a second archived file
    assert structured["labels_path"].startswith("kb_raw/dataset/")
    assert json.loads((tmp / structured["labels_path"]).read_text(encoding="utf-8")) == {"for": "Q9001", "labels": {"Q5551": "Label Q5551"}}
    assert (tmp / structured["raw_path"]).read_bytes() == WIKIDATA                               # the original is untouched
    assert all(d["status"] == "published" for d in reg.recent_documents())
    stages = reg.stages_of([m["doc_id"] for m in sink.messages])
    assert all({"collected", "raw", "kafka"} <= set(s) for s in stages.values())


def test_stop_then_resume_does_not_repeat_finished_sources(env):
    reg, sink, _, _ = env
    asked = {"n": 0}

    def stop_after_two_stages():
        asked["n"] += 1
        return len(reg.person(PERSON["key"])["sources_done"]) >= 2

    c = collector(env, should_stop=stop_after_two_stages)
    assert c.run() == "stopped"
    p = reg.person(PERSON["key"])
    assert p["status"] == "in_progress" and set(p["sources_done"]) == {"wikidata", "wikipedia_en"}
    sent = len(sink.messages)
    wm2 = FakeWikimedia()
    assert collector(env, wm2).run() == "done"
    assert ("entity", "Q9001") not in wm2.calls and ("html", "en", "Arulmozhi Vendhan") not in wm2.calls     # finished sources are skipped
    assert len(sink.messages) > sent and reg.person(PERSON["key"])["status"] == "done"


def test_unconfirmed_messages_are_sent_again(env):
    reg, sink, archive, _ = env

    class FlakySink(ListSink):
        fail = True

        def flush(self):
            if self.fail:
                raise RuntimeError("Kafka down")

    flaky = FlakySink()
    c = PersonCollector(reg, FakeWikimedia(), archive, flaky, say=lambda m: None, sleep=lambda s: None)
    assert c.run() == "error" or reg.person(PERSON["key"])["status"] in ("failed", "in_progress")
    assert reg.unpublished()                                                      # saved but never confirmed
    flaky.fail = False
    c2 = PersonCollector(reg, FakeWikimedia(), archive, flaky, say=lambda m: None, sleep=lambda s: None)
    c2.publish_pending()
    assert not reg.unpublished()


def test_network_failures_are_retried_and_a_dead_network_stops_the_run(env):
    reg, sink, _, _ = env
    assert collector(env, FakeWikimedia(fail_first=2)).run() == "done"          # two hiccups, then fine
    assert reg.counter("retries") >= 2
    reg2 = Registry(env[3] / "d.sqlite")
    for i in range(7):
        reg2.upsert_person({**PERSON, "key": f"wd:Q{i}", "name": f"Person {i}", "first_year": i})
    dead = PersonCollector(reg2, FakeWikimedia(fail_first=10_000), env[2], ListSink(), say=lambda m: None, sleep=lambda s: None)
    assert dead.run() == "error"
    assert reg2.counts()["failed"] == 5 and reg2.counts()["pending"] == 2        # stopped after five failures in a row


def test_build_message_schema():
    f = Found("person", "web", "html", "text/html", "https://x/y", "T", "en", "Wikipedia", "CC BY-SA 4.0", "reference", b"abc", wikidata_id="Q1")
    m = build_message(f, PERSON, "doc_1", "s" * 64, "kb_raw/web/d/doc_1.html", 3)
    for key in ("event_type", "doc_id", "source", "source_url", "raw_path", "mime_type", "file_size_bytes", "sha256", "language", "collected_at", "status"):
        assert key in m
    assert m["topic"] == "person" and m["person_name"] == "Arulmozhi Vendhan" and m["status"] == "NEW"
    assert m["aliases"] == ["Arulmozhi Vendhan", "Vendhan"]                                # full name, then the family name (7+ letters)
    short = build_message(f, {**PERSON, "name": "A. Sen", "en_title": "A. Sen"}, "d", "s" * 64, "kb_raw/x", 1)["aliases"]
    assert short == ["A. Sen"]                                                              # a short family name is too ambiguous to be an alias


# ---- official records (courts and agencies): matching ----------------------------------------------------------------------------------
def test_spelling_variants_fold_together_and_titles_are_matched_with_care():
    assert official_index.norm("JAYALALITHAA") == official_index.norm("Jeyalalitha".replace("Je", "Ja"))
    p = {"name": "Arulmozhi Vendhanathan", "en_title": "A. Vendhanathan"}
    ok = lambda t: official_index.candidate_title(official_index.norm(t), p)    # noqa: E731
    assert ok("STATE OF TAMIL NADU versus A. ARULMOZHI VENDHANATHAN")
    assert ok("VENDHANATHAN versus STATE")                                             # no initials written: accepted as a candidate
    assert not ok("S. VENDHANATHAN versus STATE")                                      # different initial: a namesake
    assert not ok("ILANGO VENDHANATHAN versus STATE")                                  # another given name before the family name
    assert not ok("SOMEONE ELSE versus STATE")


def test_text_must_show_a_political_role_next_to_the_name():
    p = {"name": "Arulmozhi Vendhanathan", "en_title": "A. Vendhanathan"}
    yes = official_index.norm("The accused Vendhanathan, then Chief Minister of Tamil Nadu, sanctioned the grant.")
    no = official_index.norm("Vendhanathan sold a plot of land to the respondent in a civil dispute about a boundary wall.")
    assert official_index.confirms_text(yes, p) and not official_index.confirms_text(no, p)


def test_official_index_match_orders_courts_first(tmp_path):
    idx = official_index.OfficialIndex(tmp_path / "i.sqlite")
    idx.add_rows("u1", [
        {"kind": "cbi", "year": 2012, "date": "2012-01-01", "title": "CBI books A. Arulmozhi Vendhanathan", "url": "https://cbi/x"},
        {"kind": "hc", "year": 2015, "date": "2015-05-05", "title": "CRL OP/1/2015 of A.VENDHANATHAN Vs STATE", "url": "https://hc/1", "bucket": "b", "key": "k"},
        {"kind": "sc", "year": 2010, "date": "2010-02-02", "title": "A. ARULMOZHI VENDHANATHAN versus STATE", "url": "https://sc/1", "bucket": "b", "key": "k2"},
        {"kind": "hc", "year": 2015, "date": "2015-06-06", "title": "WP/2/2015 of RAJA Vs STATE", "url": "https://hc/2", "bucket": "b", "key": "k3"}])
    rows = idx.match({"name": "Arulmozhi Vendhanathan", "en_title": "A. Vendhanathan"})
    assert [r["kind"] for r in rows] == ["sc", "hc", "cbi"]
    assert idx.unit_done("u1") and idx.counts() == {"cbi": 1, "hc": 2, "sc": 1}


def test_official_records_stage_links_known_documents_and_downloads_new_ones(env, monkeypatch):
    reg, sink, archive, tmp = env
    idx = official_index.OfficialIndex(tmp / "i.sqlite")
    idx.add_rows("u", [{"kind": "hc", "year": 2015, "date": "2015-05-05", "title": "CRL OP/1/2015 of A.ARULMOZHI VENDHAN Vs STATE", "url": "https://hc/1",
                        "bucket": "b", "key": "k1", "meta": {"judge": "J"}},
                       {"kind": "hc", "year": 2016, "date": "2016-05-05", "title": "CRL OP/2/2016 of A.ARULMOZHI VENDHAN Vs STATE", "url": "https://hc/2",
                        "bucket": "b", "key": "k2"}])
    reg.upsert_person({**PERSON, "key": "wd:Q777", "name": "Another Person"})
    reg.add_document({"doc_id": "doc_known", "person_key": "wd:Q777", "topic": "court_record", "source": "pdf", "url": "https://hc/2", "title": "known",
                      "raw_path": "kb_raw/pdf/d/doc_known.pdf", "sha256": "known", "bytes": 1})

    def fake_official(row, person, s3, fetcher):
        return Found("court_record", "pdf", "pdf", "application/pdf", row["url"], row["title"], "en", "Madras High Court", "CC-BY-4.0", "order",
                     b"%PDF fake " + row["url"].encode(), tokens=50, extra={"published_date": row["date"], **row["meta"]})
    monkeypatch.setattr(person_sources, "official_doc", fake_official)
    c = collector(env, index=idx, s3=object())
    found = c.official_records(PERSON)
    assert [f.url for f in found] == ["https://hc/1"]                              # the other record was already collected: only linked
    assert "doc_known" in {d["doc_id"] for d in reg.documents_of(PERSON["key"])}
    assert {"court_records", "ed_records", "dvac_records", "cbi_records"} <= set(STAGES)


# ---- start / stop state ------------------------------------------------------------------------------------------------------------
def test_the_collector_is_started_without_a_window(monkeypatch, tmp_path):
    """A detached Windows process has no console, so the venv's python launcher opened a visible cmd window for the real interpreter."""
    import subprocess
    seen = {}

    class FakeProc:
        pid = 4242

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: seen.update(kw=kw, cmd=cmd) or FakeProc())
    monkeypatch.setattr(control, "LOG_PATH", tmp_path / "collector.log")
    control._launch(tmp_path / "control.json", None, False, ("--person", "wd:Q1"))
    assert seen["cmd"][-3:] == ["run", "--person", "wd:Q1"]
    options = control.hidden_process_options()
    if control.os.name == "nt":
        flags = seen["kw"]["creationflags"]
        assert flags & subprocess.CREATE_NO_WINDOW and not flags & subprocess.DETACHED_PROCESS
        assert seen["kw"]["startupinfo"].wShowWindow == 0 and options["creationflags"] == flags
    else:
        assert options == {"start_new_session": True}
    assert control.read(tmp_path / "control.json")["pid"] == 4242


def test_every_kafka_client_is_forced_to_ipv4():
    """'localhost' tries ::1 first; the broker listens on IPv4 only, so each connection waited for a refused attempt and status checks randomly timed out."""
    from backend.pipeline import kafka_io
    assert kafka_io.CLIENT_OPTIONS["broker.address.family"] == "v4"
    import inspect
    from backend.pipeline import monitor
    assert "CLIENT_OPTIONS" in inspect.getsource(monitor.Monitor.kafka) and "CLIENT_OPTIONS" in inspect.getsource(control.kafka_up)
    assert "CLIENT_OPTIONS" in inspect.getsource(kafka_io.KafkaSink.__init__) and "CLIENT_OPTIONS" in inspect.getsource(kafka_io.consume)


def test_control_state_survives_a_dead_process_and_stop_is_graceful(tmp_path):
    path = tmp_path / "control.json"
    assert control.state(path)["state"] == "idle"
    control.write({"state": "running", "pid": 99999999, "started_at": "2020-01-01T00:00:00+00:00"}, path)
    assert control.state(path)["state"] == "idle"                                 # the process died without saying so
    import os
    control.write({"state": "running", "pid": os.getpid(), "started_at": "2020-01-01T00:00:00+00:00"}, path)
    assert control.is_running(path)
    assert control.request_stop(path)["state"] == "stopping" and control.stop_requested(path)
    control.write({"state": "running", "pid": None, "started_at": control._now()}, path)
    assert control.state(path)["state"] == "running"                              # still starting Kafka: not mistaken for a dead collector


def test_directorate_of_enforcement_feed_becomes_index_rows(tmp_path):
    payload = json.dumps([
        {"Id": 1, "Description": "ED conducted searches at premises linked to Thiru A. Arulmozhi Vendhanathan, a Member of the Legislative Assembly",
         "ReleaseDate": "2023-08-05T00:00:00", "PdfUrl": "/media/press-release-documents/abc_Press Release 1.pdf"},
        {"Id": 2, "Description": "", "ReleaseDate": "2023-08-06T00:00:00", "PdfUrl": "/x.pdf"},
        {"Id": 3, "Description": "No file attached", "ReleaseDate": "2023-08-07T00:00:00", "PdfUrl": None}]).encode()
    rows = official_index.parse_ed(payload)
    assert len(rows) == 1 and rows[0]["kind"] == "ed" and rows[0]["date"] == "2023-08-05" and rows[0]["year"] == 2023
    assert rows[0]["url"] == "https://enforcementdirectorate.gov.in/media/press-release-documents/abc_Press%20Release%201.pdf"
    idx = official_index.OfficialIndex(tmp_path / "i.sqlite")
    idx.add_rows("agency:ed", rows)
    assert [r["kind"] for r in idx.match({"name": "Arulmozhi Vendhanathan", "en_title": "A. Vendhanathan"})] == ["ed"]


# ---- a typed request: one person, some sources -----------------------------------------------------------------------------------------
from backend.pipeline import collect_request, person_folder        # noqa: E402


def test_a_typed_request_is_split_into_a_name_and_sources():
    assert collect_request.parse("collect C. Joseph Vijay ED cases") == ("c. joseph vijay", ["ed"])
    assert collect_request.parse("get all CBI and DVAC records of Arulmozhi Vendhan") == ("arulmozhi vendhan", ["cbi", "dvac"])
    assert collect_request.parse("Arulmozhi Vendhan court judgments, wikipedia") == ("arulmozhi vendhan", ["courts", "wikipedia"])
    assert collect_request.parse("collect Arulmozhi Vendhan") == ("arulmozhi vendhan", [])                  # no source named: everything
    assert collect_request.stages_for(["ed"]) == ["ed_records"]
    assert collect_request.stages_for(["wikipedia", "courts"]) == ["wikipedia_en", "wikipedia_ta", "court_records"]
    assert collect_request.stages_for([]) == list(STAGES)


def test_plan_runs_a_single_match_asks_when_several_and_falls_back_to_wikidata(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    reg.upsert_person({"key": "wd:Q1", "name": "Chandra Josephraj Vijayan", "en_title": "C. Josephraj Vijayan", "priority": 0})
    reg.upsert_person({"key": "wd:Q2", "name": "Vijayan Kumar", "en_title": "Vijayan Kumar", "priority": 2})
    reg.upsert_person({"key": "wd:Q3", "name": "Vijayan Senthil", "en_title": "Vijayan Senthil", "priority": 2})
    one = collect_request.plan(reg, "collect Josephraj Vijayan ED cases")
    assert one["ok"] and one["person"]["key"] == "wd:Q1" and one["stages"] == ["ed_records"] and one["sources"] == ["ed"]
    many = collect_request.plan(reg, "collect Vijayan CBI")
    assert not many["ok"] and many["needs_choice"] and {c["key"] for c in many["candidates"]} == {"wd:Q1", "wd:Q2", "wd:Q3"}
    picked = collect_request.plan(reg, person="wd:Q2", sources=many["sources"])                          # the owner picked one from the list
    assert picked["ok"] and picked["person"]["name"] == "Vijayan Kumar" and picked["stages"] == ["cbi_records"]
    assert collect_request.plan(reg, "")["ok"] is False

    def lookup(name):
        return [{"qid": "Q99", "name": "Somebody New", "description": "Indian politician"}]

    def add(registry, qid):
        registry.upsert_person({"key": "wd:" + qid, "name": "Somebody New", "en_title": "Somebody New", "priority": 3})
        return registry.person("wd:" + qid)
    new = collect_request.plan(reg, "somebody new wikipedia", lookup=lookup, add=add)
    assert new["ok"] and new["person"]["key"] == "wd:Q99" and new["stages"] == ["wikipedia_en", "wikipedia_ta"]
    assert collect_request.plan(reg, "nobody here", lookup=lambda n: [])["ok"] is False


def test_a_targeted_run_collects_only_the_requested_stages_and_leaves_the_roster_status_alone(env):
    reg, sink, archive, tmp = env
    wm = FakeWikimedia()
    c = collector(env, wm)
    assert c.run_targeted(PERSON["key"], ["wikipedia_en"]) == "done"
    assert [m["topic"] for m in sink.messages] == ["person"]                                       # no Wikidata, no Tamil page, no case articles
    assert ("entity", "Q9001") not in wm.calls and not any(call[0] == "search" for call in wm.calls)
    p = reg.person(PERSON["key"])
    assert p["status"] == "pending" and p["sources_done"] == ["wikipedia_en"]                         # still waiting for the full run, which skips this stage
    assert collector(env, FakeWikimedia()).run() == "done"
    assert len([m for m in sink.messages if m["topic"] == "person" and m["language"] == "en"]) == 1  # the English article was not collected twice


# ---- one folder per person; nothing the knowledge base already has is stored again -------------------------------------------------------
def test_each_person_gets_a_folder_of_linked_files_and_a_manifest(env):
    reg, sink, archive, tmp = env
    assert collector(env).run() == "done"
    folder = tmp / "kb_raw" / "people" / person_folder.folder_name(PERSON)
    assert folder.name == "Arulmozhi-Vendhan__Q9001"
    topics = sorted(p.name for p in folder.iterdir() if p.is_dir())
    assert topics == ["corruption", "person", "structured"]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "Arulmozhi Vendhan" and manifest["documents"] == 4 and manifest["documents_by_topic"]["corruption"] == 1
    for entry in manifest["files"]:
        linked = folder / entry["path"]
        original = tmp / "kb_raw" / [m["raw_path"] for m in sink.messages if m["doc_id"] == entry["doc_id"]][0].removeprefix("kb_raw/")
        assert linked.read_bytes() == original.read_bytes()
    assert person_folder.folder_name({"key": "wd:Q5", "name": "அருள்", "wikidata_id": "Q5"}) == "Q5"            # a Tamil-only name falls back to the id


def test_a_page_the_knowledge_base_already_holds_is_linked_not_downloaded_or_published_again(env):
    reg, sink, archive, tmp = env
    wm = FakeWikimedia()
    en_url = person_sources.wikipedia_url("en", "Arulmozhi Vendhan")
    c = collector(env, wm, known_urls=lambda: {en_url})
    assert c.run() == "done"
    assert ("html", "en", "Arulmozhi Vendhan") not in wm.calls                                  # not even requested
    assert "person" not in {m["topic"] for m in sink.messages if m["language"] == "en"}
    docs = {d["doc_id"]: d for d in reg.documents_of(PERSON["key"])}
    ref = [d for d in docs.values() if d["status"] == "in_kb"]
    assert len(ref) == 1 and ref[0]["url"] == en_url and ref[0]["raw_path"] is None
    manifest = json.loads((tmp / "kb_raw" / "people" / person_folder.folder_name(PERSON) / "manifest.json").read_text(encoding="utf-8"))
    assert any(f["in_knowledge_base_before"] for f in manifest["files"])
    assert reg.unlinked_in_graph() and all(r["person_key"] == PERSON["key"] for r in reg.unlinked_in_graph())


# ---- news: headlines and links only -----------------------------------------------------------------------------------------------------------
from backend.pipeline import process                                  # noqa: E402

GDELT_REPLY = json.dumps({"articles": [
    {"url": "https://news.example.com/a", "title": "ED searches premises linked to Arulmozhi Vendhan in a money laundering case", "seendate": "20261014T080000Z",
     "domain": "news.example.com", "language": "English", "sourcecountry": "India"},
    {"url": "https://news.example.com/b", "title": "Court dismisses petition filed by Vendhan against the income tax penalty", "seendate": "20261013T080000Z",
     "domain": "news.example.com", "language": "English", "sourcecountry": "India"},
    {"url": "https://news.example.com/c", "title": "Vendhan inaugurates a new bus stand in Chennai", "seendate": "20261012T080000Z", "domain": "news.example.com"},
    {"url": "https://news.example.com/d", "title": "ED raids a mining company in another state", "seendate": "20261011T080000Z", "domain": "news.example.com"},
    {"url": "https://news.example.com/a", "title": "ED searches premises linked to Arulmozhi Vendhan in a money laundering case", "seendate": "20261014T080000Z"},
]}).encode()


class FakeNewsFetcher:
    def __init__(self, body=GDELT_REPLY):
        self.body, self.urls = body, []

    def get(self, url):
        self.urls.append(url)
        return self.body, "application/json"


def test_only_headlines_that_name_the_person_and_a_case_are_kept_and_nothing_but_the_record_is_stored():
    found = person_sources.news_docs(lambda url: GDELT_REPLY, PERSON)
    assert [f.url for f in found] == ["https://news.example.com/a", "https://news.example.com/b"]              # not the bus stand, not another state, no repeat
    f = found[0]
    assert (f.topic, f.doc_type, f.source, f.ext, f.language) == ("news", "news", "dataset", "json", "en")
    assert f.extra["published_date"] == "2026-10-14" and "headline and link only" in f.licence.lower()
    assert json.loads(f.data)["title"].startswith("ED searches")                                           # the index's own record, no article text
    twice = json.dumps({"articles": [{"url": "https://a.test/1", "title": "ED searches Vendhan premises", "seendate": "20261014T080000Z"},
                                     {"url": "https://b.test/2", "title": "ED searches  Vendhan premises!", "seendate": "20261014T090000Z"}]}).encode()
    assert len(person_sources.news_docs(lambda url: twice, PERSON)) == 1                                       # one headline, two links: kept once
    assert person_sources.news_docs(lambda url: b"Your query was too short", PERSON) == []                  # a plain-text error from the index is not a crash


def test_the_news_stage_collects_links_once_and_skips_what_the_knowledge_base_already_holds(env):
    reg, sink, archive, tmp = env
    fetcher = FakeNewsFetcher()
    c = collector(env, news_fetcher=fetcher, known_urls=lambda: {"https://news.example.com/b"})
    assert c.run_targeted(PERSON["key"], ["news_records"]) == "done"
    assert "api.gdeltproject.org" in fetcher.urls[0] and "sourcelang%3Aenglish" in fetcher.urls[0]
    assert [m["url"] for m in sink.messages] == ["https://news.example.com/a"]                              # /b is in the knowledge base already: linked, not repeated
    m = sink.messages[0]
    assert m["topic"] == "news" and m["doc_type"] == "news" and m["published_date"] == "2026-10-14" and m["raw_path"].startswith("kb_raw/dataset/")
    assert collector(env, news_fetcher=FakeNewsFetcher()).run_targeted(PERSON["key"], ["news_records"]) == "done"
    assert len(sink.messages) == 1                                                                         # a second run adds nothing
    assert "news" in {d["topic"] for d in reg.documents_of(PERSON["key"])}
    assert (tmp / "kb_raw" / "people" / person_folder.folder_name(PERSON) / "news").is_dir()


def test_a_news_record_becomes_one_short_passage_that_says_who_reported_it(env, tmp_path):
    reg, sink, archive, tmp = env
    c = collector(env, news_fetcher=FakeNewsFetcher())
    c.run_targeted(PERSON["key"], ["news_records"])
    msg = sink.messages[0]
    out = process.process_message(msg, tmp / "kb_raw")
    assert out["skipped"] is False
    assert out["text"].startswith("ED searches premises linked to Arulmozhi Vendhan") and "Reported by news.example.com (India) on 2026-10-14." in out["text"]
    assert len(out["passages"]) == 1 and "http" not in out["text"]
    assert collect_request.parse("collect Arulmozhi Vendhan news") == ("arulmozhi vendhan", ["news"])
    assert collect_request.stages_for(["news"]) == ["news_records"]


# ---- full articles, where the site allows it --------------------------------------------------------------------------------------------------
ARTICLE = page(("ED searched premises linked to Arulmozhi Vendhan in Chennai, Tamil Nadu, in a money laundering case, officials said. " * 20)).replace(b"<nav>menu</nav>", b"<nav>menu</nav><footer>cookies</footer>")


def test_the_article_page_replaces_the_headline_record_only_when_it_can_be_read_and_is_about_the_person():
    pages = {"https://news.example.com/a": (ARTICLE, "text/html; charset=utf-8"),
             "https://news.example.com/b": (page("Short stub. Subscribe to read more."), "text/html"),
             "https://news.example.com/x": (b"%PDF", "application/pdf")}

    def reader(url):
        if url not in pages:
            raise FetchError("robots.txt disallows " + url)
        return pages[url]
    found = person_sources.news_docs(lambda url: GDELT_REPLY, PERSON, fetch_article=reader)
    a, b = found
    assert (a.source, a.ext, a.mime, a.doc_type) == ("web", "html", "text/html", "news") and a.data == ARTICLE and "copyright of the publisher" in a.licence.lower()
    assert a.tokens > 300 and a.extra["published_date"] == "2026-10-14"                                         # the original page, not a rewrite
    assert (b.source, b.ext) == ("dataset", "json") and json.loads(b.data)["url"] == "https://news.example.com/b"        # a paywall stub: the headline record stands


def test_a_site_that_forbids_robots_is_never_read_and_only_the_headline_is_kept():
    def blocked(url):
        raise FetchError("robots.txt disallows " + url)
    found = person_sources.news_docs(lambda url: GDELT_REPLY, PERSON, fetch_article=blocked)
    assert [f.ext for f in found] == ["json", "json"]
    assert person_sources.news_docs(lambda url: GDELT_REPLY, PERSON, fetch_article=lambda u: (ARTICLE, "text/html"), max_articles=1)[1].ext == "json"


def test_news_text_is_kept_out_of_the_public_snapshot_and_the_training_export(kb, rag_store):
    from backend.knowledge import snapshot
    from backend.knowledge.export_training import OFFICIAL
    from backend.knowledge.store import store_document
    text = "ED searched premises linked to a fictional politician in a money laundering case, officials said on Tuesday in a long report."
    store_document(kb, rag_store, title="A report", doc_type="news", text=text, url="https://news.example.com/a", publisher="news.example.com",
                   passages=[{"locator": "p1", "text": text}])
    store_document(kb, rag_store, title="An order", doc_type="order", text="The Special Court framed charges against a fictional person on 1 May 2019 in the case.",
                   url="https://example.org/o", passages=[{"locator": "p1", "text": "The Special Court framed charges against a fictional person on 1 May 2019 in the case."}])
    ids, reasons = snapshot.select_passage_ids(kb)
    titles = {r["t"] for r in kb._read("MATCH (p:Passage)-[:FROM]->(s:Source) WHERE p.id IN $ids RETURN s.title AS t", ids=ids)}
    assert titles == {"An order"}                                                          # the news passage is not in the public vector snapshot
    assert "news" not in OFFICIAL
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        kb.export_jsonl(Path(d) / "g.jsonl", skip_reference=True)
        assert "A report" not in (Path(d) / "g.jsonl").read_text(encoding="utf-8")        # nor in the public graph snapshot
