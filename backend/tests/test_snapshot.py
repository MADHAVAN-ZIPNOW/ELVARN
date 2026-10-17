"""The free hosted copy: a read-only in-memory graph + a small Chroma, built from a snapshot of the real stores.
Needs the disposable Neo4j for the `kb` fixture. All names and cases are FICTIONAL."""
import json

from backend.knowledge.snapshot_graph import SnapshotGraph
from test_graphrag import load


def rich_data(kb, rag_store):
    """load() plus a second case, co-accused, a claim to an entity, a claim with a plain value, and a rejected review decision."""
    load(kb, rag_store)
    pid = kb._read("MATCH (p:Passage)-[:FROM]->(s:Source {doc_type:'order'}) RETURN p.id AS id")[0]["id"]
    pr = kb._read("MATCH (p:Passage)-[:FROM]->(s:Source {doc_type:'press_release'}) RETURN p.id AS id")[0]["id"]
    person = kb.find_entities("Test Person")[0]["id"]
    second = kb.entity("Person", "Second Accused", aliases=["S. Accused"])
    case2 = kb.add_case("Second Trap Case", year_registered=2026, department="Revenue", location="Nilgiris", amount_inr=20000)
    kb.add_party(case2, person, "accused", "reported", pr, status="arrested", status_date="2026-10-12")
    kb.add_party(case2, second, "complainant", "alleged", pr)
    kb.add_event(case2, "arrest", "Arrested while accepting a bribe", "reported", pr, event_date="2026-10-12")
    ministry = kb.entity("Organisation", "Test Ministry")
    kb.add_claim(person, "holds_office", "reported", pid, object_id=ministry, valid_from="2016-01-01")
    kb.add_claim(person, "age", "reported", pr, object_value="55")
    return person, second, case2


def snapshot_of(kb, tmp_path):
    kb.export_jsonl(tmp_path / "lean.jsonl", skip_reference=True)
    return SnapshotGraph(tmp_path / "lean.jsonl")


def norm(rows):
    return sorted(json.dumps(r, sort_keys=True, default=str) for r in rows)


def test_snapshot_graph_answers_exactly_like_neo4j(kb, rag_store, tmp_path):
    person, second, case2 = rich_data(kb, rag_store)
    snap = snapshot_of(kb, tmp_path)
    case1 = "Case:test housing scam"
    assert snap.populated() is True and kb.populated() is True
    for text in ("List the cases of T. Person", "tell me about Test Person and Second Accused", "what about S. Accused?", "nothing relevant here",
                 "the Second Trap Case timeline", "Test Housing Scam", "Test Ministry", "PERSONS are not the same as person"):
        assert norm(snap.find_entities(text)) == norm(kb.find_entities(text)), text
        for type_ in ("Person", "Case"):
            assert norm(snap.find_entities(text, type_)) == norm(kb.find_entities(text, type_)), (text, type_)
    for who in (person, second, "Person:nobody"):
        assert snap.cases_of_person(who) == kb.cases_of_person(who), who                     # same rows, same order
    for case in (case1, case2, "Case:nothing"):
        assert snap.case_timeline(case) == kb.case_timeline(case), case
        assert norm(snap.case_parties(case)) == norm(kb.case_parties(case)), case
        assert snap.case_info(case) == kb.case_info(case), case
    for who in (person, second):
        assert norm(snap.entity_claims(who)) == norm(kb.entity_claims(who)), who


def test_snapshot_graph_drives_the_same_retrieval_context(kb, rag_store, memory, tmp_path):
    """graphrag.build_context must give the same facts from the snapshot as from Neo4j."""
    from backend.knowledge import graphrag
    person, second, case2 = rich_data(kb, rag_store)
    snap = snapshot_of(kb, tmp_path)
    for question in ("List the corruption cases of Test Person", "What happened in the Second Trap Case?", "village officer arrested for a bribe"):
        a = graphrag.build_context(kb, rag_store, question)
        b = graphrag.build_context(snap, rag_store, question)
        assert [(i["kind"], i["text"], i["score"]) for i in a.items] == [(i["kind"], i["text"], i["score"]) for i in b.items], question
        assert a.understanding.entities == b.understanding.entities and a.understanding.intent == b.understanding.intent


def test_a_snapshot_with_no_facts_is_still_valid(kb, rag_store, tmp_path):
    kb.export_jsonl(tmp_path / "empty.jsonl")
    snap = SnapshotGraph(tmp_path / "empty.jsonl")
    assert snap.populated() is False and snap.find_entities("Test Person") == [] and snap.cases_of_person("Person:x") == []


# ---- exporting and loading the snapshot ------------------------------------------------------------------------
import uuid  # noqa: E402

import chromadb  # noqa: E402
import pytest  # noqa: E402

from backend.knowledge import snapshot  # noqa: E402
from knowledge.store import store_document  # noqa: E402
from knowledge.vectorstore import RagStore  # noqa: E402


def mixed_corpus(kb, rag):
    """Official documents, the project's own notes, a Tamil Nadu corruption article, and a general article that must be left out."""
    docs = [("order", "Test v. State (Madras High Court)", "The Special Court framed charges against the accused under the Prevention of Corruption Act in Tamil Nadu.", "Madras High Court"),
            ("reference", "Anti-Corruption Law (ELVARN project notes)", "The Prevention of Corruption Act punishes public servants who take a bribe or hold assets beyond their income.", "ELVARN project domain notes (unverified)"),
            ("reference", "Madras bribery scandal (Wikipedia)", "The Vigilance department in Tamil Nadu registered a bribery case against a Chennai official who was later convicted in court.", "Wikipedia"),
            ("reference", "Kaveri (Wikipedia)", "The Kaveri is the longest river in the south and supports farming across many delta districts of India.", "Wikipedia")]
    for doc_type, title, text, publisher in docs:
        store_document(kb, rag, title=title, doc_type=doc_type, text=text, passages=[{"locator": "chunk 1", "text": text}], publisher=publisher, url="https://example.org/" + title[:6])


def test_export_selects_official_notes_and_tamil_nadu_corruption_articles_only(kb, rag_store, tmp_path):
    mixed_corpus(kb, rag_store)
    manifest = snapshot.export_snapshot(kb, rag_store, tmp_path / "snap")
    assert manifest["passages"] == 3 and {k: v for k, v in manifest["passages_by_reason"].items() if not k.startswith("judgments")} == {"official": 1, "project notes/files": 1, "wikipedia (Tamil Nadu + corruption)": 1}
    ids, texts, metas, vectors = snapshot.read_passages(tmp_path / "snap")
    assert not any("Kaveri" in t for t in texts) and vectors.shape[0] == 3 and {m["title"] for m in metas} >= {"Test v. State (Madras High Court)"}


def test_snapshot_files_are_checked_and_deterministic(kb, rag_store, tmp_path):
    mixed_corpus(kb, rag_store)
    a = snapshot.export_snapshot(kb, rag_store, tmp_path / "a")
    b = snapshot.export_snapshot(kb, rag_store, tmp_path / "b")
    assert a["files"] == b["files"]                                                      # same data -> byte-identical files (clean git diffs)
    assert snapshot.check_snapshot(tmp_path / "a")["passages"] == 3
    (tmp_path / "a" / snapshot.PASSAGES).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        snapshot.check_snapshot(tmp_path / "a")


def test_export_refuses_when_a_passage_is_not_indexed_yet(kb, rag_store, tmp_path):
    mixed_corpus(kb, rag_store)
    rag_store.col.delete(where={"doc_type": "order"})
    with pytest.raises(RuntimeError, match="reindex"):
        snapshot.export_snapshot(kb, rag_store, tmp_path / "snap")


def test_loading_fills_a_store_with_the_same_search_results_and_is_idempotent(kb, rag_store, tmp_path):
    mixed_corpus(kb, rag_store)
    snapshot.export_snapshot(kb, rag_store, tmp_path / "snap")
    target = RagStore(client=chromadb.EphemeralClient(), embedding_function=rag_store.col._embedding_function, collection="t" + uuid.uuid4().hex[:12])
    marker = tmp_path / "chroma" / "snapshot.marker"
    assert snapshot.load_vectors(tmp_path / "snap", target, marker_file=marker) == 3 and target.count() == 3
    q = "bribery case Tamil Nadu vigilance"
    assert target.query(q, k=1)[0]["meta"]["title"] == "Madras bribery scandal (Wikipedia)"
    assert snapshot.load_vectors(tmp_path / "snap", target, marker_file=marker) == 0       # same snapshot: nothing to do
    kb._run("MATCH (n) DETACH DELETE n")
    rag_store.col.delete(where={"doc_type": "reference"})
    store_document(kb, rag_store, title="Another order", doc_type="order", text="A different High Court order about a Tamil Nadu bribery trial and its charges.",
                   passages=[{"locator": "p1", "text": "A different High Court order about a Tamil Nadu bribery trial and its charges."}])
    snapshot.export_snapshot(kb, rag_store, tmp_path / "snap2")
    assert snapshot.load_vectors(tmp_path / "snap2", target, marker_file=marker) == 1 and target.count() == 1      # a new snapshot replaces the old one


# ---- the service, end to end, with the REAL embedding model ------------------------------------------------------
def test_service_runs_from_a_snapshot_with_no_external_store(kb, tmp_path, monkeypatch):
    from knowledge_service import app as svc
    from fastapi.testclient import TestClient
    real = RagStore(client=chromadb.EphemeralClient(), collection="t" + uuid.uuid4().hex[:12])          # default embedder: all-MiniLM-L6-v2
    load(kb, real)
    rich_data_extra = kb.find_entities("Test Person")[0]["id"]
    assert rich_data_extra
    snapshot.export_snapshot(kb, real, tmp_path / "snap")
    for name in ("NEO4J_URI", "CHROMA_HOST", "CHROMA_API_KEY", "REDIS_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SNAPSHOT_DIR", str(tmp_path / "snap"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("KNOWLEDGE_TOKEN", "t" * 32)
    app = svc.create_from_env()
    c = TestClient(app, headers={"Authorization": "Bearer " + "t" * 32})
    assert c.get("/healthz").json() == {"status": "ok"}
    out = c.post("/v1/context", json={"question": "List the corruption cases of Test Person"}).json()
    assert out["populated"] is True and out["items"][0]["kind"] == "fact" and "Test Housing Scam" in out["items"][0]["text"]
    out2 = c.post("/v1/context", json={"question": "village officer trapped and arrested for demanding a bribe"}).json()
    assert out2["items"] and out2["items"][0]["source"]["url"] == "https://example.org/pr12"            # vector search over the loaded snapshot
    assert c.post("/v1/context", json={"question": "Who won the latest FIFA World Cup?"}).json()["items"] == []
    assert (tmp_path / "chroma" / "snapshot.marker").is_file()


def test_loading_a_snapshot_needs_none_of_the_exporting_machines_code():
    """The image contains only backend/knowledge + backend/usermemory + knowledge_service. Importing the loader and the in-memory graph must not
    reach for anything else (found when the first image build failed on `backend.data_tools`)."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    code = ("import sys\nfor m in ('backend.data_tools', 'backend.pipeline', 'backend.files', 'backend.search', 'neo4j'):\n    sys.modules[m] = None\n"
            "import backend.knowledge.snapshot, backend.knowledge.snapshot_graph, backend.knowledge.vectorstore, backend.knowledge.graphrag\n"
            "import knowledge_service.app\nprint('OK')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, timeout=120)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-1200:]


def test_judgments_on_corruption_and_elections_are_capped_per_case_and_other_subjects_left_out(kb, rag_store, tmp_path, monkeypatch):
    import gzip
    monkeypatch.setattr(snapshot, "JUDGMENT_PASSAGES_PER_CASE", 3)

    def judgment(title, subjects, count):
        passages = [{"locator": f"chunk {i}", "text": f"Passage {i} of {title}: the matter before the court is explained in detail number {i}."} for i in range(1, count + 1)]
        store_document(kb, rag_store, title=title, doc_type="judgment", text=" ".join(p["text"] for p in passages), passages=passages,
                       publisher="Supreme Court of India", url="https://example.org/" + title[:6])
        kb._run("MATCH (s:Source {title: $t}) SET s.subjects = $subs", t=title, subs=subjects)

    judgment("Election Case A", ["election"], 6)
    judgment("Family Case B", ["family"], 6)
    manifest = snapshot.export_snapshot(kb, rag_store, tmp_path / "snap")
    reason = next(k for k in manifest["passages_by_reason"] if k.startswith("judgments"))
    assert manifest["passages_by_reason"][reason] == 3 and manifest["passages"] == 3                   # 3 of the 6 passages of the election case, none of the family case
    ids, texts, metas, vectors = snapshot.read_passages(tmp_path / "snap")
    assert all("Election Case A" in t for t in texts) and not any("Family Case B" in t for t in texts)
    graph = gzip.open(tmp_path / "snap" / snapshot.GRAPH, "rt", encoding="utf-8").read()
    assert "Election Case A" not in graph and "Family Case B" not in graph                               # judgments are served by the vector store only
