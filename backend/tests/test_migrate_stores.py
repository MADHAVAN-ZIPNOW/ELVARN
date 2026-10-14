"""Moving the stores to hosted services. Neo4j tests need the disposable database (see the `kb` fixture); Chroma is in memory."""
import json
import uuid

import chromadb
import pytest

from backend.knowledge import migrate_stores
from backend.knowledge import vectorstore
from knowledge.vectorstore import RagStore
from test_graphrag import load


def snapshot(kb):
    """Everything that matters about a graph, ignoring Neo4j's internal ids."""
    nodes = kb._read("MATCH (n) RETURN labels(n) AS l, properties(n) AS p")
    rels = kb._read("MATCH (a)-[r]->(b) RETURN labels(a) AS a, type(r) AS t, labels(b) AS b, properties(r) AS p")
    key = lambda x: json.dumps(x, sort_keys=True, default=str)                         # noqa: E731
    return sorted(key([sorted(n["l"]), n["p"]]) for n in nodes), sorted(key([sorted(r["a"]), r["t"], sorted(r["b"]), r["p"]]) for r in rels)


# ---- graph -----------------------------------------------------------------------------------------
def test_graph_survives_export_wipe_import_unchanged(kb, rag_store, tmp_path):
    load(kb, rag_store)
    before = snapshot(kb)
    person = kb.find_entities("Test Person")[0]["id"]
    facts_before = [(r["case_name"], r["status"], r["level"]) for r in kb.cases_of_person(person)]
    exported = kb.export_jsonl(tmp_path / "g.jsonl")
    kb._run("MATCH (n) DETACH DELETE n")
    assert kb.import_jsonl(tmp_path / "g.jsonl") == exported
    assert snapshot(kb) == before                                                       # same nodes, properties and relationships
    assert [(r["case_name"], r["status"], r["level"]) for r in kb.cases_of_person(person)] == facts_before
    assert kb.audit() == []
    assert kb._read("MATCH (n) WHERE n._xid IS NOT NULL RETURN count(n) AS n")[0]["n"] == 0        # the temporary id was removed
    assert not [i for i in kb._read("SHOW INDEXES YIELD name RETURN name") if i["name"].startswith("xid_")]


def test_import_refuses_a_target_that_is_not_empty(kb, rag_store, tmp_path):
    load(kb, rag_store)
    kb.export_jsonl(tmp_path / "g.jsonl")
    with pytest.raises(ValueError, match="not empty"):
        kb.import_jsonl(tmp_path / "g.jsonl")


def test_import_rejects_a_backup_with_unexpected_labels_or_types(kb, tmp_path):
    bad_label = tmp_path / "a.jsonl"
    bad_label.write_text(json.dumps({"node": "1", "labels": ["Source) DETACH DELETE (x"], "props": {}}) + chr(10), encoding="utf-8")
    with pytest.raises(ValueError, match="labels"):
        kb.import_jsonl(bad_label)
    bad_rel = tmp_path / "b.jsonl"
    bad_rel.write_text(json.dumps({"node": "1", "labels": ["Source"], "props": {}}) + chr(10)
                       + json.dumps({"rel": "FROM]->() DETACH DELETE (n", "from": "1", "to": "1", "props": {}}) + chr(10), encoding="utf-8")
    with pytest.raises(ValueError, match="relationship type"):
        kb.import_jsonl(bad_rel)
    assert kb._read("MATCH (n) RETURN count(n) AS n")[0]["n"] == 0                       # nothing was created


def test_copy_graph_between_two_graphs(kb, rag_store, tmp_path):
    load(kb, rag_store)
    # the same database plays both roles: export, wipe (the "target" starts empty), restore
    expected = kb.export_jsonl(tmp_path / "x.jsonl")
    kb._run("MATCH (n) DETACH DELETE n")
    assert kb.import_jsonl(tmp_path / "x.jsonl") == expected


# ---- chroma ----------------------------------------------------------------------------------------
def fresh_store(like):
    """A second, empty collection in the same in-memory Chroma, using the same embedder."""
    return RagStore(client=chromadb.EphemeralClient(), embedding_function=like.col._embedding_function, collection="t" + uuid.uuid4().hex[:12])


def test_chroma_copy_keeps_text_metadata_and_embeddings_and_is_repeatable(rag_store):
    items = [{"id": f"id{i}", "text": f"Passage number {i} about the Madras High Court and bribery.", "meta": {"title": f"T{i}", "doc_type": "order", "url": f"https://x/{i}"}}
             for i in range(7)]
    rag_store.add(items)
    dst = fresh_store(rag_store)
    assert migrate_stores.copy_chroma(rag_store, dst, batch=3, progress=lambda *a, **k: None) == 7
    assert dst.count() == 7
    a, b = rag_store.col.get(ids=["id4"], include=["documents", "metadatas", "embeddings"]), dst.col.get(ids=["id4"], include=["documents", "metadatas", "embeddings"])
    assert a["documents"] == b["documents"] and a["metadatas"] == b["metadatas"] and list(a["embeddings"][0]) == pytest.approx(list(b["embeddings"][0]))
    assert dst.query("Madras High Court bribery passage 4", k=1)[0]["meta"]["title"] in {f"T{i}" for i in range(7)}
    assert migrate_stores.copy_chroma(rag_store, dst, progress=lambda *a, **k: None) == 0          # second run copies nothing


# ---- connection settings ----------------------------------------------------------------------------
class Spy:
    def __init__(self):
        self.calls = []

    def HttpClient(self, **kw):                                                                       # noqa: N802
        self.calls.append(("http", kw))
        return "http-client"

    def CloudClient(self, **kw):                                                                      # noqa: N802
        self.calls.append(("cloud", kw))
        return "cloud-client"


@pytest.fixture()
def spy(monkeypatch):
    s = Spy()
    monkeypatch.setattr(chromadb, "HttpClient", s.HttpClient)
    monkeypatch.setattr(chromadb, "CloudClient", s.CloudClient)
    for name in ("CHROMA_HOST", "CHROMA_PORT", "CHROMA_SSL", "CHROMA_TOKEN", "CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE",
                 "TARGET_CHROMA_HOST", "TARGET_CHROMA_API_KEY", "TARGET_CHROMA_TENANT", "TARGET_CHROMA_DATABASE"):
        monkeypatch.delenv(name, raising=False)
    return s


def test_nothing_configured_means_disabled(spy):
    assert vectorstore.configured() is False and RagStore.from_env() is None


def test_local_server_defaults(spy, monkeypatch):
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    assert vectorstore.make_client() == "http-client"
    assert spy.calls == [("http", {"host": "localhost", "port": 8001, "ssl": False, "headers": None})]


def test_hosted_server_over_https_with_a_token(spy, monkeypatch):
    monkeypatch.setenv("CHROMA_HOST", "chroma.example.com")
    monkeypatch.setenv("CHROMA_SSL", "1")
    monkeypatch.setenv("CHROMA_TOKEN", "secret-token")
    vectorstore.make_client()
    assert spy.calls == [("http", {"host": "chroma.example.com", "port": 443, "ssl": True, "headers": {"Authorization": "Bearer secret-token"}})]


def test_chroma_cloud_uses_the_cloud_client(spy, monkeypatch):
    monkeypatch.setenv("CHROMA_API_KEY", "ck-123")
    monkeypatch.setenv("CHROMA_TENANT", "tenant-1")
    monkeypatch.setenv("CHROMA_DATABASE", "elvarn")
    assert vectorstore.configured() is True and vectorstore.make_client() == "cloud-client"
    assert spy.calls == [("cloud", {"tenant": "tenant-1", "database": "elvarn", "api_key": "ck-123"})]


def test_target_prefix_selects_a_second_store(spy, monkeypatch):
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("TARGET_CHROMA_API_KEY", "ck-9")
    monkeypatch.setenv("TARGET_CHROMA_TENANT", "t")
    monkeypatch.setenv("TARGET_CHROMA_DATABASE", "d")
    vectorstore.make_client("TARGET_")
    assert spy.calls[-1] == ("cloud", {"tenant": "t", "database": "d", "api_key": "ck-9"}) and vectorstore.configured("TARGET_")


# ---- a lean hosted graph (free-tier node limit): Wikipedia/notes passages live only in Chroma ------------------
def titles(kb):
    return sorted(r["t"] for r in kb._read("MATCH (s:Source) RETURN s.title AS t"))


def test_official_only_export_drops_uncited_reference_documents(kb, rag_store, tmp_path):
    load(kb, rag_store)
    full = kb.export_jsonl(tmp_path / "full.jsonl")
    lean = kb.export_jsonl(tmp_path / "lean.jsonl", skip_reference=True)
    assert lean["nodes"] < full["nodes"] and lean["relationships"] < full["relationships"]
    kb._run("MATCH (n) DETACH DELETE n")
    assert kb.import_jsonl(tmp_path / "lean.jsonl") == lean
    assert titles(kb) == ["DVAC press release 12", "Test v. State (Madras High Court)"]                 # the Wikipedia article is gone
    assert kb._read("MATCH (p:Passage) RETURN count(p) AS n")[0]["n"] == 2 and kb.audit() == []
    person = kb.find_entities("Test Person")[0]["id"]
    assert kb.cases_of_person(person)                                                                 # facts and entities came across intact


def test_a_reference_document_that_a_fact_cites_is_kept(kb, rag_store, tmp_path):
    load(kb, rag_store)
    pid = kb._read("MATCH (p:Passage)-[:FROM]->(s:Source {doc_type:'reference'}) RETURN p.id AS id")[0]["id"]
    kb.add_claim(kb.find_entities("Test Person")[0]["id"], "lives_near", "reported", pid, object_value="the Kaveri")
    kb.export_jsonl(tmp_path / "lean.jsonl", skip_reference=True)
    kb._run("MATCH (n) DETACH DELETE n")
    kb.import_jsonl(tmp_path / "lean.jsonl")
    assert "Rivers of Tamil Nadu (Wikipedia)" in titles(kb) and kb.audit() == []                         # the citation still resolves


def test_reference_documents_can_be_kept_out_of_the_graph(kb, rag_store, monkeypatch):
    from knowledge.store import store_document
    monkeypatch.setenv("GRAPH_REFERENCE_DOCS", "0")
    text = "The Kaveri is the longest river flowing through Tamil Nadu and supports farming in the delta districts."
    assert store_document(kb, rag_store, title="Kaveri (Wikipedia)", doc_type="reference", text=text, passages=[{"locator": "chunk 1", "text": text}]) == "loaded"
    assert kb._read("MATCH (n) RETURN count(n) AS n")[0]["n"] == 0                                       # nothing in the graph...
    assert rag_store.query("longest river Tamil Nadu farming", k=1)[0]["meta"]["title"] == "Kaveri (Wikipedia)"   # ...but searchable
    assert store_document(kb, rag_store, title="Kaveri (Wikipedia)", doc_type="reference", text=text, passages=[{"locator": "chunk 1", "text": text}]) == "duplicate"
    official = "DVAC press release dated 2026-10-12: A village officer was trapped and arrested for demanding a bribe in Nilgiris District."
    assert store_document(kb, rag_store, title="DVAC 1", doc_type="press_release", text=official, passages=[{"locator": "listing", "text": official}]) == "loaded"
    assert kb._read("MATCH (s:Source) RETURN count(s) AS n")[0]["n"] == 1                                # official documents still go to both stores
    monkeypatch.delenv("GRAPH_REFERENCE_DOCS")
    other = "Another reference text about the Vaigai river and the districts it flows through in southern Tamil Nadu and its history."
    store_document(kb, rag_store, title="Vaigai (Wikipedia)", doc_type="reference", text=other, passages=[{"locator": "chunk 1", "text": other}])
    assert kb._read("MATCH (s:Source) RETURN count(s) AS n")[0]["n"] == 2                                # default behaviour is unchanged
