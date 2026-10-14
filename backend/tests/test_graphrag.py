"""GraphRAG pipeline: Chroma + Neo4j + Redis memory (needs the disposable Neo4j for the `kb` fixture). All names and cases are FICTIONAL."""
import pytest

from backend.knowledge import graphrag
from backend.knowledge.store import consistency, store_document


def load(kb, rag_store):
    """Documents go into BOTH stores through the single loader; approved facts are added to the graph."""
    j = "Test v. State: the Special Court framed charges against Test Person under the Prevention of Corruption Act on 1 May 2019. Test Person was accused of possessing assets disproportionate to known sources of income."
    store_document(kb, rag_store, title="Test v. State (Madras High Court)", doc_type="order", text=j, url="https://example.org/order1",
                   published_date="2019-05-02", publisher="Madras High Court", licence="CC-BY-4.0",
                   passages=[{"locator": "page 1", "text": j}])
    pr = "DVAC press release dated 2026-10-12: A village officer was trapped and arrested for demanding a bribe of Rs.20,000/- in Nilgiris District."
    store_document(kb, rag_store, title="DVAC press release 12", doc_type="press_release", text=pr, url="https://example.org/pr12",
                   published_date="2026-10-12", publisher="DVAC", passages=[{"locator": "listing", "text": pr}])
    river = "The Kaveri is the longest river flowing through Tamil Nadu and supports farming in the delta districts."
    store_document(kb, rag_store, title="Rivers of Tamil Nadu (Wikipedia)", doc_type="reference", text=river, passages=[{"locator": "chunk 1", "text": river}])
    pid = kb._read("MATCH (p:Passage)-[:FROM]->(s:Source {doc_type:'order'}) RETURN p.id AS id")[0]["id"]
    person = kb.entity("Person", "Test Person", aliases=["T. Person"])
    case = kb.add_case("Test Housing Scam", case_number="CC 1/2018", year_registered=2018, department="Housing Board")
    other = kb.entity("Person", "Second Accused")
    kb.add_party(case, person, "accused", "reported", pid, status="charged", status_date="2019-05-01")
    kb.add_party(case, other, "accused", "alleged", pid, status="accused")
    kb.add_event(case, "trial", "Special Court framed charges", "court_finding", pid, event_date="2019-05-01") if False else \
        kb.add_event(case, "trial", "Special Court framed charges", "reported", pid, event_date="2019-05-01")
    kb.add_event(case, "fir", "FIR registered", "reported", pid, event_date="2018-03-10")


# ---- both stores hold everything ------------------------------------------------------------------
def test_one_load_fills_both_stores_with_the_same_ids(kb, rag_store):
    load(kb, rag_store)
    neo_ids = {r["id"] for r in kb._read("MATCH (p:Passage) RETURN p.id AS id")}
    assert len(neo_ids) == 3 and rag_store.count() == 3
    assert rag_store.missing(list(neo_ids)) == []                                        # every Neo4j passage id is in Chroma
    assert consistency(kb, rag_store) == {"neo4j_passages": 3, "chroma_passages": 3, "sampled": 3, "sample_missing_in_chroma": 0}


def test_reloading_is_idempotent_in_both_stores_and_tops_up_chroma(kb, rag_store):
    load(kb, rag_store)
    load.__wrapped__ if hasattr(load, "__wrapped__") else None
    text = "DVAC press release dated 2026-10-12: A village officer was trapped and arrested for demanding a bribe of Rs.20,000/- in Nilgiris District."
    again = store_document(kb, rag_store, title="DVAC press release 12", doc_type="press_release", text=text, passages=[{"locator": "listing", "text": text}])
    assert again == "duplicate" and rag_store.count() == 3
    rag_store.col.delete(where={"doc_type": "press_release"})                              # Chroma lost a document...
    assert rag_store.count() == 2
    store_document(kb, rag_store, title="DVAC press release 12", doc_type="press_release", text=text, passages=[{"locator": "listing", "text": text}])
    assert rag_store.count() == 3                                                          # ...and the next load repaired it


# ---- query understanding ---------------------------------------------------------------------------
def test_understanding_finds_entities_intent_and_resolves_followups(kb, rag_store, memory):
    load(kb, rag_store)
    u = graphrag.understand(kb, "Give me the timeline of the Test Housing Scam")
    assert [e["name"] for e in u.entities] == ["Test Housing Scam"] and u.intent == "timeline" and not u.followup
    first = graphrag.understand(kb, "List all cases of T. Person")
    assert first.entities[0]["name"] == "Test Person" and first.intent == "cases"
    memory.remember("u1", "List all cases of T. Person", "...", entities=first.entities)
    follow = graphrag.understand(kb, "what about his bail?", memory.recall("u1"))
    assert follow.followup and follow.entities[0]["name"] == "Test Person" and "Test Person" in follow.resolved
    assert not graphrag.understand(kb, "what about his bail?", {"entities": []}).entities        # nothing to resolve against


# ---- the two branches ------------------------------------------------------------------------------
def test_graph_branch_returns_cited_facts_with_levels_and_second_hop(kb, rag_store):
    load(kb, rag_store)
    u = graphrag.understand(kb, "cases of Test Person")
    facts = graphrag.graph_items(kb, u)
    texts = " | ".join(f["text"] for f in facts)
    assert "Test Person is accused in Test Housing Scam (CC 1/2018): charged on 2019-05-01 - reported." in texts       # first hop
    assert "Second Accused is accused" in texts and "FIR registered" in texts                                          # second hop: co-accused, timeline
    assert all(f["source"]["url"] == "https://example.org/order1" and f["level"] in graphrag.LEVEL_BONUS for f in facts)


def test_vector_branch_returns_only_really_relevant_passages(kb, rag_store):
    load(kb, rag_store)
    u = graphrag.understand(kb, "village officer arrested for demanding a bribe")
    hits = graphrag.vector_items(rag_store, u)
    assert hits and hits[0]["source"]["title"] == "DVAC press release 12" and hits[0]["kind"] == "chunk"
    assert not graphrag.vector_items(rag_store, graphrag.understand(kb, "capital of Brazil"))


# ---- merge and rank ---------------------------------------------------------------------------------
def test_merge_puts_best_fact_and_best_passage_first_and_limits_per_source():
    src = lambda i: {"title": f"S{i}", "url": "", "label": "x", "source_id": f"s{i}"}   # noqa: E731
    facts = [{"kind": "fact", "text": f"fact {i}", "score": 0.9 - i * 0.01, "level": "reported", "source": src(1)} for i in range(4)]
    chunks = [{"kind": "chunk", "text": "chunk a", "score": 0.4, "level": None, "source": src(2)},
              {"kind": "chunk", "text": "fact 0", "score": 0.5, "level": None, "source": src(3)}]        # duplicate text of fact 0
    merged = graphrag.merge(facts, chunks, max_items=6, per_source=2)
    assert [m["text"] for m in merged[:2]] == ["fact 0", "chunk a"]                                         # one of each kind first
    assert sum(1 for m in merged if m["source"]["source_id"] == "s1") == 3                                 # at most 3 facts per document
    assert [m["text"] for m in merged].count("fact 0") == 1                                                 # duplicates dropped


# ---- the whole pipeline ----------------------------------------------------------------------------
def test_pipeline_merges_graph_facts_and_passages(kb, rag_store, memory):
    load(kb, rag_store)
    ctx = graphrag.build_context(kb, rag_store, "Tell me about Test Person and the housing board assets", memory=memory, session="u1")
    kinds = [i["kind"] for i in ctx.items]
    assert kinds[0] == "fact" and "chunk" in kinds[:3] and ctx.notes == []
    assert all(i["source"]["title"] and "level" in i for i in ctx.items)


def test_pipeline_works_with_only_one_branch(kb, rag_store):
    load(kb, rag_store)
    only_vectors = graphrag.build_context(None, rag_store, "village officer arrested for demanding a bribe")
    assert only_vectors.items and all(i["kind"] == "chunk" for i in only_vectors.items)
    only_graph = graphrag.build_context(kb, None, "cases of Test Person")
    assert only_graph.items and all(i["kind"] == "fact" for i in only_graph.items)


def test_a_failing_branch_does_not_lose_the_other(kb, rag_store, monkeypatch):
    load(kb, rag_store)
    monkeypatch.setattr(graphrag, "vector_items", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("chroma down")))
    ctx = graphrag.build_context(kb, rag_store, "cases of Test Person")
    assert ctx.items and all(i["kind"] == "fact" for i in ctx.items) and ctx.notes == ["vector search unavailable"]


def test_context_is_cached_and_new_data_invalidates_the_cache(kb, rag_store, memory, monkeypatch):
    load(kb, rag_store)
    first = graphrag.build_context(kb, rag_store, "cases of Test Person", memory=memory)
    assert "cache" not in first.notes
    calls = []
    monkeypatch.setattr(graphrag, "graph_items", lambda *a, **k: calls.append(1) or [])
    second = graphrag.build_context(kb, rag_store, "cases of Test Person", memory=memory)
    assert "cache" in second.notes and calls == [] and [i["text"] for i in second.items] == [i["text"] for i in first.items]
    memory.bump_version()                                                                 # a new document arrived
    graphrag.build_context(kb, rag_store, "cases of Test Person", memory=memory)
    assert calls == [1]                                                                    # recomputed, not served stale


def test_nothing_relevant_gives_an_empty_context(kb, rag_store):
    load(kb, rag_store)
    assert graphrag.build_context(kb, rag_store, "hii").items == []
    assert graphrag.build_context(kb, rag_store, "What is the capital of Brazil?").items == []


def test_reindex_refills_chroma_from_neo4j(kb, rag_store):
    from backend.knowledge.reindex import reindex
    load(kb, rag_store)
    rag_store.col.delete(where={"doc_type": "order"})
    rag_store.col.delete(where={"doc_type": "reference"})
    assert rag_store.count() == 1
    written = reindex(kb, rag_store, batch=2, progress=lambda *a, **k: None)
    assert written == 2 and rag_store.count() == 3
    assert consistency(kb, rag_store)["sample_missing_in_chroma"] == 0
    assert reindex(kb, rag_store, progress=lambda *a, **k: None) == 0                       # nothing left to do
    hit = rag_store.query("longest river Tamil Nadu delta farming", k=1)[0]                  # metadata came across too
    assert hit["meta"]["title"] == "Rivers of Tamil Nadu (Wikipedia)" and hit["meta"]["doc_type"] == "reference"


# ---- domain check (thresholds measured on the real MiniLM embeddings: off-topic 0.00-0.53, in-domain 0.54-0.81) ----
def _chunk(score, label, text="a passage"):
    return {"kind": "chunk", "text": text, "score": score, "level": None, "source": {"title": "T", "url": "", "label": label, "source_id": text}}


def test_in_domain_rules():
    wiki, order = "reference (Wikipedia)", "High Court order"
    assert graphrag.in_domain("a bribe case", [_chunk(0.3, wiki)])                                # corruption wording
    assert graphrag.in_domain("Enforcement Directorate bail petition", [_chunk(0.74, order)])      # hits an official document
    assert graphrag.in_domain("who is he", [{"kind": "fact", "text": "x", "score": 0.9, "level": "reported", "source": {"label": wiki}}])
    assert not graphrag.in_domain("Who won the latest FIFA World Cup?", [_chunk(0.48, wiki)])      # only a weak Wikipedia hit
    assert not graphrag.in_domain("latest news about the stock market", [_chunk(0.53, wiki), _chunk(0.49, order)])  # official hit below 0.5


def test_off_domain_questions_keep_only_strong_hits(kb, rag_store, monkeypatch):
    load(kb, rag_store)
    wiki = "reference (Wikipedia)"
    weak = [_chunk(0.48, wiki, "south america passage"), _chunk(0.52, wiki, "africa passage")]
    monkeypatch.setattr(graphrag, "vector_items", lambda *a, **k: list(weak))
    assert graphrag.build_context(kb, rag_store, "Who won the latest FIFA World Cup?").items == []      # junk notes dropped -> web / model gets its turn
    strong = weak + [_chunk(0.80, wiki, "kaveri river passage")]
    monkeypatch.setattr(graphrag, "vector_items", lambda *a, **k: list(strong))
    kept = graphrag.build_context(kb, rag_store, "Tell me about the Kaveri river").items
    assert [i["text"] for i in kept] == ["kaveri river passage"]                                       # a strong general-knowledge hit is still used
    monkeypatch.setattr(graphrag, "vector_items", lambda *a, **k: list(weak))
    assert len(graphrag.build_context(kb, rag_store, "Tell me about the 2G scam arrests").items) == 2   # corruption wording: weak hits stay


def test_loosely_related_passages_are_not_used_when_the_questions_wording_is_mostly_absent(kb, rag_store):
    """Found live: 'Panama papers scam arrests in Kerala' got general scam passages and the model invented an answer.
    A passage must cover most of the question's words (55%), unless it is very close in meaning."""
    load(kb, rag_store)
    vague = graphrag.build_context(kb, rag_store, "Panama papers scam arrests in Kerala bribe")      # only 2 of 6 words are found in the stored text
    assert vague.items == []
    specific = graphrag.build_context(kb, rag_store, "village officer arrested for demanding a bribe")
    assert specific.items and "bribe" in specific.items[0]["text"]


def test_cached_contexts_from_older_ranking_rules_are_not_served(kb, rag_store, memory, monkeypatch):
    """Found live: a context cached before the thresholds changed kept being served afterwards."""
    load(kb, rag_store)
    graphrag.build_context(kb, rag_store, "village officer arrested for demanding a bribe", memory=memory)
    assert "cache" in graphrag.build_context(kb, rag_store, "village officer arrested for demanding a bribe", memory=memory).notes
    monkeypatch.setattr(graphrag, "LOGIC_VERSION", graphrag.LOGIC_VERSION + 1)        # the rules changed
    assert "cache" not in graphrag.build_context(kb, rag_store, "village officer arrested for demanding a bribe", memory=memory).notes
