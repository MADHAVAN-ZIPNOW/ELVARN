"""Knowledge graph tests (need a disposable Neo4j, see the `kb` fixture in conftest). All names and cases are FICTIONAL."""
from pathlib import Path

import pytest

from knowledge.kb import NOT_FOUND

ONTOLOGY = Path(__file__).resolve().parent.parent / "knowledge" / "ontology.ttl"


def seed(kb):
    src = kb.add_source("Test judgment A v. State", "judgment", text="judgment text", url="https://example.org/a", published_date="2020-01-02")
    p1 = kb.add_passage(src, 0, "Accused Test Person was charged under the Prevention of Corruption Act.", locator="para 4")
    p2 = kb.add_passage(src, 1, "The trial court framed charges on 2019-05-01.", locator="para 9")
    person = kb.entity("Person", "Test Person", aliases=["T. Person"])
    case = kb.add_case("Test Housing Scam", case_number="CC 1/2018", year_registered=2018)
    kb.add_party(case, person, "accused", "alleged", p1, status="charged", status_date="2019-05-01")
    kb.add_event(case, "trial", "Charges framed", "reported", p2, event_date="2019-05-01")
    kb.add_event(case, "fir", "FIR registered", "reported", p1, event_date="2018-03-10")
    return person, case, p1


def test_answer_lists_cases_with_citation(kb):
    seed(kb)
    out = kb.answer("List all corruption cases of T. Person")
    assert out["found"]
    assert "Test Housing Scam" in out["text"] and "charged" in out["text"] and "[1]" in out["text"]
    assert out["sources"][0]["url"] == "https://example.org/a"
    assert "not findings of guilt" in out["text"]


def test_timeline_is_in_date_order(kb):
    seed(kb)
    text = kb.answer("What happened in the Test Housing Scam?")["text"]
    assert text.index("FIR registered") < text.index("Charges framed")


def test_unknown_question_says_not_found(kb):
    seed(kb)
    assert kb.answer("Tell me about Someone Else")["text"] == NOT_FOUND


def test_names_match_whole_words_only(kb):
    seed(kb)
    kb.entity("Person", "Raj")
    assert [e["name"] for e in kb.find_entities("the Rajasthan case")] == []
    assert [e["name"] for e in kb.find_entities("cases of Raj please")] == ["Raj"]


def test_fact_must_cite_an_existing_passage(kb):
    person, case, _ = seed(kb)
    with pytest.raises(ValueError):
        kb.add_event(case, "trial", "x", "reported", "no-such-passage")
    assert kb.audit() == []


def test_invalid_values_rejected_and_nothing_half_written(kb):
    person, case, pid = seed(kb)
    before = kb._read("MATCH (n:CaseParty) RETURN count(n) AS n")[0]["n"]
    for bad in ({"level": "proven"}, {"status": "guilty"}, {"role": "boss"}):
        args = {"role": "accused", "level": "alleged", "status": None, **bad}
        with pytest.raises(ValueError):
            kb.add_party(case, person, args["role"], args["level"], pid, status=args["status"])
    with pytest.raises(ValueError):
        kb.add_event(case, "somewhere", "x", "reported", pid)
    with pytest.raises(ValueError):
        kb.add_source("x", "blog", text="t")
    assert kb._read("MATCH (n:CaseParty) RETURN count(n) AS n")[0]["n"] == before


def test_reingesting_same_text_is_idempotent(kb):
    a = kb.add_source("Doc", "news", text="same text")
    b = kb.add_source("Doc again", "news", text="same text")
    assert a == b and kb.source_exists("same text") and not kb.source_exists("other")


def test_court_finding_needs_a_judgment_or_order(kb):
    person, case, _ = seed(kb)
    news = kb.add_source("Newspaper story", "news", text="news text")
    news_p = kb.add_passage(news, 0, "X was convicted, the paper says.")
    with pytest.raises(ValueError):
        kb.add_party(case, person, "accused", "court_finding", news_p, status="convicted")
    with pytest.raises(ValueError):
        kb.add_event(case, "final_judgment", "Convicted", "court_finding", news_p)
    order = kb.add_source("Order", "order", text="order text")
    order_p = kb.add_passage(order, 0, "The accused is convicted.")
    kb.add_party(case, person, "accused", "court_finding", order_p, status="convicted")      # allowed
    assert kb.audit() == []


def test_case_fields_new_source_types_and_statuses(kb):
    person, case, pid = seed(kb)
    for t in ("press_release", "audit_report", "affidavit", "rti_response"):
        kb.add_source(t, t, text=t)
    kb.add_case("Another Case", fir_ecr_number="ECR/1/2020", department="PWD", location="Chennai", amount_inr=5_000_000, confidence="medium")
    row = kb._read("MATCH (c:Case {name:'Another Case'}) RETURN c.department AS d, c.amount_inr AS a")[0]
    assert (row["d"], row["a"]) == ("PWD", 5_000_000)
    with pytest.raises(ValueError):
        kb.add_case("Bad", confidence="certain")
    kb.add_party(case, person, "accused", "alleged", pid, status="allegation_only")
    kb.add_party(case, person, "accused", "reported", pid, status="registered")


def test_add_case_update_keeps_other_fields(kb):
    kb.add_case("Keep Case", case_number="CC 9/2020", department="PWD")
    kb.add_case("Keep Case", location="Madurai")
    row = kb._read("MATCH (c:Case {name:'Keep Case'}) RETURN c.case_number AS n, c.department AS d, c.location AS l")[0]
    assert (row["n"], row["d"], row["l"]) == ("CC 9/2020", "PWD", "Madurai")


def test_claim_needs_object_and_citation(kb):
    person, case, pid = seed(kb)
    office = kb.entity("Organisation", "Test Ministry")
    kb.add_claim(person, "holds_office", "reported", pid, object_id=office)
    kb.add_claim(person, "age", "reported", pid, object_value="55")
    with pytest.raises(ValueError):
        kb.add_claim(person, "nothing", "reported", pid)
    assert kb.audit() == []


def test_audit_detects_an_uncited_fact(kb):
    seed(kb)
    kb._run("CREATE (:CaseEvent {id:'rogue', stage:'trial', description:'x', level:'reported'})")
    assert any("without a cited passage" in p for p in kb.audit())


def test_export_backup_counts(kb, tmp_path):
    seed(kb)
    out = kb.export_jsonl(tmp_path / "backup.jsonl")
    assert out["nodes"] > 5 and out["relationships"] > 5 and (tmp_path / "backup.jsonl").stat().st_size > 0


def test_fulltext_search_finds_passages(kb):
    seed(kb)
    hits = kb.search_passages("Prevention of Corruption")
    assert hits and hits[0]["title"] == "Test judgment A v. State"


def test_wikipedia_ingest_chunks_and_is_idempotent(kb):
    from backend.knowledge.ingest_wikipedia import chunk, ingest_article
    text = "\n\n".join(f"Paragraph {i} on a Tamil Nadu corruption case. " + "word " * 60 for i in range(10)) + "\n\n== References ==\n"
    parts = chunk(text, size=700)
    assert len(parts) > 1 and all("References" not in p for p in parts)
    assert ingest_article(kb, "Test Scam", text) == len(parts)
    assert ingest_article(kb, "Test Scam", text) == 0          # same text again: nothing added
    row = kb._read("MATCH (s:Source) RETURN s.doc_type AS t, s.licence AS l, s.url AS u")[0]
    assert (row["t"], row["l"]) == ("reference", "CC BY-SA 4.0") and row["u"].endswith("Test_Scam")


def test_only_tamil_nadu_corruption_articles_are_ingested(kb):
    from backend.knowledge.ingest_wikipedia import ingest_article, is_relevant
    other_state = "A long article about a bribery scam in Gujarat. " * 20
    tn_no_corruption = "A long article about the rivers of Tamil Nadu and Chennai weather. " * 20
    tn_corruption = "The Vigilance department in Tamil Nadu filed a disproportionate assets case. " * 20
    assert not is_relevant(other_state) and not is_relevant(tn_no_corruption) and is_relevant(tn_corruption)
    assert ingest_article(kb, "Gujarat scam", other_state) == 0
    assert ingest_article(kb, "TN rivers", tn_no_corruption) == 0
    assert ingest_article(kb, "TN DA case", tn_corruption) > 0


def test_ontology_parses():
    rdflib = pytest.importorskip("rdflib")
    g = rdflib.Graph()
    g.parse(str(ONTOLOGY), format="turtle")
    assert len(g) > 100


def test_wikipedia_run_uses_its_cache_folder_and_clears_the_cache_once(kb, monkeypatch):
    """Regression: the folder variable was reassigned to the Redis client, so the nightly Wikipedia task would have crashed."""
    import fakeredis
    from backend.knowledge import ingest_wikipedia as iw
    from backend.usermemory.store import Memory
    mem = Memory(fakeredis.FakeRedis(decode_responses=True))
    folders = []

    def fake_fetch(title, cache_dir):
        folders.append(cache_dir)
        return {"title": title, "text": f"The Vigilance department in Tamil Nadu filed a disproportionate assets case against officer {title}. " * 20}
    monkeypatch.setattr(iw.wiki, "fetch_text", fake_fetch)
    monkeypatch.setattr(iw.wiki, "category_titles", lambda *a, **k: [])
    monkeypatch.setattr(iw, "SEED_TITLES", ["Case A", "Case B"])
    monkeypatch.setattr(Memory, "from_env", classmethod(lambda cls: mem))
    out = iw.run(kb=kb)
    assert out["articles"] == 2
    assert folders and all(hasattr(f, "mkdir") for f in folders)                     # a folder, not the Redis client
    assert mem.version() == 1                                                          # cache cleared once at the end, not once per article
