"""Training export (needs the disposable Neo4j, see the `kb` fixture). The exported file must be readable by the project's own
training code. All names and cases are FICTIONAL."""
import json
import sys
from pathlib import Path

import pytest

import corpus  # backend/training (on sys.path via conftest)
from knowledge.export_training import export

JUDGMENT = ("The Special Court framed charges against the accused under the Prevention of Corruption Act. The prosecution alleged that the accused "
            "possessed assets disproportionate to his known sources of income. The accused was working as an executive engineer in the public works "
            "department of the State at the relevant time. The investigation was conducted by the Directorate of Vigilance and Anti-Corruption.")
PRESS = ("DVAC press release dated 2026-10-12: A village administrative officer was trapped and arrested while accepting a bribe of Rs.20,000/- "
         "from a farmer for issuing a land ownership certificate in Nilgiris District. The accused officer was produced before the court and remanded to judicial custody for fifteen days.")
WIKI = "A long Wikipedia style paragraph about the river Kaveri and the districts it flows through in the southern part of India. " * 3


def seed(kb):
    kb.add_document("Test v. State (Madras High Court)", "order", JUDGMENT, [{"locator": "page 1", "text": JUDGMENT}], url="https://example.org/o1", published_date="2019-05-02")
    kb.add_document("DVAC press release 12", "press_release", PRESS, [{"locator": "listing", "text": PRESS}], url="https://example.org/pr12", published_date="2026-10-12")
    kb.add_document("Kaveri (Wikipedia)", "reference", WIKI, [{"locator": "chunk 1", "text": WIKI}], url="https://en.wikipedia.org/wiki/Kaveri")
    garbled = "�� " * 80
    kb.add_document("Garbled order", "order", garbled, [{"locator": "page 1", "text": garbled}], url="https://example.org/bad")
    pid = kb._read("MATCH (p:Passage)-[:FROM]->(s:Source {doc_type:'order', title:'Test v. State (Madras High Court)'}) RETURN p.id AS id")[0]["id"]
    person = kb.entity("Person", "Test Person")
    case = kb.add_case("Test Housing Scam", case_number="CC 1/2018")
    kb.add_party(case, person, "accused", "alleged", pid, status="charged", status_date="2019-05-01")
    kb.add_event(case, "chargesheet", "Charge sheet filed", "reported", pid, event_date="2018-09-01")


def test_export_contains_official_passages_and_cited_fact_blocks_only(kb, tmp_path):
    seed(kb)
    out = tmp_path / "kb_corpus.txt"
    info = export(kb, out, tmp_path / "training.json", known={"kaveri"})
    text = out.read_text(encoding="utf-8")
    assert info["passage_blocks"] == 2 and info["fact_blocks"] == 2
    assert "Garbled" not in text and "Kaveri" not in text                              # garbled text, and an article the model already knows, are left out
    assert "Source: https://example.org/o1" in text and "Source: https://example.org/pr12" in text
    # approved facts are written with their assertion level, never as bare claims
    assert "Test Person" in text and "(alleged, source: Test v. State (Madras High Court))" in text and "Allegations are not findings of guilt." in text
    assert "Charge sheet filed (chargesheet; reported)" in text


def test_short_press_release_lines_are_grouped_by_month(kb, tmp_path):
    for i in range(4):
        line = f"DVAC press release dated 2026-10-1{i}: Officer number {i} of the revenue department was trapped and arrested while accepting a bribe in Salem District."
        kb.add_document(f"DVAC press release {i}", "press_release", line, [{"locator": "listing", "text": line}], url=f"https://example.org/pr{i}",
                        published_date=f"2026-10-1{i}", publisher="DVAC")
    text = (export(kb, tmp_path / "kb_corpus.txt", tmp_path / "s.json"), (tmp_path / "kb_corpus.txt").read_text(encoding="utf-8"))[1]
    assert text.count("<DOMAIN_KNOWLEDGE>") == 1 and "DVAC press releases, 2026-10" in text            # four lines -> one multi-sentence passage
    assert all(f"Officer number {i}" in text for i in range(4)) and "Source: https://example.org/pr0" in text


def test_export_is_deterministic_atomic_and_records_state(kb, tmp_path):
    seed(kb)
    a = export(kb, tmp_path / "a.txt", tmp_path / "s.json")
    b = export(kb, tmp_path / "b.txt", tmp_path / "s.json")
    assert a["sha256"] == b["sha256"] and not list(tmp_path.glob("*.part"))
    state = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert state["last_export"]["passage_blocks"] == 2


def test_export_is_readable_by_the_projects_own_training_code(kb, tmp_path):
    seed(kb)
    export(kb, tmp_path / "kb_corpus.txt", tmp_path / "s.json")
    docs = corpus.load_documents(tmp_path)                                              # language-model data path
    assert len(docs) == 4 and all(d["category"] == "DOMAIN_KNOWLEDGE" for d in docs)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_tools"))
    import build_rag_dataset as brd                                                    # RAG fine-tune data path (reads kb_*.txt)
    passages = brd.load_wikipedia_passages(tmp_path)
    topics = {p["topic"] for p in passages}
    assert "Test v. State (Madras High Court)" in topics and any(t.endswith("press releases, 2026-10") for t in topics)
    assert all(len(p["notes"]) >= 2 for p in passages)


def test_nothing_to_export_gives_an_empty_but_valid_file(kb, tmp_path):
    info = export(kb, tmp_path / "kb_corpus.txt", tmp_path / "s.json")
    assert info["passage_blocks"] == 0 and info["fact_blocks"] == 0


def test_wikipedia_articles_new_to_the_model_are_included_known_ones_are_not(kb, tmp_path):
    seed(kb)
    new_text = "Kalaignar TV is a Tamil language television channel based in Chennai that was launched in 2007 by a company linked to a political family. " * 2
    kb.add_document("Kalaignar TV (Wikipedia)", "reference", new_text, [{"locator": "chunk 1", "text": new_text}], url="https://en.wikipedia.org/wiki/Kalaignar_TV", publisher="Wikipedia")
    kb.add_document("ELVARN note (ELVARN project notes)", "reference", new_text + "x", [{"locator": "note", "text": new_text + "x"}], publisher="ELVARN project domain notes")
    info = export(kb, tmp_path / "kb_corpus.txt", tmp_path / "s.json", known={"kaveri"})
    text = (tmp_path / "kb_corpus.txt").read_text(encoding="utf-8")
    assert info["reference_blocks"] == 1 and "Topic: Kalaignar TV\n" in text and "Source: https://en.wikipedia.org/wiki/Kalaignar_TV" in text
    assert "Topic: Kaveri" not in text and "project notes" not in text                      # known article and the project's own notes are skipped
    from knowledge.export_training import known_topics
    (tmp_path / "wikipedia_domain.txt").write_text("<DOMAIN_KNOWLEDGE>\nTopic: Kalaignar TV\nText: x\n</DOMAIN_KNOWLEDGE>", encoding="utf-8")
    assert "kalaignar tv" in known_topics(tmp_path)                                           # read from the existing training files
    assert export(kb, tmp_path / "kb_corpus.txt", tmp_path / "s.json")["reference_blocks"] == 0         # default: uses those files -> already known
