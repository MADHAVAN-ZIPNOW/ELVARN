"""RAG evaluation: scoring with mocked generation, and a run on a tiny random model."""
import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "eval"))
sys.path.insert(0, str(BACKEND / "data_tools"))

import run_rag_eval as ev  # noqa: E402
from conftest import write_checkpoint  # noqa: E402

NOTES = ["The institute was founded by Maria Okonkwo in 1987 after a long survey of the northern valley.",
         "Visitors can see a public garden with more than 40 kinds of local trees."]
CASE = {"id": 1, "type": "single", "mode": "domain", "topic": "T", "notes": NOTES, "question": "Who founded the institute?",
        "answerable": True, "groups": [["1987"], ["Maria"]], "reference_answer": "x"}
NEG = {"id": 2, "type": "negative", "mode": "english", "topic": "T", "notes": NOTES, "question": "What happened in 2013?",
       "answerable": False, "groups": [], "reference_answer": "The notes do not say."}


def test_good_answer_scores_full():
    r = ev.score_case(CASE, "According to [1]: Maria Okonkwo founded it in 1987.")
    assert r["key_fact_coverage"] == 1.0 and r["cites_a_note"] and not r["refused"] and not r["invented_any"]


def test_partial_and_invented():
    r = ev.score_case(CASE, "It was founded in 1987 by John Smith in 1995.")
    assert r["key_fact_coverage"] == 0.5
    assert r["invented_any"] and "Smith" in r["invented"] and "1995" in r["invented"]


def test_citation_must_exist():
    assert not ev.score_case(CASE, "Maria Okonkwo 1987 [5]")["cites_a_note"]
    assert not ev.score_case(CASE, "Maria Okonkwo 1987")["cites_a_note"]


@pytest.mark.parametrize("text", ["The notes do not say.", "I cannot find that in the notes.", "Not mentioned in the notes.",
                                  "The notes don't say.", "The notes do not contain this information."])
def test_refusals_detected(text):
    assert ev.is_refusal(text)
    r = ev.score_case(NEG, text)
    assert r["refusal_correct"] is True and not r["invented_any"]


def test_wrong_answer_on_not_in_notes_case_and_false_refusal():
    assert ev.score_case(NEG, "In 2013 Maria Okonkwo left.")["refusal_correct"] is False
    assert ev.score_case(CASE, "The notes do not say.")["refused"] is True


def test_repetition_measure_used():
    r = ev.score_case(CASE, "the cat sat on the mat the cat sat on the mat the cat sat on the mat")
    assert r["repetition"] > 0.3


def test_summary_with_mocked_generation():
    answers = {1: "According to [1]: Maria Okonkwo founded it in 1987.", 2: "The notes do not say."}
    rows = ev.run_cases([CASE, NEG], lambda c: answers[c["id"]])
    s = ev.summarise(rows)
    assert s["cases"] == 2 and s["key_fact_coverage_pct"] == 100.0 and s["refusal_correct_pct"] == 100.0
    assert s["false_refusal_pct"] == 0.0 and s["invented_rate_pct"] == 0.0 and s["citation_rate_pct"] == 100.0
    bad = ev.summarise(ev.run_cases([CASE, NEG], lambda c: "Zorblax 77 is great."))
    assert bad["key_fact_coverage_pct"] == 0.0 and bad["refusal_correct_pct"] == 0.0 and bad["invented_rate_pct"] == 100.0


def test_shipped_cases_file_is_valid_and_from_validation_topics():
    import build_rag_dataset
    cases = ev.load_cases()
    assert len(cases) >= 50 and any(not c["answerable"] for c in cases)
    for c in cases:
        assert c["notes"] and c["question"] and c["mode"] in ("domain", "english", "chat")
        assert c["answerable"] == bool(c["groups"])
        assert build_rag_dataset.is_val_topic(c["topic"])


def test_end_to_end_on_tiny_random_model_and_never_overwrites(tmp_path, tok, capsys):
    write_checkpoint(tmp_path / "ck", tok)
    (tmp_path / "ck" / "tokenizer.json").write_bytes(tok.path.read_bytes())
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps({"cases": [CASE, NEG]}), encoding="utf-8")
    out = tmp_path / "res.json"
    res = ev.main([str(tmp_path / "ck"), "--cases", str(cases_file), "--max-new-tokens", "6", "--save", str(out)])
    saved = json.loads(out.read_text(encoding="utf-8"))
    (key,) = saved
    assert saved[key]["summary"]["cases"] == 2 and saved[key]["info"]["sequence_length"] == 32
    assert saved[key]["info"]["context_too_short"] == 2          # the tiny 32-token model cannot hold the notes
    assert "Saved" in capsys.readouterr().out and res
    with pytest.raises(SystemExit):
        ev.main([str(tmp_path / "ck"), "--cases", str(cases_file), "--save", str(out)])
