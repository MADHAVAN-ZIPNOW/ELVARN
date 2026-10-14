"""Folder ingest and review-queue tests (need a disposable Neo4j, see the `kb` fixture). Names and cases are FICTIONAL."""
import json

import pytest

from backend.knowledge import review
from backend.knowledge.ingest_folder import ingest_inbox

JUDGMENT = (
    "The FIR was registered on 12.03.2018 under Section 13(1)(e) of the Prevention of Corruption Act against the petitioner. "
    "The chargesheet was filed on 4 June 2019. The trial court convicted the accused on 5 January 2021. "
    "The weather in Chennai was pleasant that day."
)


def candidates_by_stage(kb):
    return {r["c"]["stage"]: r["c"] for r in kb.pending_candidates(100)}


def test_inbox_loads_text_once_with_sidecar_details(kb, tmp_path):
    (tmp_path / "judgment").mkdir()
    f = tmp_path / "judgment" / "a.txt"
    f.write_text(JUDGMENT, encoding="utf-8")
    (tmp_path / "judgment" / "a.txt.json").write_text(json.dumps({"title": "Test v. State", "url": "https://example.org/j", "published_date": "2021-02-02"}))
    (tmp_path / "nonsense").mkdir()
    (tmp_path / "nonsense" / "x.txt").write_text("x")
    out = ingest_inbox(kb, tmp_path)
    assert any(n > 0 and msg == "ok" for _, n, msg in out) and any("unknown document type" in m for _, _, m in out)
    src = kb._read("MATCH (s:Source) RETURN s.title AS title, s.doc_type AS t, s.url AS url")[0]
    assert (src["title"], src["t"], src["url"]) == ("Test v. State", "judgment", "https://example.org/j")
    assert any(m == "already loaded" for _, _, m in ingest_inbox(kb, tmp_path))


def test_pdf_is_read_with_page_locators(kb, tmp_path):
    pytest.importorskip("pypdf")
    from backend.files.pdfwriter import build_pdf
    from backend.files.readers import Doc
    d = Doc("txt")
    d.add("p", "The accused was acquitted by the High Court.")
    pdf, _ = build_pdf(d)
    (tmp_path / "order").mkdir()
    (tmp_path / "order" / "o.pdf").write_bytes(pdf)
    out = ingest_inbox(kb, tmp_path)
    assert out and out[0][1] >= 1, out
    assert kb._read("MATCH (p:Passage) RETURN p.locator AS l")[0]["l"] == "page 1"


def test_extract_proposes_with_dates_and_levels(kb, tmp_path):
    (tmp_path / "judgment").mkdir()
    (tmp_path / "judgment" / "a.txt").write_text(JUDGMENT, encoding="utf-8")
    ingest_inbox(kb, tmp_path)
    assert review.extract(kb) >= 3
    rows = candidates_by_stage(kb)
    assert rows["fir"]["event_date"] == "2018-03-12" and "Prevention of Corruption Act" in rows["fir"]["sections"]
    assert rows["chargesheet"]["event_date"] == "2019-06-04"
    assert rows["final_judgment"]["level"] == "court_finding" and rows["final_judgment"]["status"] == "convicted"
    assert not any("weather" in c["sentence"] for c in rows.values())
    assert review.extract(kb) == 0                                   # idempotent


def test_nothing_reaches_answers_until_approved(kb, tmp_path):
    (tmp_path / "judgment").mkdir()
    (tmp_path / "judgment" / "a.txt").write_text(JUDGMENT, encoding="utf-8")
    ingest_inbox(kb, tmp_path)
    review.extract(kb)
    kb.entity("Person", "Test Person")
    assert not kb.answer("Test Person")["found"]
    uid = candidates_by_stage(kb)["final_judgment"]["uid"]
    review.approve(kb, uid, "Test Case", person="Test Person")
    out = kb.answer("cases of Test Person")
    assert out["found"] and "convicted" in out["text"] and "court finding" in out["text"]
    assert kb.audit() == []
    other = kb.pending_candidates(1)[0]["c"]["uid"]
    review.reject(kb, other)
    with pytest.raises(ValueError):
        review.approve(kb, other, "Test Case")                       # rejected ones cannot be approved
    with pytest.raises(ValueError):
        review.approve(kb, uid, "Test Case")                         # nor approved twice


def test_failed_approval_writes_nothing(kb, tmp_path):
    (tmp_path / "judgment").mkdir()
    (tmp_path / "judgment" / "a.txt").write_text(JUDGMENT, encoding="utf-8")
    ingest_inbox(kb, tmp_path)
    review.extract(kb)
    uid = candidates_by_stage(kb)["fir"]["uid"]
    with pytest.raises(ValueError):
        review.approve(kb, uid, "Test Case", person="Test Person", role="boss")      # invalid role fails the party step
    assert kb._read("MATCH (n) WHERE n:CaseEvent OR n:CaseParty OR n:Case RETURN count(n) AS n")[0]["n"] == 0
    assert any(r["c"]["uid"] == uid for r in kb.pending_candidates(100))              # still pending
