"""Collectors, fetcher, processing. Offline: fixtures are shaped like the real DVAC / CBI pages. Names are FICTIONAL.
Tests marked with the `kb` fixture need a disposable Neo4j (NEO4J_TEST_URI)."""
import io
import urllib.error

import pytest

from backend.pipeline import process
from backend.pipeline.collectors import cbi, dvac
from backend.pipeline.common import FetchError, Fetcher, ListSink, RawStore, SeenStore, in_window, is_tamil_nadu, make_message

DVAC_LIST = """
<ul>
<li style="color:#00F"><a href="pdf/Press_Release/2026/12-10-2026-Trap - Press Release-106 - Tr.A.Test, Panchayat Secretary.pdf" target="_blank">V&amp;AC - Press Release-106 - Trapped and arrested, Tr.A.Test, Panchayat Secretary, Nilgiris District for demand and acceptance of bribe Rs.20,000/-</a></li>
<li><a href="pdf/Press_Release/2017/03-02-2017-Judgement - Tr.B.Test.pdf">V&amp;AC - Judgement - Tr.B.Test, convicted by Special Court, Chennai</a></li>
<li><a href="pdf/Press_Release/2017/03-02-2017-Judgement - Tr.B.Test.pdf">duplicate link</a></li>
<li><a href="pdf/Press_Release/1990/01-01-1990-Old.pdf">An old release</a></li>
</ul>"""

CBI_LIST = """<a href="https://cbi.gov.in/press-detail/AAA1">CBI Arrests Official in Chennai Bribery Case</a>
<a href="https://cbi.gov.in/press-detail/BBB2">CBI Arrests Official in Bhopal</a>
<a href="https://cbi.gov.in/press-detail/CCC3">Broken page</a>"""


def cbi_detail(date, title, body):
    return (f'<div class="date"><div class="date-back"><p>{date}</p></div><h4>{title}</h4></div>'
            f'<div class="press-txt"><p>{body}</p><p>Second paragraph.</p></div>').encode()


class Resp:
    def __init__(self, body, ctype="text/html"):
        self.body, self.headers = body, {"Content-Type": ctype}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return self.body if n < 0 else self.body[:n]


def fake_web(pages):
    """open_fn serving {url: bytes | Exception}; unknown urls are 404."""
    calls = []

    def open_fn(req, timeout=None):
        url = req.full_url
        calls.append(url)
        page = pages.get(url)
        if isinstance(page, Exception):
            raise page
        if page is None:
            raise urllib.error.HTTPError(url, 404, "not found", {}, io.BytesIO())
        return Resp(page)
    open_fn.calls = calls
    return open_fn


def fetcher_for(pages, **kw):
    sleeps = []
    f = Fetcher(delay=kw.pop("delay", 0), open_fn=fake_web(pages), sleep=sleeps.append, **kw)
    f.sleeps = sleeps
    return f


# ---- helpers ------------------------------------------------------------------------------------
def test_date_window_and_tamil_nadu_terms():
    assert in_window("1947-08-15") and in_window(2026) and not in_window("1946-12-31") and not in_window("2027-01-01") and not in_window(None)
    assert is_tamil_nadu("raid in Coimbatore") and is_tamil_nadu("the DMK government") and not is_tamil_nadu("Bhopal CGST office")


def test_fetcher_honours_robots_missing_robots_and_user_agent():
    blocked = fetcher_for({"https://x.test/robots.txt": b"User-agent: *\nDisallow: /private/\n", "https://x.test/private/a": b"secret", "https://x.test/ok": b"fine"})
    assert blocked.get("https://x.test/ok")[0] == b"fine"
    with pytest.raises(FetchError):
        blocked.get("https://x.test/private/a")
    open_fn = fake_web({"https://y.test/page": b"x"})              # no robots.txt (404): allowed
    f = Fetcher(delay=0, open_fn=open_fn, sleep=lambda s: None)
    assert f.get("https://y.test/page")[0] == b"x"
    from backend.pipeline.common import USER_AGENT
    assert "@" not in USER_AGENT                                    # no personal details are sent to the sites


def test_fetcher_rate_limits_retries_and_caps_size():
    f = fetcher_for({"https://z.test/a": b"a", "https://z.test/b": b"b"}, delay=5)
    f.get("https://z.test/a")
    f.get("https://z.test/b")
    assert any(s > 0 for s in f.sleeps)                             # it waited between requests
    flaky = {"n": 0}

    def open_fn(req, timeout=None):
        if req.full_url.endswith("robots.txt"):
            raise urllib.error.HTTPError(req.full_url, 404, "x", {}, io.BytesIO())
        flaky["n"] += 1
        if flaky["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, io.BytesIO())
        return Resp(b"finally")
    assert Fetcher(delay=0, open_fn=open_fn, sleep=lambda s: None).get("https://q.test/x")[0] == b"finally"
    with pytest.raises(FetchError):
        fetcher_for({"https://big.test/f": b"x" * 100}, max_bytes=10).get("https://big.test/f")


def test_raw_store_is_content_addressed_and_seen_store_persists(tmp_path):
    store = RawStore(tmp_path / "raw")
    a = store.save(b"same", ".pdf")
    assert a == store.save(b"same", ".pdf") and (tmp_path / "raw" / a[1]).read_bytes() == b"same"
    s = SeenStore("t", root=tmp_path)
    s.add("u1")
    s.save()
    assert "u1" in SeenStore("t", root=tmp_path) and "u2" not in SeenStore("t", root=tmp_path)


# ---- DVAC ---------------------------------------------------------------------------------------
def test_dvac_parse_list():
    items = dvac.parse_list(DVAC_LIST)
    assert len(items) == 3                                          # the repeated link is dropped
    first = items[0]
    assert first["date"] == "2026-10-12" and "Trapped and arrested, Tr.A.Test" in first["title"] and "&amp;" not in first["title"]
    assert first["url"].startswith("https://dvac.tn.gov.in/pdf/Press_Release/2026/") and "%20" in first["url"]


def test_dvac_collect_is_incremental_and_respects_window(tmp_path):
    pages = {dvac.LIST_URL: DVAC_LIST.encode()}
    for it in dvac.parse_list(DVAC_LIST):
        pages[it["url"]] = b"%PDF-1.4 fake " + it["filename"].encode()
    sink, seen, store = ListSink(), SeenStore("dvac", root=tmp_path), RawStore(tmp_path / "raw")
    stats = dvac.collect(fetcher_for(pages), store, sink, seen, min_year=2000)
    assert stats["published"] == 2 and stats["out_of_window"] == 1
    m = sink.messages[0]
    assert m["doc_type"] == "press_release" and m["text"].startswith("DVAC press release dated 2026-10-12") and m["raw_path"].endswith(".pdf")
    assert (tmp_path / "raw" / m["raw_path"]).exists() and m["extra"]["pdf_text_reliable"] is False
    again = dvac.collect(fetcher_for(pages), store, ListSink(), SeenStore("dvac", root=tmp_path), min_year=2000)
    assert again["published"] == 0 and again["skipped_seen"] == 2


def test_dvac_pdf_failure_still_collects_the_english_title(tmp_path):
    pages = {dvac.LIST_URL: DVAC_LIST.encode()}                     # PDFs are 404
    sink = ListSink()
    stats = dvac.collect(fetcher_for(pages, retries=1), RawStore(tmp_path / "raw"), sink, SeenStore("d", root=tmp_path), min_year=2000)
    assert stats["published"] == 2 and stats["pdf_failed"] == 2 and sink.messages[0]["raw_path"] is None


# ---- CBI ----------------------------------------------------------------------------------------
def test_cbi_parse_and_collect_filters_tamil_nadu(tmp_path):
    items = cbi.parse_list(CBI_LIST)
    assert [i["url"] for i in items] == ["https://cbi.gov.in/press-detail/AAA1", "https://cbi.gov.in/press-detail/BBB2", "https://cbi.gov.in/press-detail/CCC3"]
    pages = {cbi.LIST_URL: CBI_LIST.encode(),
             items[0]["url"]: cbi_detail("13-10-2026", "CBI Arrests Official in Chennai Bribery Case", "The CBI arrested an official in Chennai."),
             items[1]["url"]: cbi_detail("13-10-2026", "CBI Arrests Official in Bhopal", "Case in Bhopal."),
             items[2]["url"]: b"<html>layout changed</html>"}
    sink, seen = ListSink(), SeenStore("cbi", root=tmp_path)
    stats = cbi.collect(fetcher_for(pages), RawStore(tmp_path / "raw"), sink, seen)
    assert (stats["published"], stats["not_tn"], stats["unparsed"]) == (1, 1, 1)
    m = sink.messages[0]
    assert m["published_date"] == "2026-10-13" and "Chennai" in m["text"] and m["doc_type"] == "press_release"
    assert items[0]["url"] in seen and items[1]["url"] in seen and items[2]["url"] not in seen   # unparsed pages are retried later


def test_cbi_out_of_window_is_not_published(tmp_path):
    pages = {cbi.LIST_URL: b'<a href="press-detail/OLD1">Old Chennai case</a>', "https://cbi.gov.in/press-detail/OLD1": cbi_detail("01-01-1930", "Old Chennai case", "Chennai.")}
    stats = cbi.collect(fetcher_for(pages), RawStore(tmp_path / "raw"), ListSink(), SeenStore("c", root=tmp_path))
    assert stats["published"] == 0 and stats["out_of_window"] == 1


# ---- processing ---------------------------------------------------------------------------------
def test_garbled_text_detection():
    assert process.looks_garbled("u�aOiQ�6Q�lGlsuja^ �^iMi^g-j�i�siMQj") and process.looks_garbled("   ")
    assert not process.looks_garbled("The accused was arrested on 12.03.2018 for demanding a bribe of Rs.20,000/-.")


def test_process_message_uses_reliable_text_and_proposes_facts():
    msg = make_message(site="dvac", doc_type="press_release", title="t", url="https://u", publisher="DVAC", licence="l",
                       published_date="2026-10-12", text="DVAC press release dated 2026-10-12: Tr.A.Test was arrested on 12.10.2026 for demanding a bribe of Rs.20,000/-.")
    r = process.process_message(msg)
    assert not r["skipped"] and r["passages"] and r["passages"][0]["locator"] == "listing"
    cands = [c for p in r["passages"] for c in p["candidates"]]
    assert any(c["stage"] == "arrest" and c["event_date"] == "2026-10-12" and c["level"] == "reported" for c in cands)
    assert not any(c["level"] == "court_finding" for c in cands)          # a press release can never be a court finding


def test_garbled_pdf_is_archived_but_not_loaded(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from backend.files.pdfwriter import build_pdf
    from backend.files.readers import Doc
    d = Doc("txt")
    d.add("p", "��� " * 40)
    pdf, _ = build_pdf(d)
    sha, rel = RawStore(tmp_path).save(pdf, ".pdf")
    r = process.process_message(make_message(site="x", doc_type="order", title="t", url="u", publisher="p", licence="l", raw_path=rel, sha256=sha), tmp_path)
    assert r["skipped"] and r["notes"]


def test_process_reads_a_clean_pdf(tmp_path):
    pytest.importorskip("pypdf")
    from backend.files.pdfwriter import build_pdf
    from backend.files.readers import Doc
    d = Doc("txt")
    d.add("p", "The accused is acquitted of all charges by this Court on 5 January 2021. The trial court convicted him earlier, but that order is set aside.")
    pdf, _ = build_pdf(d)
    sha, rel = RawStore(tmp_path).save(pdf, ".pdf")
    r = process.process_message(make_message(site="x", doc_type="judgment", title="t", url="u", publisher="p", licence="l", raw_path=rel, sha256=sha), tmp_path)
    assert not r["skipped"] and r["passages"][0]["locator"] == "page 1"
    assert any(c["status"] == "acquitted" and c["level"] == "court_finding" for p in r["passages"] for c in p["candidates"])


def test_discussion_in_a_judgment_is_not_proposed_as_a_court_finding():
    from backend.knowledge import review
    # arguments / recitals: never a court finding; non-operative "quashing" talk is dropped entirely
    talk = "Learned counsel argued that the High Court erred in quashing criminal proceedings against the respondents."
    assert review.propose(talk, "judgment", operative=bool(review.OPERATIVE.search(talk))) == []
    recital = "The trial court convicted the accused on 5 January 2021 under the Prevention of Corruption Act."
    got = review.propose(recital, "judgment", operative=bool(review.OPERATIVE.search(recital)))
    assert got and all(c["level"] == "reported" for c in got)
    for operative_sentence in ("The appeal is hereby dismissed.", "The accused stands acquitted of all charges.", "The conviction is set aside.",
                               "He is sentenced to three years rigorous imprisonment."):
        assert review.OPERATIVE.search(operative_sentence), operative_sentence


# ---- loading into the graph (needs the disposable Neo4j) ----------------------------------------
def test_load_result_is_atomic_idempotent_and_creates_review_candidates(kb):
    msg = make_message(site="dvac", doc_type="press_release", title="Trap case", url="https://dvac.example/1", publisher="DVAC", licence="l",
                       published_date="2026-10-12", text="Tr.A.Test was arrested on 12.10.2026 for demanding a bribe of Rs.20,000/- in Nilgiris District.")
    r = process.process_message(msg)
    assert process.load_result(kb, r) == "loaded" and process.load_result(kb, r) == "duplicate"
    assert kb._read("MATCH (s:Source) RETURN count(s) AS n")[0]["n"] == 1
    assert kb._read("MATCH (c:Candidate {state:'pending'}) RETURN count(c) AS n")[0]["n"] >= 1
    assert kb.audit() == [] and process.load_result(kb, {"skipped": True}) == "skipped"


def test_load_failure_leaves_nothing_behind(kb):
    bad = {"skipped": False, "title": "t", "doc_type": "press_release", "url": "u", "publisher": "p", "published_date": None, "licence": None, "text": "some text",
           "passages": [{"locator": "a", "text": "ok", "candidates": []},
                        {"locator": "b", "text": "bad", "candidates": [{"stage": "arrest", "status": "arrested", "level": "PROVEN", "sentence": "x"}]}]}
    with pytest.raises(ValueError):
        process.load_result(kb, bad)
    assert kb._read("MATCH (n) WHERE n:Source OR n:Passage OR n:Candidate RETURN count(n) AS n")[0]["n"] == 0


def test_same_sentence_twice_in_a_passage_is_one_proposal_not_a_failure(kb):
    """Found on real data: a judgment repeated one operative sentence inside one passage and the whole document failed."""
    text = "The accused is hereby acquitted of all the charges framed against him. " * 3
    msg = make_message(site="x", doc_type="judgment", title="Repeat", url="https://u/repeat", publisher="p", licence="l", text=text)
    r = process.process_message(msg)
    assert process.load_result(kb, r) == "loaded"
    assert kb._read("MATCH (c:Candidate) RETURN count(c) AS n")[0]["n"] == 2          # 6 raw proposals -> 2 distinct ones (acquitted; charges framed)


def test_a_poisonous_document_is_set_aside_and_does_not_block_the_next(kb):
    good = process.process_message(make_message(site="x", doc_type="press_release", title="Good", url="https://u/good", publisher="p", licence="l",
                                                text="DVAC press release dated 2026-10-12: An officer was trapped and arrested for demanding a bribe in Salem District."))
    bad = {"skipped": False, "title": "Bad", "doc_type": "not-a-real-type", "url": "https://u/bad", "publisher": "p", "published_date": None,
           "licence": None, "text": "bad text", "passages": [{"locator": "a", "text": "bad text", "candidates": []}]}
    failed = []
    counts = process.load_many(kb, [bad, good], on_failure=lambda r, e: failed.append((r["url"], type(e).__name__)))
    assert counts["loaded"] == 1 and counts["failed"] == 1 and failed == [("https://u/bad", "ValueError")]
    assert kb._read("MATCH (s:Source) RETURN s.title AS t")[0]["t"] == "Good"


def test_a_store_outage_is_raised_so_the_batch_is_retried_not_skipped():
    from neo4j.exceptions import ServiceUnavailable

    class Down:
        def add_document(self, *a, **k):
            raise ServiceUnavailable("neo4j down")
    doc = {"skipped": False, "title": "t", "doc_type": "order", "url": "u", "publisher": "p", "published_date": None, "licence": None,
           "text": "some text", "passages": []}
    with pytest.raises(ServiceUnavailable):
        process.load_many(Down(), [doc], on_failure=lambda r, e: None)
    assert process.is_transient(ConnectionError()) and not process.is_transient(ValueError())
