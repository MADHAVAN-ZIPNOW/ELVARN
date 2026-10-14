"""Court open-dataset collector, offline: an in-memory fake of the S3 buckets. All cases and names are FICTIONAL."""
import pytest

pytest.importorskip("pypdf")

from backend.files.pdfwriter import build_pdf  # noqa: E402
from backend.files.readers import Doc  # noqa: E402
from backend.pipeline.collectors import court_opendata as co  # noqa: E402
from backend.pipeline.common import ListSink, RawStore, SeenStore  # noqa: E402

CORRUPT_TN = ("In the Madras High Court. The State represented by the Deputy Superintendent of Police, Vigilance and Anti-Corruption, "
              "Chennai registered a case against the accused, a public servant, under Section 13(1)(e) of the Prevention of Corruption Act "
              "for possessing assets disproportionate to his known sources of income in Tamil Nadu. " * 3)
CORRUPT_OTHER = ("The CBI registered a case in Gujarat under the Prevention of Corruption Act against a customs officer for demanding a bribe at the port. " * 4)
NOT_CORRUPT = ("This appeal concerns the interpretation of a service rule on seniority in the Tamil Nadu Educational Subordinate Service. " * 5)


def pdf(text):
    d = Doc("txt")
    d.add("p", text)
    return build_pdf(d)[0]


class FakeSource:
    def __init__(self, tables=None, files=None, benches=None):
        self.tables, self.files, self.benches = tables or {}, files or {}, benches or {}
        self.reads = []

    def table(self, bucket, key, columns):
        if (bucket, key) not in self.tables:
            raise FileNotFoundError(key)
        return self.tables[(bucket, key)]

    def children(self, bucket, prefix):
        return self.benches.get(prefix, [])

    def read(self, bucket, key):
        self.reads.append(key)
        if (bucket, key) not in self.files:
            raise FileNotFoundError(key)
        return self.files[(bucket, key)]


def sc_row(title, path, date="30-11-2003", **kw):
    return {"title": title, "petitioner": kw.get("petitioner", ""), "respondent": kw.get("respondent", ""), "decision_date": date, "path": path,
            "case_id": f"CASE {path}", "judge": "J. TEST", "disposal_nature": "Appeal allowed"}


def sc_source():
    rows = [sc_row("STATE OF TAMIL NADU versus A. TEST", "p1"),
            sc_row("STATE OF TAMIL NADU versus SERVICE CASE", "p2"),
            sc_row("CBI versus B. OTHER", "p3"),
            sc_row("UNRELATED versus PARTY", "p4"),
            sc_row("STATE OF TAMIL NADU versus MISSING", "p5"),
            sc_row("STATE OF TAMIL NADU versus OLD", "p6", date="01-01-1930")]
    files = {(co.SC_BUCKET, f"data/pdf/year=2003/english/{n}_EN.pdf"): pdf(t)
             for n, t in (("p1", CORRUPT_TN), ("p2", NOT_CORRUPT), ("p3", CORRUPT_OTHER), ("p4", CORRUPT_TN))}
    return FakeSource(tables={(co.SC_BUCKET, "metadata/parquet/year=2003/metadata.parquet"): rows}, files=files)


# ---- filters ------------------------------------------------------------------------------------
def test_filters_and_dates():
    assert co.title_matches("sc", "STATE OF TAMIL NADU versus X") and co.title_matches("sc", "State of Karnataka versus Selvi J Jayalalitha")
    assert not co.title_matches("sc", "RAM versus SHYAM")
    assert co.title_matches("hc", "Crl.OP/1/2020 of X Vs State rep by Inspector, Vigilance and Anti Corruption") and not co.title_matches("hc", "WP/1/2020 of X Vs Collector")
    assert co.text_is_relevant(CORRUPT_TN, "sc") and not co.text_is_relevant(CORRUPT_OTHER, "sc") and co.text_is_relevant(CORRUPT_OTHER, "hc")
    assert not co.text_is_relevant(NOT_CORRUPT, "hc")
    assert co.iso_date("30-11-2023") == "2023-11-30" and co.iso_date("2024-12-02 00:00:00") == "2024-12-02" and co.iso_date("bad") is None
    import datetime
    assert co.iso_date(datetime.datetime(2015, 5, 5, 0, 0)) == "2015-05-05" and co.iso_date(datetime.date(2015, 5, 6)) == "2015-05-06"   # real timestamps
    assert co.iso_date(None) is None


# ---- Supreme Court ------------------------------------------------------------------------------
def test_supreme_court_publishes_only_verified_tamil_nadu_corruption_judgments(tmp_path):
    src, sink, store = sc_source(), ListSink(), RawStore(tmp_path / "raw")
    stats = co.collect(src, store, sink, SeenStore("c", root=tmp_path), courts=("sc",), min_year=2003, max_year=2003, current_year=2026)
    assert stats["published"] == 1 and stats["not_relevant"] == 2          # p2 not corruption, p3 corruption but not Tamil Nadu
    assert stats["missing_pdf"] == 1 and stats["candidates"] == 5            # p4 never matched the party-name filter; p5 has no PDF
    m = sink.messages[0]
    assert m["doc_type"] == "judgment" and m["title"] == "STATE OF TAMIL NADU versus A. TEST" and m["published_date"] == "2003-11-30"
    assert m["url"] == f"https://{co.SC_BUCKET}.s3.amazonaws.com/data/pdf/year=2003/english/p1_EN.pdf"
    assert "CC-BY-4.0" in m["licence"] and (tmp_path / "raw" / m["raw_path"]).read_bytes().startswith(b"%PDF")
    assert "p4_EN.pdf" not in " ".join(src.reads)                          # filtered out by metadata: never downloaded


def test_second_run_downloads_nothing_new(tmp_path):
    src, store = sc_source(), RawStore(tmp_path / "raw")
    co.collect(src, store, ListSink(), SeenStore("c", root=tmp_path), courts=("sc",), min_year=2003, max_year=2003, current_year=2026)
    first_reads = len(src.reads)
    stats = co.collect(src, store, ListSink(), SeenStore("c", root=tmp_path), courts=("sc",), min_year=2003, max_year=2003, current_year=2026)
    assert stats["published"] == 0 and len(src.reads) == first_reads        # the finished year is skipped entirely


def test_max_downloads_stops_and_resumes_without_losing_work(tmp_path):
    src, store = sc_source(), RawStore(tmp_path / "raw")
    sink = ListSink()
    first = co.collect(src, store, sink, SeenStore("c", root=tmp_path), courts=("sc",), min_year=2003, max_year=2003, max_downloads=1, current_year=2026)
    assert first["downloaded"] == 1
    rest = co.collect(src, store, sink, SeenStore("c", root=tmp_path), courts=("sc",), min_year=2003, max_year=2003, current_year=2026)
    assert rest["downloaded"] == 2 and len(sink.messages) == 1               # p2 and p3 remained (p1 was done in the first run)
    pdfs = [k for k in src.reads if not k.endswith("p5_EN.pdf")]
    assert len(pdfs) == len(set(pdfs)) == 3                                  # no judgment was downloaded twice


def test_out_of_window_decisions_are_not_published(tmp_path):
    src = sc_source()
    co.collect(src, RawStore(tmp_path / "raw"), ListSink(), SeenStore("c", root=tmp_path), courts=("sc",), min_year=2003, max_year=2003, current_year=2026)
    assert not any("p6_EN" in k for k in src.reads)


# ---- Madras High Court --------------------------------------------------------------------------
def test_madras_high_court_benches_and_key_mapping(tmp_path):
    prefix = f"metadata/parquet/year=2015/court={co.MADRAS}/"
    rows = [{"title": "Crl.OP/5/2015 of A. TEST Vs The State rep by Vigilance and Anti Corruption, Chennai", "pdf_link": "court/cnrorders/hc_cis_mas/orders/HCMA01_1_2015-05-05.pdf",
             "decision_date": "2015-05-05 00:00:00", "disposal_nature": "DISMISSED", "judge": "JUSTICE TEST", "cnr": "HCMA01"},
            {"title": "WP/9/2015 of B Vs Collector", "pdf_link": "court/x/HCMA02_1_2015-05-06.pdf", "decision_date": "2015-05-06 00:00:00", "disposal_nature": "X", "judge": "J", "cnr": "HCMA02"}]
    src = FakeSource(benches={prefix: ["bench=hc_cis_mas", "bench=mdubench"]},
                     tables={(co.HC_BUCKET, prefix + "bench=hc_cis_mas/metadata.parquet"): rows},     # mdubench has no metadata file: skipped, not fatal
                     files={(co.HC_BUCKET, f"data/pdf/year=2015/court={co.MADRAS}/bench=hc_cis_mas/HCMA01_1_2015-05-05.pdf"): pdf(CORRUPT_TN)})
    sink = ListSink()
    stats = co.collect(src, RawStore(tmp_path / "raw"), sink, SeenStore("c", root=tmp_path), courts=("hc",), min_year=2015, max_year=2015, current_year=2026)
    assert stats["published"] == 1 and stats["candidates"] == 1
    m = sink.messages[0]
    assert m["doc_type"] == "order" and m["site"] == "hc_opendata" and m["extra"]["bench"] == "hc_cis_mas" and m["extra"]["ref"] == "HCMA01"
    assert co.collect(src, RawStore(tmp_path / "raw"), ListSink(), SeenStore("c", root=tmp_path), courts=("hc",), min_year=2015, max_year=2015, current_year=2026)["published"] == 0


def test_published_court_messages_can_back_a_court_finding(tmp_path):
    from backend.pipeline import process
    src, sink, store = sc_source(), ListSink(), RawStore(tmp_path / "raw")
    co.collect(src, store, sink, SeenStore("c", root=tmp_path), courts=("sc",), min_year=2003, max_year=2003, current_year=2026)
    r = process.process_message(sink.messages[0], tmp_path / "raw")
    assert not r["skipped"] and r["doc_type"] == "judgment" and r["passages"]
