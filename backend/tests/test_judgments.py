"""The all-India judgments collector: case names, tags, the order of work, the disk budget, duplicates, and what goes to Kafka (no network, no store)."""
import io
import json

from backend.pipeline import judgments, law_tags
from backend.pipeline.collectors import court_opendata as co
from backend.pipeline.person_sources import RawArchive
from backend.pipeline.registry import Registry

TEXT = ("IN THE HIGH COURT. The petitioner, a former MLA, filed an election petition under the Representation of the People Act, 1951 and Article 226 of "
        "the Constitution of India. " * 12)


def make_pdf(text=TEXT):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    w = PdfWriter()
    page = w.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})})
    lines = [text[i:i + 90] for i in range(0, min(len(text), 2400), 90)]
    stream = DecodedStreamObject()
    stream.set_data(("BT /F1 9 Tf 20 770 Td 11 TL " + " ".join("(" + l.replace("(", "").replace(")", "") + ") Tj T*" for l in lines) + " ET").encode())
    page[NameObject("/Contents")] = w._add_object(stream)
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


class FakeSource:
    """Stands in for the AWS buckets: one Supreme Court year and one High Court bench."""

    def __init__(self, files, sc_rows, hc_rows):
        self.files, self.sc_rows, self.hc_rows = files, sc_rows, hc_rows
        self.reads = []

    def children(self, bucket, prefix):
        if prefix.endswith("metadata/parquet/year=2020/"):
            return ["court=33_10"]
        if prefix.endswith("court=33_10/"):
            return ["bench=mdubench"]
        raise OSError("not listed")

    def metadata(self, path, wanted):
        if co.SC_BUCKET in path:
            return self.sc_rows if "year=2020" in path else []
        return self.hc_rows if "year=2020" in path else []

    def read(self, bucket, key):
        self.reads.append(key)
        if key not in self.files:
            raise OSError("missing")
        return self.files[key]


class Sink:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)

    def flush(self):
        pass


def collector(tmp_path, source, **kw):
    args = dict(registry=Registry(tmp_path / "r.sqlite"), archive=RawArchive(tmp_path / "raw", today=lambda: "2026-10-16"), sink=Sink(), s3=source,
                plan_path=tmp_path / "plan.sqlite", say=lambda m: None, from_year=2020, to_year=2020, current_year=2020, workers=2)
    args.update(kw)
    return judgments.JudgmentCollector(**args)


def source():
    pdf_a, pdf_b = make_pdf(TEXT), make_pdf(TEXT.replace("MLA", "MP"))
    sc = [{"title": "State of Tamil Nadu vs J. Jayalalithaa on 4 May, 2020", "path": "p1", "case_id": "CA 1/2020", "decision_date": "04-05-2020", "judge": "A B", "disposal_nature": "Allowed"}]
    hc = [{"title": "W.P.(MD)/55/2020 of Amit Kumar Vs The State Of Bihar", "pdf_link": "x/y/one.pdf", "decision_date": "2020-03-02 00:00:00", "judge": "C D",
           "disposal_nature": "Dismissed", "cnr": "HCMA01-1", "court": "Madras High Court"},
          {"title": "CR. MISC./7/2020 of Rani Vs State", "pdf_link": "x/y/two.pdf", "decision_date": "2020-03-01", "cnr": "HCMA01-2", "court": "Madras High Court"},
          {"title": "scanned", "pdf_link": "x/y/scan.pdf", "decision_date": "2020-02-01", "court": "Madras High Court"}]
    files = {"data/pdf/year=2020/english/p1_EN.pdf": pdf_a, "data/pdf/year=2020/court=33_10/bench=mdubench/one.pdf": pdf_b,
             "data/pdf/year=2020/court=33_10/bench=mdubench/two.pdf": make_pdf("The appellant and the respondent are neighbours who dispute the boundary of a plot of land and the tenant who lives there. " * 12),
             "data/pdf/year=2020/court=33_10/bench=mdubench/scan.pdf": make_pdf("x")}
    return FakeSource(files, sc, hc)


def test_case_names_are_split_and_cleaned():
    hc = law_tags.case_of_hc("CR. MISC./5208/2019 of Amit Kumar Vs The State Of Bihar")
    assert hc == {"case_name": "Amit Kumar vs The State Of Bihar", "case_type": "CR. MISC.", "case_number": "5208/2019"}
    sc = law_tags.case_of_sc({"title": "State of Tamil Nadu vs J. Jayalalithaa on 4 May, 2017", "case_id": "CA 1/2017"})
    assert sc["case_name"] == "State of Tamil Nadu vs J. Jayalalithaa" and sc["case_number"] == "CA 1/2017"
    assert law_tags.case_of_sc({"title": "", "petitioner": "A", "respondent": "B"})["case_name"] == "A vs B"
    assert law_tags.full_title(hc, "Patna High Court", "2019-04-01") == "Amit Kumar vs The State Of Bihar, CR. MISC. 5208/2019, Patna High Court, 2019-04-01"


def test_subjects_and_statutes_are_read_from_the_text():
    text = "Under the Representation of the People Act, 1951 the MLA filed an election petition. The Prevention of Corruption Act, 1988 case was decided; bail refused."
    assert {"election", "corruption", "criminal"} <= set(law_tags.subjects_of(text))
    assert law_tags.statutes_of(text) == ["Representation of the People Act, 1951", "Prevention of Corruption Act, 1988"]
    assert "election" not in law_tags.subjects_of("a plain land dispute between two neighbours")


def test_a_run_stores_names_and_announces_every_readable_judgment(tmp_path):
    c = collector(tmp_path, source())
    assert c.run() == "done"
    sink = c.sink
    titles = sorted(m["title"] for m in sink.sent)
    assert len(sink.sent) == 3 and c.counts["no_text"] == 1                     # the scanned one is rejected, not stored
    assert "State of Tamil Nadu vs J. Jayalalithaa, No. CA 1/2020, Supreme Court of India, 2020-05-04" in titles
    hc = next(m for m in sink.sent if m["case_number"] == "55/2020")
    assert hc["case_name"] == "Amit Kumar vs The State Of Bihar" and hc["case_type"] == "W.P.(MD)" and hc["court"] == "Madras High Court" and hc["year"] == 2020
    assert hc["doc_type"] == "judgment" and hc["topic"] == "judgment" and hc["event_type"] == "document_ingested" and hc["ref"] == "HCMA01-1"
    assert "election" in hc["subjects"] and "Representation of the People Act, 1951" in hc["statutes"]
    folder = tmp_path / "raw" / "courts" / "Madras-High-Court" / "2020"
    assert any(p.name.startswith("Amit-Kumar-vs-The-State-Of-Bihar__") for p in folder.iterdir())      # the file name is the case name
    assert json.dumps(sink.sent)                                                       # every message is plain JSON


def test_second_run_reads_nothing_again_and_dedupes(tmp_path):
    src = source()
    collector(tmp_path, src).run()
    first = len(src.reads)
    again = collector(tmp_path, src)
    again.run()
    assert len(src.reads) == first and again.counts["stored"] == 0                  # stored, scanned and rejected files are all remembered


def test_topic_filter_keeps_only_matching_judgments(tmp_path):
    c = collector(tmp_path, source(), topics=["election"])
    c.run()
    assert len(c.sink.sent) == 2 and c.counts["off_topic"] == 1                       # 'two.pdf' is about land, not elections


def test_disk_budget_stops_the_run_and_a_bigger_budget_carries_on(tmp_path):
    src = source()
    small = collector(tmp_path, src, max_bytes=len(make_pdf()) + 10)
    assert small.run() == "full" and len(small.sink.sent) == 1
    big = collector(tmp_path, src)
    assert big.run() == "done" and len(small.sink.sent) + len(big.sink.sent) == 3


def test_supreme_court_comes_first_and_high_court_sweeps_are_limited(tmp_path):
    c = collector(tmp_path, source(), quotas=(1, None))
    c.plan()
    assert [u["kind"] for u in c.units(0)] == ["sc", "hc"]
    c.run()
    assert len(c.sink.sent) == 3                                                     # sweep 0 took the newest HC judgment, sweep 1 the rest


def test_stop_request_ends_the_run_cleanly(tmp_path):
    c = collector(tmp_path, source(), should_stop=lambda: True)
    assert c.run() == "stopped" and c.sink.sent == []


def test_case_lifecycle_stage_and_petition_kind():
    assert law_tags.stage_of("sc", "", "Dismissed") == "Final judgment" and law_tags.stage_of("sc", "", None) == "Supreme Court"
    assert law_tags.stage_of("hc", "") == "High Court"
    assert law_tags.stage_of("cbi", "CBI registered a case against the former minister") == "FIR"
    assert law_tags.stage_of("ed", "The Directorate filed a chargesheet before the Special Court") == "Chargesheet"
    assert law_tags.stage_of("dvac", "DVAC raids the house of the former minister") == "Investigation"
    assert law_tags.stage_rank("FIR") < law_tags.stage_rank("Chargesheet") < law_tags.stage_rank("High Court") < law_tags.stage_rank("Supreme Court") < law_tags.stage_rank("Final judgment")
    assert {"writ petition", "bail petition", "election petition"} <= set(law_tags.petitions_of("W.P.(MD) No. 5 of 2020: an election petition and a bail application"))


def test_run_plan_judgments_then_politicians():
    from backend.pipeline import remote_ops
    assert judgments.should_chain("done", "people") and judgments.should_chain("full", "people")
    assert not judgments.should_chain("stopped", "people") and not judgments.should_chain("error", "people") and not judgments.should_chain("done", None)
    words, message = remote_ops.judgments_args({"mode": "judgments", "then": "people", "max_gb": 50})
    assert words[-2:] == ["--then", "people"] and "50" in words and message.startswith("Step 1 of 2")
    words, message = remote_ops.judgments_args({"mode": "judgments"})
    assert "--then" not in words and not message.startswith("Step 1")


def test_start_settings_are_checked_before_they_reach_the_command_line():
    from backend.pipeline import remote_ops
    words, _ = remote_ops.judgments_args({"scope": "; rm -rf /", "max_gb": 99999, "from_year": "abc", "topics": ["election", "$(evil)"], "then": "x; y"})
    assert words[words.index("--scope") + 1] == "all" and words[words.index("--max-gb") + 1] == "500" and "--then" not in words
    assert words[words.index("--topics") + 1] == "election"
