"""Court judgments from two open datasets on AWS (CC-BY-4.0, by Dattam Labs, built from the eCourts / Supreme Court records):

  * Indian Supreme Court Judgments   s3://indian-supreme-court-judgments   1950-2025   (metadata/parquet + data/pdf)
  * Indian High Court Judgments      s3://indian-high-court-judgments      court=33_10 is the Madras High Court

No scraping, no captcha: the buckets are public. The datasets only carry party names / title in metadata, so relevance is
decided in two steps:
  1. metadata filter on party names (agencies such as Vigilance and Anti-Corruption, CBI, ED; for the Supreme Court also
     Tamil Nadu / Madras and a few named cases) to choose which PDFs are worth downloading, then
  2. a text check on the downloaded PDF: it must contain corruption wording (Prevention of Corruption Act, bribe,
     disproportionate assets, PMLA ...) and, for the Supreme Court, a Tamil Nadu term. Only then is it archived and published.
Judgments the metadata filter cannot see (an accused named only as "X vs State rep. by Inspector of Police") are NOT found by
this step; they need a manual list (add terms with EXTRA_TITLE_TERMS or drop the PDFs into backend/data/kb_inbox/judgment/).

Attribution (CC-BY-4.0) travels with every record in its `licence` field.
"""
import re
from datetime import date, datetime
from io import BytesIO

from backend.pipeline.common import TN_TERMS, in_window, make_message

SC_BUCKET = "indian-supreme-court-judgments"
HC_BUCKET = "indian-high-court-judgments"
MADRAS = "33_10"
REGION = "ap-south-1"
SC_PUBLISHER = "Supreme Court of India (via the Indian Supreme Court Judgments open dataset, Dattam Labs)"
HC_PUBLISHER = "Madras High Court (via the Indian High Court Judgments open dataset, Dattam Labs)"
LICENCE = "CC-BY-4.0 (Dattam Labs open dataset; original records from the eCourts / Supreme Court of India)"

AGENCY = r"VIGILANCE\s*(AND|&)\s*ANTI|ANTI.?CORRUPTION|\bDVAC\b|CENTRAL BUREAU OF INVESTIGATION|\bCBI\b|ENFORCEMENT DIRECTORATE|DIRECTORATE OF ENFORCEMENT"
# Named cases the generic filter cannot see; extend as needed.
EXTRA_TITLE_TERMS = ["JAYALALITH", "SELVI J", "KARUNANIDHI", "A. RAJA", "ANDIMUTHU RAJA", "SASIKALA", "SUDHAKARAN", "SENTHIL BALAJI"]
SC_TITLE = re.compile(AGENCY + r"|TAMIL ?NADU|MADRAS|CORRUPTION|" + "|".join(re.escape(t) for t in EXTRA_TITLE_TERMS), re.I)
HC_TITLE = re.compile(AGENCY, re.I)
CORRUPTION_TEXT = re.compile(r"Prevention of Corruption Act|disproportionate to (?:his|her|their) known sources|disproportionate assets|"
                             r"illegal gratification|\bbribe|Vigilance and Anti.?Corruption|\bDVAC\b|criminal misconduct|"
                             r"Prevention of Money.?Laundering Act|\bPMLA\b|money.?laundering|\bPC Act\b", re.I)


def title_matches(court, *fields):
    rx = SC_TITLE if court == "sc" else HC_TITLE
    return bool(rx.search(" ".join(f for f in fields if f)))


def text_is_relevant(text, court):
    """The downloaded judgment must show corruption wording (and, for the Supreme Court, a Tamil Nadu link)."""
    if not CORRUPTION_TEXT.search(text):
        return False
    return court != "sc" or bool(TN_TERMS.search(text))


def iso_date(raw):
    """'30-11-2023' or '2024-12-02 00:00:00' -> '2023-11-30' / '2024-12-02' (None if unparseable)."""
    if isinstance(raw, datetime):                       # some parquet years store a real timestamp, not text
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    raw = str(raw or "").strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def pdf_text(data):
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(BytesIO(data)).pages)


class S3Source:
    """Anonymous, read-only access to the public buckets with pyarrow (column-only parquet reads, so the large raw_html
    column is never downloaded)."""

    def __init__(self):
        import pyarrow.fs as pafs
        self.fs = pafs.S3FileSystem(anonymous=True, region=REGION)

    def table(self, bucket, key, columns):
        import pyarrow.parquet as pq
        return pq.read_table(f"{bucket}/{key}", columns=columns, filesystem=self.fs).to_pylist()

    def metadata(self, path, wanted):
        """The columns of a metadata table that exist in it (the schema differs a little between years), as a list of dicts."""
        import pyarrow.fs as pafs
        import pyarrow.parquet as pq
        try:
            names = set(pq.read_schema(path, filesystem=self.fs).names)
            return pq.read_table(path, columns=[c for c in wanted if c in names], filesystem=self.fs).to_pylist()
        except OSError:
            if self.fs.get_file_info(path).type == pafs.FileType.NotFound:       # this year/court has no metadata file: normal, nothing to read
                return []
            raise                                                                # anything else (network, throttling) is NOT "no data": the caller must retry later

    def ping(self):
        """Raises when the bucket cannot be reached at all (no internet, DNS, AWS down)."""
        import pyarrow.fs as pafs
        if self.fs.get_file_info(f"{SC_BUCKET}/metadata/parquet/year=2019/metadata.parquet").type == pafs.FileType.Unknown:
            raise OSError("the court data bucket did not answer")

    def children(self, bucket, prefix):
        import pyarrow.fs as pafs
        infos = self.fs.get_file_info(pafs.FileSelector(f"{bucket}/{prefix}", recursive=False, allow_not_found=True))
        return sorted(i.base_name for i in infos)

    def read(self, bucket, key):
        with self.fs.open_input_stream(f"{bucket}/{key}") as f:
            return f.read()


def scan_units(source, court, year):
    """The independently scannable parts of one year: one per Supreme Court year, one per Madras High Court bench."""
    if court == "sc":
        return [("sc", year, None)]
    return [("hc", year, b.replace("bench=", "")) for b in source.children(HC_BUCKET, f"metadata/parquet/year={year}/court={MADRAS}/")]


def candidates(source, unit):
    """Yield {key, bucket, title, date, judge, outcome, ref, bench} for the metadata rows of one scan unit that pass the
    party-name filter. An unreadable or missing metadata file yields nothing."""
    court, year, bench = unit
    try:
        if court == "sc":
            cols = ["title", "petitioner", "respondent", "decision_date", "path", "case_id", "judge", "disposal_nature"]
            rows = source.table(SC_BUCKET, f"metadata/parquet/year={year}/metadata.parquet", cols)
        else:
            cols = ["title", "pdf_link", "decision_date", "disposal_nature", "judge", "cnr"]
            rows = source.table(HC_BUCKET, f"metadata/parquet/year={year}/court={MADRAS}/bench={bench}/metadata.parquet", cols)
    except Exception:
        return
    for r in rows:
        if court == "sc":
            if title_matches("sc", r.get("title"), r.get("petitioner"), r.get("respondent")):
                yield {"bucket": SC_BUCKET, "key": f"data/pdf/year={year}/english/{r['path']}_EN.pdf", "title": r["title"],
                       "date": iso_date(r.get("decision_date")), "judge": r.get("judge"), "outcome": r.get("disposal_nature"),
                       "ref": r.get("case_id"), "bench": None}
        elif r.get("pdf_link") and title_matches("hc", r.get("title")):
            base = r["pdf_link"].rsplit("/", 1)[-1]
            yield {"bucket": HC_BUCKET, "key": f"data/pdf/year={year}/court={MADRAS}/bench={bench}/{base}", "title": r["title"],
                   "date": iso_date(r.get("decision_date")), "judge": r.get("judge"), "outcome": r.get("disposal_nature"),
                   "ref": r.get("cnr"), "bench": bench}


def collect(source, store, sink, seen, courts=("sc", "hc"), min_year=1947, max_year=2026, limit=None, max_downloads=None, current_year=2026):
    """Publish every new, relevant judgment. Returns counts. `seen` remembers downloaded candidates (relevant or not) and
    finished scan units, so each run only does new work; the current year is always rescanned (the data keeps updating)."""
    stats = {"candidates": 0, "downloaded": 0, "published": 0, "not_relevant": 0, "no_text": 0, "missing_pdf": 0, "skipped_seen": 0}
    first = {"sc": max(min_year, 1950), "hc": max(min_year, 2000)}

    def done():
        return (limit is not None and stats["published"] >= limit) or (max_downloads is not None and stats["downloaded"] >= max_downloads)

    for court in courts:
        for year in range(first[court], min(max_year, current_year) + 1):
            for unit in scan_units(source, court, year):
                scan_key = "scan:" + ":".join(str(x) for x in unit if x is not None)
                if done():
                    return stats
                if scan_key in seen and year < current_year:
                    continue
                finished = True
                for c in candidates(source, unit):
                    stats["candidates"] += 1
                    url = f"https://{c['bucket']}.s3.amazonaws.com/{c['key']}"
                    if url in seen:
                        stats["skipped_seen"] += 1
                        continue
                    if done():
                        finished = False
                        break
                    if not in_window(c["date"] or year, min_year, max_year):
                        seen.add(url)
                        continue
                    try:
                        data = source.read(c["bucket"], c["key"])
                    except Exception:
                        stats["missing_pdf"] += 1
                        seen.add(url)
                        continue
                    stats["downloaded"] += 1
                    try:
                        text = pdf_text(data)
                    except Exception:
                        text = ""
                    seen.add(url)
                    if len(text.strip()) < 200:
                        stats["no_text"] += 1
                    elif not text_is_relevant(text, court):
                        stats["not_relevant"] += 1
                    else:
                        sha, raw_path = store.save(data, ".pdf")
                        is_sc = court == "sc"
                        sink.send(make_message(site="sc_opendata" if is_sc else "hc_opendata", doc_type="judgment" if is_sc else "order",
                                               title=c["title"][:240], url=url, publisher=SC_PUBLISHER if is_sc else HC_PUBLISHER,
                                               licence=LICENCE, published_date=c["date"], raw_path=raw_path, sha256=sha,
                                               content_type="application/pdf",
                                               extra={"judge": c["judge"], "outcome": c["outcome"], "ref": c["ref"], "bench": c["bench"]}))
                        sink.flush()
                        stats["published"] += 1
                    seen.save()
                if finished:
                    seen.add(scan_key)
                seen.save()
    return stats
