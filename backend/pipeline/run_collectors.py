"""Run one collector: download new documents and publish them to Kafka.

    python -m backend.pipeline.run_collectors dvac [--limit 20] [--no-pdf]
    python -m backend.pipeline.run_collectors cbi  [--limit 20] [--max-fetch 200]
    python -m backend.pipeline.run_collectors sc   [--max-fetch 100]     # Supreme Court judgments (open dataset)
    python -m backend.pipeline.run_collectors hc   [--max-fetch 100]     # Madras High Court (open dataset)

Env: KAFKA_BOOTSTRAP (default localhost:9094), KAFKA_TOPIC, COLLECT_DELAY (seconds between requests, default 3).
Airflow calls run() from the DAG in dags/elvarn_tn_corruption.py.
"""
import argparse
import json
import os

from backend.pipeline.collectors import cbi, court_opendata, dvac
from backend.pipeline.common import MAX_YEAR, MIN_YEAR, Fetcher, RawStore, SeenStore
from backend.pipeline.kafka_io import KafkaSink


def run(site, limit=None, download_pdfs=True, max_fetch=None, sink=None, fetcher=None):
    fetcher = fetcher or Fetcher(delay=float(os.environ.get("COLLECT_DELAY", "3")))
    sink = sink or KafkaSink()
    store, seen = RawStore(), SeenStore(site)
    if site == "dvac":
        return dvac.collect(fetcher, store, sink, seen, limit=limit, min_year=MIN_YEAR, max_year=MAX_YEAR, download_pdfs=download_pdfs)
    if site == "cbi":
        return cbi.collect(fetcher, store, sink, seen, limit=limit, min_year=MIN_YEAR, max_year=MAX_YEAR, max_fetch=max_fetch)
    if site in ("sc", "hc"):
        return court_opendata.collect(court_opendata.S3Source(), store, sink, seen, courts=(site,), min_year=MIN_YEAR, max_year=MAX_YEAR,
                                      limit=limit, max_downloads=max_fetch)
    raise ValueError(f"unknown collector {site!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", choices=["dvac", "cbi", "sc", "hc"])
    ap.add_argument("--limit", type=int, default=None, help="publish at most this many documents")
    ap.add_argument("--max-fetch", type=int, default=None, help="cbi: detail pages to fetch; sc/hc: judgments to download")
    ap.add_argument("--no-pdf", action="store_true", help="dvac only: do not download the PDFs")
    a = ap.parse_args()
    print(json.dumps(run(a.site, a.limit, not a.no_pdf, a.max_fetch)))


if __name__ == "__main__":
    main()
