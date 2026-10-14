"""Airflow DAG: every night collect Tamil Nadu corruption documents (1947-2026), load the three stores, refresh the training data.

    collect_dvac / collect_cbi / collect_supreme_court / collect_madras_high_court   (download + publish to Kafka)
    load_wikipedia                                                                    (straight into Neo4j + Chroma)
        -> wait_for_spark       Spark turns the Kafka messages into Neo4j + Chroma data; wait until it has caught up
        -> check_stores         repair any gap between Neo4j and Chroma, then fail the run if they still disagree
        -> audit_graph          every stored fact must still be cited (court findings only from judgments/orders)
        -> export_training      write backend/data/raw_v2/kb_corpus.txt for the model's training

Only NEW documents are collected: each collector remembers what it already fetched, and every store identifies a document by the
SHA-256 of its text, so running twice never creates duplicates. If the machine was off at 02:00, Airflow runs the missed
night as soon as it is back (catchup=False runs the latest missed interval once).

Retraining the model is a separate, heavier step (backend/training/daily_update.py, run on the host, e.g. weekly): it needs torch.
"""
import time
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

RAG_PY = "cd /opt/elvarn && /home/airflow/venv-rag/bin/python -m"        # the virtualenv that has the Chroma client


def collect(site, **kw):
    from backend.pipeline.run_collectors import run
    stats = run(site, **kw)
    print(stats)
    return stats


def wait_for_spark(poll_seconds=30, stable_polls=4, timeout_minutes=60):
    """Spark has no 'done' signal for a stream, so wait until the number of documents in Neo4j stops changing for a while."""
    from backend.knowledge.kb import KnowledgeBase
    kb = KnowledgeBase()
    try:
        last, stable, deadline = None, 0, time.time() + timeout_minutes * 60
        while time.time() < deadline:
            n = kb._read("MATCH (s:Source) RETURN count(s) AS n")[0]["n"]
            stable = stable + 1 if n == last else 0
            if stable >= stable_polls:
                print(f"loading has settled at {n} documents")
                return n
            last = n
            time.sleep(poll_seconds)
        raise TimeoutError("Spark was still loading documents after the time limit")
    finally:
        kb.close()


def audit_graph():
    """Fail the run (visible in the Airflow UI) if any stored fact breaks the citation rules."""
    from backend.knowledge.kb import KnowledgeBase
    kb = KnowledgeBase()
    try:
        problems = kb.audit()
    finally:
        kb.close()
    if problems:
        raise RuntimeError("knowledge graph audit failed: " + "; ".join(problems))


def export_training():
    from backend.knowledge.export_training import export
    from backend.knowledge.kb import KnowledgeBase
    kb = KnowledgeBase()
    try:
        info = export(kb)
    finally:
        kb.close()
    print(info)
    return info


with DAG(
    dag_id="elvarn_tn_corruption",
    description="Collect Tamil Nadu corruption sources into Kafka; load Neo4j + Chroma; audit; refresh training data",
    schedule="0 2 * * *",
    start_date=pendulum.datetime(2026, 10, 1, tz="Asia/Kolkata"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
    tags=["elvarn", "tamil-nadu", "corruption"],
) as dag:
    dvac = PythonOperator(task_id="collect_dvac", python_callable=collect, op_kwargs={"site": "dvac"})
    cbi = PythonOperator(task_id="collect_cbi", python_callable=collect, op_kwargs={"site": "cbi", "max_fetch": 400})
    sc = PythonOperator(task_id="collect_supreme_court", python_callable=collect, op_kwargs={"site": "sc", "max_fetch": 300})
    hc = PythonOperator(task_id="collect_madras_high_court", python_callable=collect, op_kwargs={"site": "hc", "max_fetch": 300})
    wiki = BashOperator(task_id="load_wikipedia", bash_command=f"{RAG_PY} backend.knowledge.ingest_wikipedia")
    wait = PythonOperator(task_id="wait_for_spark", python_callable=wait_for_spark, trigger_rule="all_done", retries=0)
    stores = BashOperator(task_id="check_stores", bash_command=f"{RAG_PY} backend.knowledge.reindex --verify")      # fails if Neo4j and Chroma still disagree
    audit = PythonOperator(task_id="audit_graph", python_callable=audit_graph)
    export = PythonOperator(task_id="export_training", python_callable=export_training)
    [dvac, cbi, sc, hc, wiki] >> wait >> stores >> audit >> export
