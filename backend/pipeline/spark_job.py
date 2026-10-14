"""Spark Structured Streaming job: Kafka topic -> text extraction + chunking + proposed facts -> Neo4j.

The heavy part (reading the archived raw files, PDF/HTML text extraction, chunking, fact proposal) runs on the Spark
executors via process.process_message. The results are small, so the driver stores them one document at a time in Neo4j (one transaction, KnowledgeBase.add_document).
Chroma is NOT written here (the Spark image's Python is too old for Chroma's client): the nightly Airflow task check_stores
(backend/knowledge/reindex.py) copies every new Neo4j passage into Chroma with the same ids. A document that was already loaded is recognised by the SHA-256 of its text and skipped,
so a retried batch is harmless. If Neo4j is down the batch raises, the Kafka offsets are NOT committed to the checkpoint,
and Spark retries the same batch later: nothing is lost.

    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3 backend/pipeline/spark_job.py [--once]

Env: KAFKA_BOOTSTRAP (kafka:9092 in the compose network), KAFKA_TOPIC, NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD,
REDIS_URL (cache: new data clears it), SPARK_CHECKPOINT.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # repo root, so `backend.*` imports work


def _process(value):
    """Runs on executors. A malformed or unreadable message must not kill the stream: it becomes a 'skipped' result."""
    from backend.pipeline.process import process_message
    try:
        return process_message(json.loads(value))
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "url": None, "notes": [f"processing failed: {exc}"]}


DEAD_LETTER = os.environ.get("SPARK_DEAD_LETTER", "/checkpoints/dead_letter.jsonl")


def _dead_letter(result, exc):
    """A document that cannot be stored goes here (one JSON line) instead of blocking the stream. Inspect and re-send it by hand."""
    line = {"url": result.get("url"), "title": result.get("title"), "error": f"{exc.__class__.__name__}: {str(exc)[:300]}"}
    print(f"DEAD LETTER {line}", flush=True)
    try:
        with open(DEAD_LETTER, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        pass


def process_batch(batch_df, batch_id):
    from backend.knowledge.kb import KnowledgeBase
    from backend.pipeline.process import load_many
    from backend.usermemory.store import Memory
    values = batch_df.selectExpr("CAST(value AS STRING) AS value").rdd.map(lambda r: r.value)
    kb, cache = KnowledgeBase(), Memory.from_env()          # the graph, and the cache (new data clears it)
    notes = []
    try:
        def results():
            for result in values.map(_process).toLocalIterator():
                notes.extend(f"{result.get('url')}: {n}" for n in result.get("notes", []))
                yield result
        counts = load_many(kb, results(), None, None, on_failure=_dead_letter)
        if cache is not None and counts.get("loaded"):
            cache.bump_version()               # once per batch (not per document): cached retrievals are cleared when new data is in
    finally:
        kb.close()
    for note in notes:
        print(f"[batch {batch_id}] {note}", flush=True)
    print(f"[batch {batch_id}] {dict(counts)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="process everything currently on the topic, then stop")
    ap.add_argument("--interval", default="30 seconds")
    a = ap.parse_args()
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("elvarn-kb-loader").getOrCreate()
    stream = (spark.readStream.format("kafka")
              .option("kafka.bootstrap.servers", os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092"))
              .option("subscribe", os.environ.get("KAFKA_TOPIC", "kb.raw_documents"))
              .option("startingOffsets", "earliest").option("failOnDataLoss", "false")
              .option("maxOffsetsPerTrigger", 200).load())
    writer = (stream.writeStream.foreachBatch(process_batch)
              .option("checkpointLocation", os.environ.get("SPARK_CHECKPOINT", "/checkpoints/kb")))
    query = (writer.trigger(availableNow=True) if a.once else writer.trigger(processingTime=a.interval)).start()
    query.awaitTermination()


if __name__ == "__main__":
    main()
