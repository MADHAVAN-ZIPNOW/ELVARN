"""Spark Structured Streaming job: Kafka topic -> clean corpus (Parquet) + knowledge graph (Neo4j).

The heavy part (reading the archived raw files, HTML/PDF/JSON text extraction, cleaning, quality gate, passages, entities) runs on
the Spark executors via process.process_message. The results are small, so the driver then (a) loads each document into Neo4j in one
transaction (KnowledgeBase.add_document_graph; messages of the older collectors keep their old path, add_document), (b) writes the new
passages to Parquet with Spark's own writer under corpus/passages/source=<s>/date=<d>/ and (c) appends the batch counters to
corpus/_stats/<date>.jsonl. Chroma is NOT written here (the Spark image's Python is too old for Chroma's client): the nightly check_stores
(backend/knowledge/check_stores.py) embeds every passage with embedded=false, with the same ids.

Idempotent: passage ids already in corpus/_state/passage_index.tsv are not written to Parquet again (the graph load is MERGE-based), an
identical passage text under another id counts as a duplicate and is dropped. Order inside a batch: Neo4j first, then Parquet, then the
index. If Neo4j is down the batch raises, the Kafka offsets are NOT committed to the checkpoint and Spark retries the same batch later.
If the process dies between the Parquet write and the index update, the retry can write a row twice; check_stores removes such rows.

    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3 backend/pipeline/spark_job.py [--once]

Env: KAFKA_BOOTSTRAP (kafka:9092 in the compose network), KAFKA_TOPIC, NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD,
REDIS_URL (cache: new data clears it), SPARK_CHECKPOINT, ELVARN_CORPUS_DIR (default backend/data/corpus).
Runs on the Spark image's Python 3.8: no newer syntax or standard-library calls here or in corpus.py / entities.py / process.py.
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # repo root, so `backend.*` imports work

CORPUS_DIR = Path(os.environ.get("ELVARN_CORPUS_DIR") or Path(__file__).resolve().parent.parent / "data" / "corpus")


def _process(value):
    """Runs on executors. A malformed or unreadable message must not kill the stream: it becomes a 'skipped' result."""
    from backend.pipeline.process import process_message
    try:
        return process_message(json.loads(value))
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "url": None, "failed": True, "notes": [f"processing failed: {exc}"]}


DEAD_LETTER = os.environ.get("SPARK_DEAD_LETTER", "/checkpoints/dead_letter.jsonl")


def _dead_letter(result, exc):
    """A document that cannot be stored goes here (one JSON line) instead of blocking the stream. Inspect and re-send it by hand."""
    line = {"url": result.get("url"), "title": result.get("title"), "doc_id": result.get("doc_id"),
            "error": f"{exc.__class__.__name__}: {str(exc)[:300]}"}
    print(f"DEAD LETTER {line}", flush=True)
    try:
        with open(DEAD_LETTER, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        pass


def handle_results(results, kb, index, write_groups, corpus_dir=CORPUS_DIR, batch_id=None, offsets=None, on_failure=None, cache=None):
    """The whole driver-side logic of one batch, without Spark: `results` are process_message outputs, `write_groups({(source, date): [row, ...]})`
    writes the Parquet rows (Spark's writer in production). Returns the counters that were appended to _stats/<date>.jsonl."""
    from backend.pipeline import corpus
    from backend.pipeline.process import is_transient, load_result
    counts, groups, doc_ids, notes, written = Counter(), {}, [], [], []
    try:
        for result in results:
            notes.extend(f"{result.get('url')}: {n}" for n in result.get("notes", []))
            if result.get("skipped"):
                counts["failed" if result.get("failed") else "skipped"] += 1
                continue
            try:
                if result.get("records") is None:                           # an older collector's message: the old graph path
                    counts["processed" if load_result(kb, result) in ("loaded", "duplicate") else "skipped"] += 1
                    continue
                kept, fresh = [], []
                counts["duplicates_removed"] += result.get("duplicates", 0)
                for rec in result["records"]:
                    kind = index.classify(rec)
                    if kind == "duplicate":
                        counts["duplicates_removed"] += 1
                    else:
                        kept.append(rec)
                        if kind == "new":
                            fresh.append(rec)
                if not kept and not result.get("relations"):
                    counts["skipped"] += 1
                    continue
                load_result(kb, dict(result, records=kept))                 # Neo4j first: idempotent, so a retried batch is harmless
                for rec in fresh:
                    rec["indexed_neo4j"] = True
                    index.stage(rec)
                    groups.setdefault((result["source"], result["date"]), []).append(corpus.row_of(rec))
                written += fresh
                counts["processed"] += 1
                if result.get("doc_id"):
                    doc_ids.append(result["doc_id"])
            except Exception as exc:  # noqa: BLE001
                if is_transient(exc):
                    raise
                counts["failed"] += 1
                if on_failure is not None:
                    on_failure(result, exc)
        if groups:
            write_groups(groups)
        index.commit()
    except BaseException:
        index.rollback()
        raise
    now = datetime.now(timezone.utc)
    line = {"at": now.isoformat(timespec="seconds"), "finished_at": now.isoformat(timespec="seconds"), "batch_id": batch_id,
            "processed": counts["processed"], "skipped": counts["skipped"], "failed": counts["failed"],
            "duplicates_removed": counts["duplicates_removed"], "passages": len(written),
            "clean_tokens": sum(r["token_estimate"] for r in written), "offsets": offsets or {}, "doc_ids": doc_ids}
    corpus.append_stats(corpus_dir, now.date().isoformat(), line)
    if cache is not None and (written or counts["processed"]):
        cache.bump_version()                                                # once per batch: cached retrievals are cleared when new data is in
    for note in notes:
        print(f"[batch {batch_id}] {note}", flush=True)
    print(f"[batch {batch_id}] {line}", flush=True)
    return line


def make_parquet_writer(spark, corpus_dir=CORPUS_DIR):
    """write_groups for handle_results: one Spark DataFrame per (source, date), appended to its partition folder. The folder name carries
    source= and date= (contract 4.1) and `source` also stays a column inside the file."""
    from pyspark.sql.types import BooleanType, DoubleType, IntegerType, StringType, StructField, StructType
    from backend.pipeline import corpus
    types = {"string": StringType(), "int": IntegerType(), "bool": BooleanType(), "double": DoubleType()}
    schema = StructType([StructField(name, types[t], True) for name, t in corpus.PASSAGE_COLUMNS])

    def write(groups):
        for (source, date), rows in groups.items():
            folder = corpus.partition_dir(corpus_dir, source, date).as_posix()
            spark.createDataFrame(rows, schema).coalesce(1).write.mode("append").parquet(folder)
    return write


def process_batch(batch_df, batch_id):
    from pyspark.sql import functions as F
    from backend.knowledge.kb import KnowledgeBase
    from backend.pipeline import corpus
    from backend.usermemory.store import Memory
    offsets = {str(r["partition"]): int(r["o"]) + 1 for r in batch_df.groupBy("partition").agg(F.max("offset").alias("o")).collect()}
    values = batch_df.selectExpr("CAST(value AS STRING) AS value").rdd.map(lambda r: r.value)
    values = values.repartition(int(os.environ.get("SPARK_PARALLEL", "8")))        # the topic has 3 partitions: spread the documents over every core
    kb, cache = KnowledgeBase(), Memory.from_env()          # the graph, and the cache (new data clears it)
    try:
        handle_results(values.map(_process).toLocalIterator(), kb, corpus.PassageIndex(CORPUS_DIR), make_parquet_writer(batch_df.sparkSession),
                       CORPUS_DIR, batch_id, offsets, on_failure=_dead_letter, cache=cache)
    finally:
        kb.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="process everything currently on the topic, then stop")
    ap.add_argument("--interval", default="5 seconds")
    a = ap.parse_args()
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("elvarn-kb-loader").getOrCreate()
    stream = (spark.readStream.format("kafka")
              .option("kafka.bootstrap.servers", os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092"))
              .option("subscribe", os.environ.get("KAFKA_TOPIC", "kb.raw_documents"))
              .option("startingOffsets", "earliest").option("failOnDataLoss", "false")
              .option("maxOffsetsPerTrigger", int(os.environ.get("SPARK_BATCH", "400"))).load())
    writer = (stream.writeStream.foreachBatch(process_batch)
              .option("checkpointLocation", os.environ.get("SPARK_CHECKPOINT", "/checkpoints/kb")))
    query = (writer.trigger(availableNow=True) if a.once else writer.trigger(processingTime=a.interval)).start()
    query.awaitTermination()


if __name__ == "__main__":
    main()
