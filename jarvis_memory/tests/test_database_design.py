"""The schema must stay small (Neon's free tier) and migrations must be repeatable."""
import psycopg
import pytest

from app.db import run_migrations
from tests.conftest import TEST_URL


def test_migrations_are_idempotent():
    assert run_migrations(TEST_URL) == []  # already applied by the session fixture; running again changes nothing


def test_only_the_expected_tables_exist(db):
    tables = {r["tablename"] for r in db.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'").fetchall()}
    # Stage 1 is deliberately two data tables: tags/projects/people/tasks live inside the memory row.
    assert tables == {"memory_entries", "memory_settings", "schema_migrations", "reminders", "job_log", "devices"}


def test_expected_indexes_exist(db):
    names = {r["indexname"] for r in db.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'memory_entries'").fetchall()}
    assert {"idx_me_owner_created", "idx_me_owner_source", "idx_me_fts", "idx_me_thread", "uq_me_external_ref", "idx_me_tasks"} <= names


def test_source_and_sentiment_are_constrained(db, add):
    with pytest.raises(psycopg.errors.CheckViolation):
        add("x", source="phone_call")
    with pytest.raises(psycopg.errors.CheckViolation):
        add("x", sentiment="ecstatic")


def test_oversized_text_is_rejected_by_the_database_itself(db, add):
    with pytest.raises(psycopg.errors.CheckViolation):
        add("x" * 20001)


def test_external_reference_id_is_unique_per_owner(db, owner_id):
    sql = "INSERT INTO memory_entries (owner_id, source, content, external_reference_id) VALUES (%s,'email','a','<id@mail>')"
    db.execute(sql, (owner_id,))
    with pytest.raises(psycopg.errors.UniqueViolation):  # the basis of idempotent email handling (Stage 2)
        db.execute(sql, (owner_id,))


def test_realistic_memories_cost_only_a_few_kb_each(db, add):
    """2,000 realistic hourly updates must stay far below the free-tier limit (measured, not assumed)."""
    for i in range(2000):
        add(
            f"Worked on the ELVARN retrieval ranking for about an hour, fixed the reranker bug number {i} and tested the "
            f"Render deployment. Talked with Ravi about the legal dataset. Next I will write more tests for the source cards.",
            title=f"ELVARN ranking work {i}", summary="Madhavan fixed a reranker bug and tested the deployment, then planned more tests.",
            tags=["elvarn", "ranking", "testing"], projects=["ELVARN"], people=["Ravi"],
            next_steps=["Write more tests for the source cards"],
            tasks=[{"task": "Write source card tests", "status": "pending", "due_at": None}], sentiment="positive")
    total = db.execute("SELECT pg_total_relation_size('memory_entries') AS b").fetchone()["b"]
    per_row = total / 2000
    print(f"\n   measured: {per_row:.0f} bytes per memory including all indexes")
    assert per_row < 4500
    # => 1 GB would hold over 200,000 such memories (about 15 a day for 36 years)
