"""Postgres access (psycopg 3) and the tiny SQL-file migration runner.

Neon's free tier suspends the database when idle and has a small connection limit, so connections are short-lived
(one per request) and the *pooled* Neon endpoint should be used in DATABASE_URL.
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

log = logging.getLogger("jarvis_memory.db")
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
_MIGRATION_LOCK_ID = 774411  # advisory lock: two instances starting at once must not both migrate


class DatabaseNotConfigured(RuntimeError):
    pass


_pools: dict[str, ConnectionPool] = {}
_pools_lock = threading.Lock()


def _pool(database_url: str) -> ConnectionPool:
    """One small pool per database URL. Opening a connection to Neon costs ~1 s (TLS + auth over an ocean), so connections are
    kept and reused; each is checked before use because Neon closes idle ones."""
    with _pools_lock:
        pool = _pools.get(database_url)
        if pool is None:
            pool = ConnectionPool(
                database_url, min_size=1, max_size=4, open=False, check=ConnectionPool.check_connection,
                max_idle=240, max_lifetime=1800, timeout=15,
                kwargs={"row_factory": dict_row, "connect_timeout": 15, "autocommit": False, "prepare_threshold": None})
            pool.open(wait=False)
            _pools[database_url] = pool
        return pool


@contextmanager
def connect(database_url: str) -> Iterator[psycopg.Connection]:
    """A connection that commits on success and rolls back on error (borrowed from the pool, returned afterwards)."""
    if not database_url:
        raise DatabaseNotConfigured("DATABASE_URL is not set.")
    try:
        with _pool(database_url).connection(timeout=20) as conn:
            yield conn
    except PoolTimeout:
        raise psycopg.OperationalError("database connection timed out") from None


def close_pools() -> None:
    with _pools_lock:
        for pool in _pools.values():
            pool.close()
        _pools.clear()


def run_migrations(database_url: str) -> list[str]:
    """Apply every ``migrations/*.sql`` file that has not been applied yet. Returns the versions applied."""
    applied: list[str] = []
    with connect(database_url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_ID,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.stem in done:
                continue
            log.info("Applying migration %s", path.stem)
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
            applied.append(path.stem)
    return applied


def database_size_mb(conn: psycopg.Connection) -> float:
    row = conn.execute("SELECT pg_database_size(current_database()) AS b").fetchone()
    return round(row["b"] / (1024 * 1024), 2)


if __name__ == "__main__":  # python -m app.db  ->  apply migrations
    import os
    import sys

    logging.basicConfig(level=logging.INFO)
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        sys.exit("Set DATABASE_URL first.")
    print("applied:", run_migrations(url) or "nothing to do (already up to date)")
