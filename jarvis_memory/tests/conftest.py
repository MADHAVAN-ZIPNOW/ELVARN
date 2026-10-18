"""Test fixtures. The tests need a REAL Postgres (full-text search is the point), so they use TEST_DATABASE_URL.

Quick start (Docker):  docker run -d --name jm-test-pg -e POSTGRES_PASSWORD=jmtest -e POSTGRES_DB=jm_test -p 127.0.0.1:55432:5432 postgres:16-alpine
Nothing here ever touches your Neon database: point TEST_DATABASE_URL at a throw-away database only.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.db import run_migrations  # noqa: E402
from app.extraction import Extractor  # noqa: E402
from app.main import create_app  # noqa: E402

TEST_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://postgres:jmtest@127.0.0.1:55432/jm_test")
OWNER_KEY = "test-owner-key-0123456789"


def _reachable() -> bool:
    try:
        with psycopg.connect(TEST_URL, connect_timeout=3) as c:
            c.execute("SELECT 1")
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _reachable():
        return
    skip = pytest.mark.skip(reason=f"No test Postgres at TEST_DATABASE_URL ({TEST_URL.split('@')[-1]}). See tests/conftest.py")
    for item in items:
        if "needs_db" in item.keywords or "db" in item.fixturenames or "client" in item.fixturenames:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _schema():
    if _reachable():
        with psycopg.connect(TEST_URL, autocommit=True) as c:  # start from nothing: proves migrations build everything
            c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        run_migrations(TEST_URL)
    yield


@pytest.fixture
def db():
    """A raw connection for arranging/inspecting data; tables are emptied before each test."""
    with psycopg.connect(TEST_URL, autocommit=True, row_factory=psycopg.rows.dict_row) as conn:
        conn.execute("TRUNCATE memory_entries, memory_settings")
        yield conn


@pytest.fixture
def settings() -> Settings:
    return Settings(database_url=TEST_URL, owner_key=OWNER_KEY, owner_name="Madhavan", owner_email="owner@example.com",
                    groq_api_key="test-key", rate_limit_per_minute=1000)


class FakeExtractor(Extractor):
    """Stands in for Groq. ``answer`` is the raw JSON the 'model' returns; it still goes through real grounding."""

    def __init__(self, settings: Settings, answer: Optional[Any] = None, fail: Optional[Exception] = None) -> None:
        super().__init__(settings)
        self.answer, self.fail, self.calls = answer, fail, []

    def _call(self, text, source, when, context):
        self.calls.append({"text": text, "source": source})
        if self.fail:
            raise self.fail
        return self.answer(text) if callable(self.answer) else (self.answer or {})


@pytest.fixture
def extractor(settings) -> FakeExtractor:
    return FakeExtractor(settings, answer={})


@pytest.fixture
def make_client(settings, db):
    def _make(extractor=None, **overrides) -> TestClient:
        s = Settings(**{**settings.__dict__, **overrides})
        app = create_app(s, extractor or FakeExtractor(s, answer={}), migrate=False)
        c = TestClient(app, headers={"X-Owner-Key": OWNER_KEY})
        c.app_state = app.state
        return c
    return _make


@pytest.fixture
def client(make_client, extractor) -> TestClient:
    return make_client(extractor)


@pytest.fixture
def owner_id(settings) -> uuid.UUID:
    return settings.owner_id


@pytest.fixture
def add(db, owner_id) -> Callable[..., str]:
    """Insert a memory directly (with a chosen created_at) so search/date tests are exact."""
    from psycopg.types.json import Jsonb

    def _add(content="note", *, source="manual", title="", summary="", tags=(), projects=(), people=(), next_steps=(),
             tasks=(), sentiment=None, raw_transcript=None, created_at: Optional[datetime] = None, deleted=False,
             owner: Optional[uuid.UUID] = None) -> str:
        row = db.execute(
            """INSERT INTO memory_entries (owner_id, source, title, content, raw_transcript, summary, created_at, tags, projects,
               people, next_steps, tasks, sentiment, deleted_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (owner or owner_id, source, title or content[:30], content, raw_transcript, summary,
             created_at or datetime.now(timezone.utc), list(tags), list(projects), list(people), list(next_steps),
             Jsonb(list(tasks)), sentiment, datetime.now(timezone.utc) if deleted else None)).fetchone()
        return str(row["id"])
    return _add


def mock_groq(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))
