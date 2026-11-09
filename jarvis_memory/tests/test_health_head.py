"""Uptime monitors probe the public health link with HEAD as well as GET; a 404 would make them report the service as down."""
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_answers_head_and_get():
    client = TestClient(create_app(Settings(), migrate=False))
    assert client.get("/api/jarvis/health").status_code == 200
    head = client.head("/api/jarvis/health")
    assert head.status_code == 200 and head.content == b""
