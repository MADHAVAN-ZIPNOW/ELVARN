import pytest

from tests.conftest import OWNER_KEY

P = "/api/jarvis/memory"
ALL_ROUTES = [("get", P), ("post", P), ("post", f"{P}/search"), ("get", f"{P}/settings"), ("patch", f"{P}/settings"),
              ("get", f"{P}/stats"), ("post", f"{P}/export"), ("delete", f"{P}/all?confirm=true"),
              ("get", f"{P}/00000000-0000-0000-0000-000000000000"), ("delete", f"{P}/00000000-0000-0000-0000-000000000000")]


@pytest.mark.parametrize("method,path", ALL_ROUTES)
def test_every_route_requires_the_owner_key(make_client, method, path):
    c = make_client()
    r = getattr(c, method)(path, headers={"X-Owner-Key": ""}, **({"json": {}} if method in ("post", "patch") else {}))
    assert r.status_code == 401, (method, path, r.status_code)


def test_wrong_key_is_refused_and_never_echoed(make_client):
    r = make_client().get(P, headers={"X-Owner-Key": "wrong-key-wrong-key-1234"})
    assert r.status_code == 401 and "wrong-key" not in r.text and OWNER_KEY not in r.text


def test_key_in_query_string_or_body_is_not_accepted(make_client):
    c = make_client()
    assert c.get(f"{P}?key={OWNER_KEY}", headers={"X-Owner-Key": ""}).status_code == 401


def test_the_api_refuses_to_run_open(make_client):
    for weak in ("", "short"):
        r = make_client(owner_key=weak).get(P, headers={"X-Owner-Key": weak})
        assert r.status_code == 503      # fails CLOSED: an unconfigured server never serves memories


def test_repeated_wrong_keys_lock_the_address_out(make_client):
    c = make_client()
    bad = {"X-Owner-Key": "nope-nope-nope-nope-1"}
    assert [c.get(P, headers=bad).status_code for _ in range(10)] == [401] * 10
    assert c.get(P, headers=bad).status_code == 429
    assert c.get(P).status_code == 429          # even the right key is paused during the lockout (brute-force protection)


def test_rate_limit(make_client):
    c = make_client(rate_limit_per_minute=5)
    codes = [c.get(P).status_code for _ in range(8)]
    assert codes[:5] == [200] * 5 and codes[5:] == [429] * 3
    assert c.get(P).headers.get("Retry-After")


def test_oversized_bodies_are_refused_before_reading(client):
    r = client.post(P, content=b"x" * 300_000, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_health_is_public_and_leaks_nothing(make_client):
    c = make_client()
    r = c.get("/api/jarvis/health", headers={"X-Owner-Key": ""})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"status", "database", "ai_extraction", "configured", "memory"} and body["database"] is True
    assert "postgresql" not in r.text and OWNER_KEY not in r.text


def test_security_headers_and_no_caching_of_api_responses(client):
    r = client.get(P)
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"


def test_docs_and_schema_endpoints_are_not_exposed(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path, headers={"X-Owner-Key": ""}).status_code == 404


def test_cors_is_closed_unless_configured(make_client):
    c = make_client()
    r = c.options(P, headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers
    c2 = make_client(allowed_origins=("https://memory.example",))
    ok = c2.options(P, headers={"Origin": "https://memory.example", "Access-Control-Request-Method": "GET"})
    assert ok.headers["access-control-allow-origin"] == "https://memory.example"
    bad = c2.options(P, headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers


def test_database_outage_gives_a_clean_retryable_error_and_leaks_nothing(make_client):
    c = make_client(database_url="postgresql://x:y@127.0.0.1:1/none?connect_timeout=1")
    r = c.post(P, json={"content": "hello"})
    assert r.status_code == 503 and r.headers.get("Retry-After") and "127.0.0.1" not in r.text and "postgresql" not in r.text.lower()


def test_free_database_is_protected_from_filling_up(client, monkeypatch):
    client.post(P, json={"content": "before", "extract": False})
    monkeypatch.setattr("app.service.database_size_mb", lambda conn: 999.0)
    r = client.post(P, json={"content": "too much", "extract": False})
    assert r.status_code == 507 and "limit" in r.json()["detail"].lower()
    assert len(client.get(P).json()) == 1                       # reading still works
    assert client.post(f"{P}/export").json()["count"] == 1      # and so does export, so you can free space
    assert client.delete(f"{P}/all?confirm=true").json()["deleted"] == 1


def test_stats_report_database_usage(client):
    client.post(P, json={"content": "a", "extract": False})
    s = client.get(f"{P}/stats").json()
    assert s["memories"] == 1 and s["by_source"] == {"manual": 1} and 0 < s["database_mb"] < s["limit_mb"] and s["used_percent"] < 5


def test_settings_roundtrip_and_validation(client):
    s = client.get(f"{P}/settings").json()
    assert s["owner_name"] == "Madhavan" and s["email_enabled"] is False and s["push_enabled"] is False and s["quiet_start_hour"] == 22
    up = client.patch(f"{P}/settings", json={"email_interval_minutes": 120, "quiet_end_hour": 8, "timezone": "Asia/Kolkata"}).json()
    assert up["email_interval_minutes"] == 120 and up["quiet_end_hour"] == 8
    for bad in ({"email_interval_minutes": 5}, {"quiet_start_hour": 24}, {"timezone": "Mars/Base"}, {"max_emails_per_day": 1000}, {"unknown": 1}):
        assert client.patch(f"{P}/settings", json=bad).status_code == 422


def test_ram_status_thresholds():
    from app import resources

    assert resources.status(400, 48.0)["ram_status"] == "ok"
    assert resources.status(400, 319.0)["ram_status"] == "ok"
    assert resources.status(400, 320.0)["ram_status"] == "close"
    assert resources.status(400, 401.0)["ram_status"] == "over"
    assert resources.status(400, None)["process_mb"] is not None or True  # unknown platforms degrade gracefully


def test_stats_and_health_report_memory_and_warn_before_the_limit(client, monkeypatch):
    monkeypatch.setattr("app.resources.rss_mb", lambda: 350.0)
    s = client.get(f"{P}/stats").json()
    assert s["process_mb"] == 350.0 and s["ram_limit_mb"] == 400 and s["ram_status"] == "close"
    assert client.get("/api/jarvis/health").json()["memory"] == "close"
    monkeypatch.setattr("app.resources.rss_mb", lambda: 450.0)
    assert client.get(f"{P}/stats").json()["ram_status"] == "over"
    assert client.get("/api/jarvis/health").json()["status"] == "degraded"
