"""The Render deploy script, against a fake Render API (no network, no real key)."""
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import deploy_render as dr  # noqa: E402

KEY, TOKEN = "rnd_" + "k" * 30, "t" * 40


class FakeRender:
    """Just enough of Render's API: owners, services, create, env var update. Records every call."""

    def __init__(self, with_knowledge=False, with_site=True):
        self.calls, self.services = [], []
        if with_site:
            self.services.append({"id": "srv-site", "name": "elvarn", "ownerId": "tea-1", "repo": "https://github.com/x/ELVARN", "branch": "main",
                                  "serviceDetails": {"region": "singapore", "url": "https://elvarn.onrender.com"}})
        if with_knowledge:
            self.services.append({"id": "srv-know", "name": "elvarn-knowledge", "ownerId": "tea-1",
                                  "serviceDetails": {"url": "https://elvarn-knowledge.onrender.com"}})

    def __call__(self, request):
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body, request.headers.get("authorization")))
        path = request.url.path
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"message": "unauthorized"})
        if path == "/v1/owners":
            return httpx.Response(200, json=[{"owner": {"id": "tea-1", "name": "My Workspace"}}])
        if path == "/v1/services" and request.method == "GET":
            return httpx.Response(200, json=[{"service": s} for s in self.services])
        if path == "/v1/services" and request.method == "POST":
            svc = {"id": "srv-new", "name": body["name"], "ownerId": "tea-1", "serviceDetails": {"url": f"https://{body['name']}.onrender.com"}}
            self.services.append(svc)
            return httpx.Response(201, json={"service": svc, "deployId": "dep-1"})
        if request.method == "PUT" and "/env-vars/" in path:
            return httpx.Response(200, json={"key": path.rsplit("/", 1)[1], "value": body["value"]})
        return httpx.Response(404, json={"message": "unknown"})


def api_for(fake):
    return dr.Render(KEY, transport=httpx.MockTransport(fake))


def test_creates_the_free_docker_service_wires_the_site_and_waits_for_it():
    fake, said = FakeRender(), []
    url = dr.deploy(api_for(fake), TOKEN, say=said.append, wait=False)
    assert url == "https://elvarn-knowledge.onrender.com"
    post = next(c for c in fake.calls if c[0] == "POST")
    body = post[2]
    assert body["type"] == "web_service" and body["name"] == "elvarn-knowledge" and body["ownerId"] == "tea-1"
    assert body["repo"] == "https://github.com/x/ELVARN" and body["branch"] == "main" and body["rootDir"] == ""          # same repo as the site
    assert body["serviceDetails"] == {"runtime": "docker", "plan": "free", "region": "singapore", "healthCheckPath": "/healthz", "numInstances": 1,
                                      "envSpecificDetails": {"dockerfilePath": "./knowledge_service/Dockerfile", "dockerContext": "."}}
    assert body["envVars"] == [{"key": "KNOWLEDGE_TOKEN", "value": TOKEN}]
    puts = {c[1]: c[2] for c in fake.calls if c[0] == "PUT"}
    assert puts["/v1/services/srv-site/env-vars/KNOWLEDGE_URL"] == {"value": url}
    assert puts["/v1/services/srv-site/env-vars/KNOWLEDGE_TOKEN"] == {"value": TOKEN}


def test_a_second_run_does_not_create_another_service():
    fake = FakeRender(with_knowledge=True)
    dr.deploy(api_for(fake), TOKEN, say=lambda m: None, wait=False)
    assert not [c for c in fake.calls if c[0] == "POST"]
    puts = [c[1] for c in fake.calls if c[0] == "PUT"]
    assert "/v1/services/srv-know/env-vars/KNOWLEDGE_TOKEN" in puts and "/v1/services/srv-site/env-vars/KNOWLEDGE_URL" in puts


def test_dry_run_changes_nothing():
    fake = FakeRender()
    assert dr.deploy(api_for(fake), TOKEN, dry_run=True, say=lambda m: None) is None
    assert {c[0] for c in fake.calls} == {"GET"}


def test_the_key_goes_only_in_the_header_and_nothing_secret_is_printed(capsys):
    fake, said = FakeRender(), []
    dr.deploy(api_for(fake), TOKEN, say=said.append, wait=False)
    assert all(KEY not in json.dumps(c[2]) and TOKEN not in c[1] for c in fake.calls)          # never in a URL or a body (the token IS in bodies: that is the point)
    assert all(c[3] == f"Bearer {KEY}" for c in fake.calls)
    out = " ".join(said) + capsys.readouterr().out
    assert KEY not in out and TOKEN not in out


def test_wrong_workspace_or_missing_site_stops_with_a_clear_message():
    with pytest.raises(dr.DeployError, match='no service named "elvarn"'):
        dr.deploy(api_for(FakeRender(with_site=False)), TOKEN, say=lambda m: None, wait=False)
    with pytest.raises(dr.DeployError, match="no workspace named"):
        dr.deploy(api_for(FakeRender()), TOKEN, owner_name="Other", say=lambda m: None, wait=False)


def test_a_bad_key_is_reported_without_echoing_it():
    bad = dr.Render("rnd_wrong", transport=httpx.MockTransport(FakeRender()))
    with pytest.raises(dr.DeployError, match="refused the API key") as e:
        dr.deploy(bad, TOKEN, say=lambda m: None, wait=False)
    assert "rnd_wrong" not in str(e.value)


def test_waiting_polls_until_healthy_and_reports_the_stores():
    answers = iter([httpx.Response(502), httpx.Response(200)])                                 # asleep / building, then up
    seen = []

    def check(path, auth=None):
        seen.append(path)
        if path == "/healthz":
            return next(answers)
        return httpx.Response(200, json={"graph": True, "vector": True, "memory": True})
    said = []
    dr.wait_until_up("https://k.onrender.com", TOKEN, say=said.append, sleep=lambda s: None, http_check=check)
    assert seen == ["/healthz", "/healthz", "/readyz"] and "'graph': True" in " ".join(said)


def test_waiting_gives_up_with_a_pointer_to_the_logs():
    with pytest.raises(dr.DeployError, match="logs"):
        dr.wait_until_up("https://k.onrender.com", TOKEN, say=lambda m: None, sleep=lambda s: None, minutes=0,
                         http_check=lambda p, auth=None: httpx.Response(502))


def test_main_without_a_key_explains_what_to_do_and_touches_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(dr, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("RENDER_API_KEY", raising=False)
    assert dr.main([]) == 2
    err = capsys.readouterr().err
    assert "RENDER_API_KEY" in err and "Do not paste" in err and not (tmp_path / ".env").exists()
