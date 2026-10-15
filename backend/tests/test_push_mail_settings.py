"""scripts/push_mail_settings.py against a fake Render API (no network, no real key or password)."""
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import deploy_render as dr  # noqa: E402
import push_mail_settings as pm  # noqa: E402

ENV = {"SMTP_HOST": "smtp.example.com", "SMTP_PORT": "587", "SMTP_USER": "me@example.com", "SMTP_PASSWORD": "secret-app-password", "OWNER_EMAIL": "me@example.com"}


def fake(calls):
    def handler(request):
        calls.append((request.method, request.url.path, request.content))
        if request.url.path == "/v1/services":
            return httpx.Response(200, json=[{"service": {"id": "srv-site", "name": "elvarn", "ownerId": "o"}}])
        return httpx.Response(200, json={})
    return dr.Render("rnd_key", transport=httpx.MockTransport(handler))


def test_it_sets_each_value_and_restarts_the_site():
    calls = []
    pm.push(fake(calls), ENV, say=lambda m: None)
    puts = [c[1] for c in calls if c[0] == "PUT"]
    assert sorted(puts) == sorted(f"/v1/services/srv-site/env-vars/{k}" for k in pm.SMTP_KEYS)
    assert ("POST", "/v1/services/srv-site/deploys") in [(c[0], c[1]) for c in calls]


def test_a_missing_password_changes_nothing_and_a_dry_run_only_talks(capsys):
    calls = []
    with pytest.raises(dr.DeployError, match="SMTP_PASSWORD"):
        pm.push(fake(calls), {**ENV, "SMTP_PASSWORD": ""}, say=lambda m: None)
    assert not calls
    said = []
    pm.push(fake(calls), ENV, dry_run=True, say=said.append)
    assert not [c for c in calls if c[0] in ("PUT", "POST")] and "secret-app-password" not in " ".join(said)
