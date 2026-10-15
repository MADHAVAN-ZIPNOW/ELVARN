"""The email-at-first-question feature: validation, consent, rate limit, header injection, and the SMTP conversation (all with a fake server)."""
import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from notify.mailer import MailError, Mailer, clean_address, one_line


class FakeSMTP:
    sent = []

    def __init__(self, host, port):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, user, password):
        assert user and password

    def send_message(self, msg):
        FakeSMTP.sent.append(msg)


@pytest.fixture()
def client(ckpt_dir, tok_file):
    FakeSMTP.sent = []
    mailer = Mailer("smtp.example.com", 587, "bot@example.com", "app-password", owner="owner@example.com", smtp_factory=FakeSMTP)
    app = create_app(ckpt_dir, load_on_start=False, max_concurrent=0, mail_service=mailer)
    return TestClient(app)


def test_addresses_are_checked_and_cleaned():
    assert clean_address("  Madhavan@Example.com ") == "madhavan@example.com"
    for bad in ("", "no-at-sign", "a@b", "a b@c.com", "x@y.com\r\nBcc: evil@z.com", "<a@b.com>", "a@b.com,c@d.com"):
        assert clean_address(bad) is None
    assert one_line("hello\r\nBcc: evil") == "hello Bcc: evil"


def test_a_visitor_with_consent_gets_a_welcome_mail_and_the_owner_a_note(client):
    r = client.post("/api/lead", json={"email": "visitor@example.com", "consent": True, "question": "hii\r\nBcc: evil@x.com"})
    assert r.status_code == 200 and r.json() == {"ok": True, "sent": True}
    to = [m["To"] for m in FakeSMTP.sent]
    assert to == ["visitor@example.com", "owner@example.com"]
    assert "Welcome to ELVARN" == FakeSMTP.sent[0]["Subject"] and "Bcc" not in FakeSMTP.sent[0].keys()
    assert "Allegations are not proof of guilt" in FakeSMTP.sent[0].get_content()


def test_no_consent_bad_address_and_repeats_are_refused_or_ignored(client):
    assert client.post("/api/lead", json={"email": "a@example.com", "consent": False}).status_code == 400
    assert client.post("/api/lead", json={"email": "not-an-email", "consent": True}).status_code == 422
    assert client.post("/api/lead", json={"email": "b@example.com", "consent": True}).json()["sent"] is True
    assert client.post("/api/lead", json={"email": "b@example.com", "consent": True}).json()["sent"] is False       # one welcome mail per address
    assert len(FakeSMTP.sent) == 2


def test_a_visitor_cannot_use_it_to_spam(client):
    codes = [client.post("/api/lead", json={"email": f"u{i}@example.com", "consent": True}).status_code for i in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]


def test_it_is_off_until_smtp_is_configured(ckpt_dir):
    app = create_app(ckpt_dir, load_on_start=False, max_concurrent=0, mail_service=None)
    c = TestClient(app)
    assert Mailer.from_env({}) is None
    assert c.get("/api/lead").json() == {"enabled": False} and c.post("/api/lead", json={"email": "a@example.com", "consent": True}).status_code == 503


def test_a_mail_server_failure_is_reported_not_crashed(ckpt_dir):
    class Down(FakeSMTP):
        def login(self, user, password):
            import smtplib
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")
    app = create_app(ckpt_dir, load_on_start=False, max_concurrent=0, mail_service=Mailer("h", 587, "u", "p", smtp_factory=Down))
    r = TestClient(app).post("/api/lead", json={"email": "a@example.com", "consent": True})
    assert r.status_code == 502 and "bad credentials" not in r.text           # the server's reason is not shown to visitors
    with pytest.raises(MailError):
        Mailer("h", 587, "u", "p", smtp_factory=Down).send("a@example.com", "s", "b")


def test_the_relay_mode_posts_to_the_web_app_and_reports_refusals():
    import json
    import httpx
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": seen[-1]["secret"] == "right"})
    good = Mailer.from_env({"MAIL_WEBHOOK_URL": "https://script.google.com/macros/s/x/exec", "MAIL_WEBHOOK_SECRET": "right", "OWNER_EMAIL": "o@example.com"}, None)
    good._transport = httpx.MockTransport(handler)
    good.send("v@example.com", "Hi\r\nBcc: evil", "body")
    assert seen[0]["to"] == "v@example.com" and seen[0]["subject"] == "Hi Bcc: evil" and good.owner == "o@example.com"
    bad = Mailer.from_env({"MAIL_WEBHOOK_URL": "https://script.google.com/macros/s/x/exec", "MAIL_WEBHOOK_SECRET": "wrong"}, None)
    bad._transport = httpx.MockTransport(handler)
    with pytest.raises(MailError):
        bad.send("v@example.com", "s", "b")
    assert Mailer.from_env({"MAIL_WEBHOOK_URL": "http://not-https.example.com", "MAIL_WEBHOOK_SECRET": "x"}) is None        # only https relays
