"""What the server sends Brevo: replies go back to the owner's mailbox and no fake .local Message-ID header (a spam signal) is attached."""
import json

import httpx

from app.mailer import GmailMailbox


def test_the_brevo_request_has_a_reply_to_and_no_fake_message_id(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, payload=json, headers=headers)
        return httpx.Response(201, json={"messageId": "<real@brevo>"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    box = GmailMailbox("owner@gmail.com", "", brevo_key="xkeysib-test")
    returned = box.send("owner@gmail.com", "Subject [JM-abc]", "plain", "<b>html</b>")
    p = seen["payload"]
    assert seen["url"] == "https://api.brevo.com/v3/smtp/email" and seen["headers"]["api-key"] == "xkeysib-test"
    assert p["replyTo"]["email"] == "owner@gmail.com" and p["sender"]["email"] == "owner@gmail.com"
    assert "headers" not in p and p["htmlContent"] == "<b>html</b>" and returned.endswith("@jarvis-memory.local>")
