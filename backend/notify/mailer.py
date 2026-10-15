"""Send email through any SMTP server (Gmail with an app password, Brevo, your own server ...): no extra package, no account of ours.

Settings come from the environment (never from code, never from git):
    SMTP_HOST       e.g. smtp.gmail.com
    SMTP_PORT       587 (STARTTLS, the default) or 465 (SSL)
    SMTP_USER       the mailbox login, e.g. madhavan.ai.research@gmail.com
    SMTP_PASSWORD   for Gmail an "app password" (Google Account > Security > 2-Step Verification > App passwords), NOT the normal password
    MAIL_FROM       optional, defaults to SMTP_USER
    OWNER_EMAIL     where ELVARN tells you about a new visitor who left an address

Some free hosts (Render free web services) block outgoing SMTP. Then use a relay instead: a tiny Google Apps Script web app in your own
Google account sends the mail (see scripts/google_mail_relay.gs), and ELVARN only makes an ordinary HTTPS request to it:
    MAIL_WEBHOOK_URL / MAIL_WEBHOOK_SECRET   the web app address and the shared secret (used first when both are set)

Safety: an address is checked before use; line breaks are removed from the subject and the address (no header injection); the
message is plain text; a visitor can only ask for one welcome mail per address and a few per hour per visitor (see api/server.py).
"""
import os
import re
import smtplib
import ssl

import httpx
from email.message import EmailMessage

ADDRESS = re.compile(r"^[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,24}$")


class MailError(Exception):
    pass


def clean_address(value):
    """The address, lower-cased, or None when it does not look like one (or has a line break / space in it)."""
    v = (value or "").strip()
    if len(v) > 254 or any(c in v for c in "\r\n \t,;<>\"'") or not ADDRESS.match(v):
        return None
    return v.lower()


def one_line(text, limit=150):
    return re.sub(r"[\r\n]+", " ", str(text or "")).strip()[:limit]


class Mailer:
    def __init__(self, host, port, user, password, sender=None, owner=None, smtp_factory=None, webhook=None, webhook_secret=None, transport=None):
        self.host, self.port, self.user, self.password = host, int(port or 587), user, password
        self.sender, self.owner = sender or user, owner
        self._factory = smtp_factory
        self.webhook, self.webhook_secret, self._transport = webhook, webhook_secret, transport

    @classmethod
    def from_env(cls, env=None, smtp_factory=None):
        e = os.environ if env is None else env
        host, user, password = e.get("SMTP_HOST", ""), e.get("SMTP_USER", ""), e.get("SMTP_PASSWORD", "")
        hook, secret = e.get("MAIL_WEBHOOK_URL", ""), e.get("MAIL_WEBHOOK_SECRET", "")
        if hook and secret and hook.startswith("https://"):
            return cls(host, e.get("SMTP_PORT") or 587, user, password, e.get("MAIL_FROM") or user, clean_address(e.get("OWNER_EMAIL")), smtp_factory, hook, secret)
        if not (host and user and password):
            return None                                   # not set up: the feature stays off
        return cls(host, e.get("SMTP_PORT") or 587, user, password, e.get("MAIL_FROM") or user, clean_address(e.get("OWNER_EMAIL")), smtp_factory)

    def send(self, to, subject, body):
        to = clean_address(to)
        if to is None:
            raise MailError("that email address does not look right")
        if self.webhook:
            return self._send_by_webhook(to, subject, body)
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, to, one_line(subject, 120)
        msg.set_content(str(body)[:5000])
        try:
            if self._factory:
                server = self._factory(self.host, self.port)
            elif self.port == 465:
                server = smtplib.SMTP_SSL(self.host, self.port, timeout=20, context=ssl.create_default_context())
            else:
                server = smtplib.SMTP(self.host, self.port, timeout=20)
            with server:
                if not self._factory and self.port != 465:
                    server.starttls(context=ssl.create_default_context())
                server.login(self.user, self.password)
                server.send_message(msg)
        except (smtplib.SMTPException, OSError) as exc:
            raise MailError("the mail server did not accept the message") from exc

    def _send_by_webhook(self, to, subject, body):
        try:
            with httpx.Client(timeout=30, follow_redirects=True, transport=self._transport) as client:
                r = client.post(self.webhook, json={"secret": self.webhook_secret, "to": to, "subject": one_line(subject, 120), "body": str(body)[:5000]})
            ok = r.status_code == 200 and r.json().get("ok") is True
        except (httpx.HTTPError, ValueError) as exc:
            raise MailError("the mail relay could not be reached") from exc
        if not ok:
            raise MailError("the mail relay did not accept the message")
