"""Gmail SMTP (send the hourly check-in) and IMAP (read the owner's replies).

Credentials come only from environment variables (Render). Mail is only ever sent to the owner's own address, and only
replies FROM the owner's address are read, so nobody else can write into the memory by e-mailing the account.
"""
from __future__ import annotations

import email
import imaplib
import logging
import re
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from email.utils import parseaddr, parsedate_to_datetime
from typing import Optional, Protocol

log = logging.getLogger("jarvis_memory.mailer")
THREAD_RE = re.compile(r"\[JM-([0-9a-f]{8,32})\]")
_QUOTE_START = re.compile(r"^(On .{5,200} wrote:|-{2,}\s*Original Message|From: .+@.+)\s*$", re.I)
MAX_REPLY_CHARS = 8000


@dataclass
class Reply:
    message_id: str
    thread_id: Optional[str]
    text: str
    sent_at: Optional[datetime]


class Mailbox(Protocol):
    def send(self, to: str, subject: str, body: str, html: Optional[str] = None) -> str: ...
    def fetch_replies(self, from_address: str) -> list[Reply]: ...


def strip_quoted(body: str) -> str:
    """Keep what the owner typed: drop quoted lines (> ...), the 'On ... wrote:' tail and the signature."""
    kept: list[str] = []
    for line in body.replace("\r\n", "\n").split("\n"):
        if _QUOTE_START.match(line.strip()) or line.strip() == "--":
            break
        if line.lstrip().startswith(">"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()[:MAX_REPLY_CHARS]


def _plain_text(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition", "")):
                return _decode(part)
        return ""
    return _decode(msg) if msg.get_content_type() == "text/plain" else ""


def _decode(part: email.message.Message) -> str:
    payload = part.get_payload(decode=True) or b""
    return payload.decode(part.get_content_charset() or "utf-8", errors="replace")


def parse_reply(raw: bytes, owner_address: str) -> Optional[Reply]:
    """A Reply only if it is FROM the owner and has a message-id; otherwise None (ignored, never stored)."""
    msg = email.message_from_bytes(raw)
    if parseaddr(msg.get("From", ""))[1].lower() != owner_address.lower():
        return None
    message_id = (msg.get("Message-ID") or "").strip()
    if not message_id:
        return None
    text = strip_quoted(_plain_text(msg))
    if not text:
        return None
    haystack = " ".join([msg.get("Subject", "") or "", msg.get("In-Reply-To", "") or "", msg.get("References", "") or ""])
    thread = THREAD_RE.search(haystack)
    try:
        sent_at = parsedate_to_datetime(msg.get("Date")) if msg.get("Date") else None
    except (TypeError, ValueError):
        sent_at = None
    return Reply(message_id=message_id[:300], thread_id=thread.group(1) if thread else None, text=text, sent_at=sent_at)


class GmailMailbox:
    def __init__(self, address: str, app_password: str) -> None:
        self.address, self._password = address, app_password

    def send(self, to: str, subject: str, body: str, html: Optional[str] = None) -> str:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = f"JARVIS <{self.address}>", to, subject
        msg["Message-ID"] = email.utils.make_msgid(domain="jarvis-memory.local")
        msg.set_content(body)
        if html:
            msg.add_alternative(html, subtype="html")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=8) as smtp:
            smtp.login(self.address, self._password)
            smtp.send_message(msg)
        return msg["Message-ID"]

    def fetch_replies(self, from_address: str) -> list[Reply]:
        replies: list[Reply] = []
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993, timeout=8)
        try:
            imap.login(self.address, self._password)
            imap.select("INBOX")
            status, data = imap.search(None, "UNSEEN", "FROM", f'"{from_address}"')
            for num in (data[0].split() if status == "OK" and data and data[0] else [])[:30]:
                status, parts = imap.fetch(num, "(RFC822)")  # fetching marks it read, so it is not processed twice
                if status != "OK" or not parts or not isinstance(parts[0], tuple):
                    continue
                reply = parse_reply(parts[0][1], from_address)
                if reply:
                    replies.append(reply)
        finally:
            try:
                imap.logout()
            except Exception:
                pass
        return replies


def from_settings(settings) -> Optional[Mailbox]:
    if settings.gmail_address and settings.gmail_app_password:
        return GmailMailbox(settings.gmail_address, settings.gmail_app_password)
    return None
