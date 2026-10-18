"""Scheduler job (runs on GitHub Actions every ~15 minutes). Standard library only.

1. Wakes the free Render service and calls POST /tick (which reads your e-mail replies and decides if a check-in is due).
2. If a check-in is due, sends it through Gmail SMTP (Render's free plan blocks outgoing SMTP; GitHub runners do not),
   then confirms with POST /mail/sent so the hourly interval and daily limit are respected.

Secrets come from environment variables (GitHub Actions secrets). Nothing is printed except short status words.
"""
import json
import os
import smtplib
import sys
import time
import urllib.error
import urllib.request
from email.message import EmailMessage
from email.utils import make_msgid

BASE = os.environ["JM_URL"].rstrip("/")
KEY = os.environ["JM_OWNER_KEY"]
GMAIL = os.environ.get("GMAIL_ADDRESS", "")
APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "")


def call(method, path, timeout=60):
    req = urllib.request.Request(BASE + path, method=method, headers={"X-Owner-Key": KEY, "Content-Type": "application/json"},
                                 data=b"{}" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def wake_service(max_wait=150):
    """A sleeping free instance needs ~30-60 s; poll the public health endpoint until it answers."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/api/jarvis/health", timeout=20) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(5)
    return False


def send_mail(item):
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = f"JARVIS <{GMAIL}>", item["to"], item["subject"]
    msg["Message-ID"] = make_msgid(domain="jarvis-memory.local")
    msg.set_content(item["text"])
    if item.get("html"):
        msg.add_alternative(item["html"], subtype="html")
    last = None
    for _ in range(3):
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
                smtp.login(GMAIL, APP_PASSWORD)
                smtp.send_message(msg)
            return
        except Exception as exc:  # report only the type: never print credentials
            last = exc
            time.sleep(3)
    raise RuntimeError(f"SMTP send failed: {type(last).__name__}")


def main():
    if not wake_service():
        print("service did not wake up in time")
        return 1
    result = call("POST", "/api/jarvis/memory/tick", timeout=120)
    email = result.get("email", {})
    print("replies:", result.get("replies"), "| email:", email.get("why") or ("sent" if email.get("sent") else "none"),
          "| reminders due:", len(result.get("reminders_due", [])))
    item = email.get("deliver")
    if item:
        if not (GMAIL and APP_PASSWORD):
            print("GMAIL_ADDRESS / GMAIL_APP_PASSWORD secrets are missing")
            return 1
        send_mail(item)
        call("POST", "/api/jarvis/memory/mail/sent")
        print("check-in e-mail delivered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
