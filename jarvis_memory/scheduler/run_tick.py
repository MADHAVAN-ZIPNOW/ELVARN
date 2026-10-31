"""Scheduler job (runs on GitHub Actions every ~15 minutes). Standard library only.

1. Wakes the free Render service and calls POST /tick (which reads your e-mail replies and decides if a check-in is due).
2. If a check-in is due, sends it through Gmail SMTP (Render's free plan blocks outgoing SMTP; GitHub runners do not),
   then confirms with POST /mail/sent so the hourly interval and daily limit are respected.

Secrets come from environment variables (GitHub Actions secrets). Nothing is printed except short status words.
"""
import base64
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
        html_part = msg.get_payload()[1]
        gif = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rocket.gif")
        if "cid:rocket" in item["html"] and os.path.exists(gif):      # the animated rocket in the team update, sent inside the e-mail
            with open(gif, "rb") as fh:
                html_part.add_related(fh.read(), "image", "gif", cid="<rocket>", disposition="inline")
        if "cid:chart3d" in item["html"] and item.get("chart"):       # the 3D chart picture drawn by the server
            html_part.add_related(base64.b64decode(item["chart"]), "image", "png", cid="<chart3d>", disposition="inline")
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
    if os.environ.get("TEST_TEAM_EMAIL") == "true":      # started by hand: queue one sample team e-mail to the owner only
        print("test e-mail:", call("POST", "/api/jarvis/memory/team-update/test"))
    if os.environ.get("RING_TEAM") == "true":      # started by hand: ring the team-update call again today
        print("team call:", call("POST", "/api/jarvis/memory/team-call/ring").get("why", "rang"))
    result = call("POST", "/api/jarvis/memory/tick", timeout=120)
    email = result.get("email", {})
    print("replies:", result.get("replies"), "| email:", email.get("why") or ("sent" if email.get("sent") else "none"),
          "| reminders due:", len(result.get("reminders_due", [])))
    item = email.get("deliver")
    review_mail = (result.get("review") or {}).get("email") or {}
    outbox = result.get("outbox") or []
    if (item or review_mail.get("deliver") or outbox) and not (GMAIL and APP_PASSWORD):
        print("GMAIL_ADDRESS / GMAIL_APP_PASSWORD secrets are missing")
        return 1
    if item:
        send_mail(item)
        call("POST", "/api/jarvis/memory/mail/sent")
        print("check-in e-mail delivered")
    if review_mail.get("deliver"):
        send_mail(review_mail["deliver"])
        call("POST", "/api/jarvis/memory/mail/review-sent?day=" + review_mail["day"])
        print("daily review e-mail delivered")
    for queued in outbox:
        send_mail(queued)
        call("POST", f"/api/jarvis/memory/mail/outbox-sent?id={queued['id']}")
        print("team update e-mail delivered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
