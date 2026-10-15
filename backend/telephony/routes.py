"""Phone-call endpoints: ELVARN calls the OWNER's phone when asked, and talks with them using the same answering pipeline as the web chat.

    GET  /api/call/status        {"enabled": bool}
    POST /api/call/start         header X-Owner-Key  -> starts a call to the owner's number (only that number, never one a visitor supplies)
    POST /api/call/voice?cid=..  the address Twilio calls during the call (signature checked)
    GET  /api/call/transcript    header X-Owner-Key  -> what was said so far, so the web chat can show the conversation

Settings (environment, never in git): TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM (your Twilio number, +1...), OWNER_PHONE (+91...),
OWNER_KEY (a long secret only you know, 16+ characters), PUBLIC_BASE_URL (https://elvarn.onrender.com). Without all of them the feature is off.
"""
import hmac
import os
import secrets
import time
from urllib.parse import parse_qs

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response

from . import twiml

MAX_TURNS = 10
MAX_SILENCES = 2
CALLS_PER_DAY = 12
GREETING = "Hi, this is ELVARN. What would you like to know?"


class CallConfig:
    def __init__(self, sid, token, from_number, owner_phone, owner_key, public_base):
        self.sid, self.token, self.from_number, self.owner_phone = sid, token, from_number, owner_phone
        self.owner_key, self.public_base = owner_key, (public_base or "").rstrip("/")

    @classmethod
    def from_env(cls, env=None):
        e = os.environ if env is None else env
        c = cls(e.get("TWILIO_ACCOUNT_SID", ""), e.get("TWILIO_AUTH_TOKEN", ""), e.get("TWILIO_FROM", ""), e.get("OWNER_PHONE", ""),
                e.get("OWNER_KEY", ""), e.get("PUBLIC_BASE_URL", ""))
        ok = all([c.sid, c.token, c.from_number, c.owner_phone, len(c.owner_key) >= 16, c.public_base.startswith("https://")])
        return c if ok else None


def install_call_routes(app, ask, config=None, transport=None, now=time.time):
    """ask(text, history_pairs, session_id, request) -> answer text (the web chat's pipeline)."""
    cfg = config if config is not None else CallConfig.from_env()
    calls, started = {}, []

    def owner_only(request):
        key = request.headers.get("x-owner-key", "")
        if cfg is None or not hmac.compare_digest(key.encode(), cfg.owner_key.encode()):
            raise HTTPException(403, "Not allowed.")

    def forget_old():
        for k in [k for k, v in calls.items() if now() - v["created"] > 3600]:
            calls.pop(k, None)

    @app.get("/api/call/status")
    def call_status():
        return {"enabled": cfg is not None}

    @app.post("/api/call/start")
    def call_start(request: Request):
        owner_only(request)
        forget_old()
        started[:] = [t for t in started if now() - t < 86400]
        if len(started) >= CALLS_PER_DAY:
            raise HTTPException(429, "Too many calls today.")
        cid = secrets.token_urlsafe(12)
        calls[cid] = {"created": now(), "turns": [], "silences": 0, "done": False}
        try:
            with httpx.Client(timeout=20, transport=transport) as client:
                r = client.post(f"https://api.twilio.com/2010-04-01/Accounts/{cfg.sid}/Calls.json", auth=(cfg.sid, cfg.token),
                                data={"To": cfg.owner_phone, "From": cfg.from_number, "Url": f"{cfg.public_base}/api/call/voice?cid={cid}", "Method": "POST"})
        except httpx.HTTPError:
            calls.pop(cid, None)
            raise HTTPException(502, "The phone service could not be reached.")
        if r.status_code >= 300:
            calls.pop(cid, None)
            raise HTTPException(502, "The phone service did not start the call.")
        started.append(now())
        return {"ok": True, "call_id": cid}

    @app.get("/api/call/transcript")
    def call_transcript(request: Request, cid: str, after: int = 0):
        owner_only(request)
        c = calls.get(cid)
        if c is None:
            raise HTTPException(404, "No such call.")
        return {"turns": [list(t) for t in c["turns"][max(0, after):]], "next": len(c["turns"]), "done": c["done"]}

    @app.post("/api/call/voice")
    async def call_voice(request: Request, cid: str = ""):
        if cfg is None:
            raise HTTPException(404, "Not found.")
        form = {k: v[0] for k, v in parse_qs((await request.body()).decode("utf-8", "replace"), keep_blank_values=True).items()}
        url = cfg.public_base + request.url.path + ("?" + request.url.query if request.url.query else "")
        if not twiml.signature_ok(cfg.token, url, form, request.headers.get("x-twilio-signature", "")):
            raise HTTPException(403, "Bad signature.")
        c = calls.get(cid)
        if c is None:
            return Response(twiml.hangup("Sorry, this call is no longer active. Goodbye."), media_type="text/xml")
        again = f"{cfg.public_base}/api/call/voice?cid={cid}"
        heard = (form.get("SpeechResult") or "").strip()
        users = sum(1 for r, _ in c["turns"] if r == "user")
        if not heard:
            if not c["turns"] and not c["silences"]:                       # the first time: greet and listen
                c["turns"].append(("bot", GREETING))
                return Response(twiml.response(twiml.gather(again, GREETING)), media_type="text/xml")
            c["silences"] += 1
            if c["silences"] > MAX_SILENCES:
                c["done"] = True
                return Response(twiml.hangup("I did not hear anything, so I will end the call. Open a new chat any time and I will call again. Goodbye."), media_type="text/xml")
            return Response(twiml.response(twiml.gather(again, "I am listening. What would you like to know?")), media_type="text/xml")
        c["silences"] = 0
        c["turns"].append(("user", heard))
        if twiml.GOODBYE.search(heard) or users + 1 >= MAX_TURNS:
            c["turns"].append(("bot", "Goodbye!"))
            c["done"] = True
            return Response(twiml.hangup("Goodbye! Talk to you soon."), media_type="text/xml")
        history, turns = [], c["turns"][:-1]
        for i in range(len(turns) - 1):
            if turns[i][0] == "user" and turns[i + 1][0] == "bot":
                history.append((turns[i][1], turns[i + 1][1]))
        try:
            answer = ask(heard, history[-2:], "call-" + cid, request) or "Sorry, I have no answer for that."
        except Exception:
            answer = "Sorry, something went wrong. Please ask again."
        c["turns"].append(("bot", answer))
        return Response(twiml.response(twiml.gather(again, answer)), media_type="text/xml")

    return cfg
