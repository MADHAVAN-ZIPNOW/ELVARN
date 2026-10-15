"""ELVARN phoning the owner (Twilio): who may start a call, what is said, signature checks, and the transcript the web chat reads. No network."""
import base64
import hashlib
import hmac

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from telephony import twiml
from telephony.routes import CallConfig, install_call_routes

BASE = "https://elvarn.example.com"
CFG = CallConfig("ACtest", "twilio-auth-token", "+15005550006", "+919000000000", "owner-key-0123456789", BASE)
KEY = {"X-Owner-Key": "owner-key-0123456789"}


def sign(url, params):
    data = url + "".join(k + str(params[k]) for k in sorted(params))
    return base64.b64encode(hmac.new(CFG.token.encode(), data.encode(), hashlib.sha1).digest()).decode()


def make(ask=None, twilio=None):
    calls = []

    def handler(request):
        calls.append(request)
        return twilio(request) if twilio else httpx.Response(201, json={"sid": "CA1"})
    asked = []

    def fake_ask(text, history, session, request):
        asked.append((text, history, session))
        return ask(text) if ask else f"The answer to {text}."
    app = FastAPI()
    install_call_routes(app, fake_ask, CFG, transport=httpx.MockTransport(handler))
    return TestClient(app), calls, asked


def voice(client, cid, speech=None):
    params = {"CallSid": "CA1"}
    if speech is not None:
        params["SpeechResult"] = speech
    url = f"{BASE}/api/call/voice?cid={cid}"
    body = "&".join(f"{k}={v}" for k, v in params.items()).replace(" ", "+")
    return client.post(f"/api/call/voice?cid={cid}", content=body, headers={"X-Twilio-Signature": sign(url, params), "Content-Type": "application/x-www-form-urlencoded"})


def test_the_signature_check_matches_twilios_own_published_example():
    params = {"CallSid": "CA1234567890ABCDE", "Caller": "+14158675310", "Digits": "1234", "From": "+14158675310", "To": "+18005551212"}
    assert twiml.signature_ok("12345", "https://mycompany.com/myapp.php?foo=1&bar=2", params, "GvWf1cFY/Q7PnoempGyD5oXAezc=")
    assert not twiml.signature_ok("12345", "https://mycompany.com/myapp.php?foo=1&bar=2", params, "wrong")
    assert not twiml.signature_ok("", "u", {}, "x")


def test_the_feature_is_off_until_everything_is_set():
    full = {"TWILIO_ACCOUNT_SID": "AC1", "TWILIO_AUTH_TOKEN": "t", "TWILIO_FROM": "+1500", "OWNER_PHONE": "+91900", "OWNER_KEY": "k" * 16, "PUBLIC_BASE_URL": BASE}
    assert CallConfig.from_env(full) is not None
    for missing in full:
        assert CallConfig.from_env({k: v for k, v in full.items() if k != missing}) is None
    assert CallConfig.from_env({**full, "OWNER_KEY": "short"}) is None and CallConfig.from_env({**full, "PUBLIC_BASE_URL": "http://x"}) is None
    app = FastAPI()
    install_call_routes(app, lambda *a: "", None)
    assert TestClient(app).get("/api/call/status").json() == {"enabled": False}


def test_only_the_owner_can_start_a_call_and_only_to_the_owners_number():
    client, calls, _ = make()
    assert client.post("/api/call/start").status_code == 403
    assert client.post("/api/call/start", headers={"X-Owner-Key": "guess"}).status_code == 403
    assert not calls
    r = client.post("/api/call/start", headers=KEY)
    assert r.status_code == 200 and r.json()["ok"] is True
    sent = calls[0]
    body = sent.content.decode()
    assert "To=%2B919000000000" in body and "From=%2B15005550006" in body and "api%2Fcall%2Fvoice%3Fcid%3D" + r.json()["call_id"] in body
    assert sent.url.path == "/2010-04-01/Accounts/ACtest/Calls.json" and sent.headers["authorization"].startswith("Basic ")


def test_no_more_than_a_dozen_calls_a_day_and_a_failed_phone_service_is_reported():
    client, _, _ = make()
    codes = [client.post("/api/call/start", headers=KEY).status_code for _ in range(14)]
    assert codes[:12] == [200] * 12 and codes[12:] == [429, 429]
    bad, _, _ = make(twilio=lambda r: httpx.Response(401, json={"message": "Authenticate"}))
    r = bad.post("/api/call/start", headers=KEY)
    assert r.status_code == 502 and "Authenticate" not in r.text


def test_a_call_greets_listens_answers_with_memory_and_says_goodbye():
    client, _, asked = make()
    cid = client.post("/api/call/start", headers=KEY).json()["call_id"]
    first = voice(client, cid).text
    assert "Hi, this is ELVARN" in first and "<Gather" in first and 'input="speech"' in first and "Polly.Aditi" in first
    second = voice(client, cid, "tell me about the DVAC").text
    assert "The answer to tell me about the DVAC." in second and "<Gather" in second and asked[0][2] == "call-" + cid and asked[0][1] == []
    third = voice(client, cid, "what did it do in 2018").text
    assert asked[1][1] == [("tell me about the DVAC", "The answer to tell me about the DVAC.")]               # the next turn sees the one before
    bye = voice(client, cid, "ok bye").text
    assert "<Hangup/>" in bye
    t = client.get(f"/api/call/transcript?cid={cid}", headers=KEY).json()
    assert t["done"] is True and [x[0] for x in t["turns"]][:3] == ["bot", "user", "bot"] and t["turns"][-1] == ["bot", "Goodbye!"]
    assert client.get(f"/api/call/transcript?cid={cid}&after={t['next']}", headers=KEY).json()["turns"] == []
    assert client.get(f"/api/call/transcript?cid={cid}").status_code == 403


def test_forged_requests_are_refused_and_answers_are_made_safe_for_xml():
    client, _, _ = make(ask=lambda q: "R&D <b>bribe</b> [1] see https://x.example.com/a?b=1 done")
    cid = client.post("/api/call/start", headers=KEY).json()["call_id"]
    assert client.post(f"/api/call/voice?cid={cid}", content="CallSid=CA1", headers={"X-Twilio-Signature": "forged"}).status_code == 403
    assert client.post(f"/api/call/voice?cid={cid}", content="CallSid=CA1").status_code == 403
    voice(client, cid)
    out = voice(client, cid, "hello").text
    assert "R&amp;D &lt;b&gt;bribe&lt;/b&gt;" in out and "https://x.example.com" not in out and "[1]" not in out


def test_silence_ends_the_call_politely_and_an_unknown_call_hangs_up():
    client, _, _ = make()
    cid = client.post("/api/call/start", headers=KEY).json()["call_id"]
    voice(client, cid)
    assert "<Gather" in voice(client, cid).text and "<Gather" in voice(client, cid).text
    assert "<Hangup/>" in voice(client, cid).text
    assert "<Hangup/>" in voice(client, "no-such-call", "hi").text
