"""Twilio voice helpers: the XML ("TwiML") a call is driven with, and the check that a request really came from Twilio.

A phone call works like this: ELVARN asks Twilio to call the owner's number. When the owner picks up, Twilio asks our /api/call/voice
address what to do; we answer with TwiML: say something, then <Gather input="speech"> to listen. Twilio turns the caller's speech into
text and posts it back to the same address, we find an answer, and so on until the owner says goodbye or hangs up.
"""
import base64
import hashlib
import hmac
import re
from xml.sax.saxutils import escape, quoteattr

VOICE = "Polly.Aditi"          # an English (India) voice
LANGUAGE = "en-IN"
HINTS = "DVAC, CBI, Tamil Nadu, High Court, Supreme Court, bribe, corruption, ELVARN"
GOODBYE = re.compile(r"\b(bye|goodbye|good bye|that is all|thats all|stop|hang up|no thanks)\b|that's all", re.I)


def speakable(text, limit=450):
    """Plain words to read aloud: no [n] marks, no links, not too long."""
    t = re.sub(r"\[\d+\]", "", str(text or ""))
    t = re.sub(r"https?://\S+", "", t)
    return re.sub(r"\s+", " ", t).strip()[:limit]


def say(text):
    return f'<Say voice="{VOICE}" language="{LANGUAGE}">{escape(speakable(text))}</Say>'


def gather(next_url, prompt=None):
    """Listen for speech. The prompt is spoken INSIDE the Gather, so the caller can interrupt it by talking."""
    inner = say(prompt) if prompt else ""
    return (f'<Gather input="speech" language="{LANGUAGE}" speechTimeout="auto" timeout="6" hints={quoteattr(HINTS)} '
            f'action={quoteattr(next_url)} method="POST">{inner}</Gather>')


def response(*parts):
    return '<?xml version="1.0" encoding="UTF-8"?><Response>' + "".join(parts) + "</Response>"


def hangup(message):
    return response(say(message), "<Hangup/>")


def signature_ok(auth_token, url, params, header):
    """Twilio signs url + (sorted form fields, name then value) with HMAC-SHA1 and the account's auth token."""
    if not (auth_token and header):
        return False
    data = url + "".join(k + str(params[k]) for k in sorted(params))
    expected = base64.b64encode(hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()).decode()
    return hmac.compare_digest(expected, header)
