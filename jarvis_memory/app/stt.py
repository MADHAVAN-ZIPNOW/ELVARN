"""Speech-to-text for voice chat from the phone: the recording is sent to Groq's Whisper (fast and accurate), never stored.

The phone only uploads a short compressed clip (a few seconds of AAC) after it has detected speech, so this is a few KB per turn.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

log = logging.getLogger("jarvis_memory.stt")
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
STT_MODEL = "whisper-large-v3-turbo"
EXTENSIONS = {"audio/mp4": "m4a", "audio/m4a": "m4a", "audio/x-m4a": "m4a", "audio/aac": "aac", "audio/wav": "wav", "audio/x-wav": "wav",
              "audio/webm": "webm", "audio/ogg": "ogg", "audio/mpeg": "mp3"}
MAX_AUDIO_BYTES = 240_000


class SttError(Exception):
    """kind: unsupported | unavailable | rate_limit | not_configured (the route turns it into a clear message)."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


def transcribe(settings, data: bytes, content_type: str, client: Optional[httpx.Client] = None) -> str:
    """The text of a short spoken clip ('' when no words were heard)."""
    ctype = content_type.split(";")[0].strip().lower()
    ext = EXTENSIONS.get(ctype)
    if ext is None:
        raise SttError("unsupported")
    if not settings.groq_api_key:
        raise SttError("not_configured")
    owns = client is None
    c = client or httpx.Client(timeout=25.0)
    try:
        resp = c.post(GROQ_STT_URL, headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                      files={"file": (f"speech.{ext}", data, ctype)},
                      data={"model": STT_MODEL, "language": settings.stt_language, "temperature": "0", "response_format": "json",
                            "prompt": "Jarvis. Madhavan. ELVARN."})   # a hint so the name and the project are spelled right
    except httpx.HTTPError as exc:
        log.warning("Speech recognition unreachable (%s)", type(exc).__name__)
        raise SttError("unavailable") from None
    finally:
        if owns:
            c.close()
    if resp.status_code == 429:
        raise SttError("rate_limit")
    if resp.status_code >= 400:
        log.warning("Speech recognition refused the clip (HTTP %s)", resp.status_code)
        raise SttError("unavailable")
    try:
        return " ".join(str(resp.json().get("text", "")).split())
    except ValueError:
        raise SttError("unavailable") from None
