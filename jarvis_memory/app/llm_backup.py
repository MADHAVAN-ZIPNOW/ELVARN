"""One place for "Groq first, Gemini if Groq is busy or down" for the jobs that call the AI directly (tagging, wording, review, Q&A)."""
from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger("jarvis.llm")
RETRY_STATUSES = (429, 500, 502, 503, 504)


def llm_post(settings, client: httpx.Client, payload: dict[str, Any]) -> httpx.Response:
    """POST a chat request to Groq. If Groq is rate limited, down or unreachable and a Gemini key is set, ask Gemini instead."""
    resp = None
    try:
        resp = client.post(settings.llm_base_url.rstrip("/") + "/chat/completions", json=payload,
                           headers={"Authorization": f"Bearer {settings.groq_api_key}"})
        if resp.status_code not in RETRY_STATUSES:
            return resp
    except httpx.HTTPError:
        resp = None
    if getattr(settings, "gemini_api_key", ""):
        alt = {k: v for k, v in payload.items() if k != "reasoning_effort"}
        alt["model"] = settings.gemini_model
        try:
            second = client.post(settings.gemini_base_url.rstrip("/") + "/chat/completions", json=alt,
                                 headers={"Authorization": f"Bearer {settings.gemini_api_key}"})
            if second.status_code < 400:
                log.info("Groq was busy; Gemini answered")
                return second
        except httpx.HTTPError:
            pass
    if resp is not None:
        return resp
    raise httpx.ConnectError("no AI service reachable")
