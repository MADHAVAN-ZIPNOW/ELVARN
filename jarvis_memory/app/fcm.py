"""Firebase Cloud Messaging (HTTP v1) with a service-account key, using only httpx + PyJWT (no heavy SDK).

The key arrives as the FIREBASE_SERVICE_ACCOUNT_JSON environment variable (Render). It is never written to disk or logged.
Messages are DATA-only and high priority, so the phone app wakes by itself and rings like a call.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import httpx
import jwt

log = logging.getLogger("jarvis_memory.fcm")
SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
TOKEN_URL = "https://oauth2.googleapis.com/token"


class Pusher:
    def __init__(self, service_account_json: str, client: Optional[httpx.Client] = None) -> None:
        info = json.loads(service_account_json)
        self._email, self._key, self._project = info["client_email"], info["private_key"], info["project_id"]
        self._client = client or httpx.Client(timeout=15.0)
        self._access: Optional[str] = None
        self._expires = 0.0

    @property
    def project_id(self) -> str:
        return self._project

    def _token(self) -> str:
        if self._access and time.time() < self._expires - 60:
            return self._access
        now = int(time.time())
        assertion = jwt.encode({"iss": self._email, "scope": SCOPE, "aud": TOKEN_URL, "iat": now, "exp": now + 3600},
                               self._key, algorithm="RS256")
        resp = self._client.post(TOKEN_URL, data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion})
        resp.raise_for_status()
        data = resp.json()
        self._access, self._expires = data["access_token"], time.time() + int(data.get("expires_in", 3600))
        return self._access

    def send(self, device_token: str, data: dict[str, Any], ttl_seconds: int = 300) -> str:
        """Returns "ok", "gone" (the app was uninstalled: forget the token) or "error"."""
        body = {"message": {"token": device_token, "data": {k: str(v) for k, v in data.items()},
                            "android": {"priority": "HIGH", "ttl": f"{ttl_seconds}s"}}}
        try:
            resp = self._client.post(f"https://fcm.googleapis.com/v1/projects/{self._project}/messages:send", json=body,
                                     headers={"Authorization": f"Bearer {self._token()}"})
        except httpx.HTTPError as exc:
            log.warning("FCM transport error: %s", type(exc).__name__)
            return "error"
        if resp.status_code == 200:
            return "ok"
        if resp.status_code in (404, 400) and any(m in resp.text for m in ("UNREGISTERED", "NOT_FOUND", "INVALID_ARGUMENT")):
            return "gone"
        log.warning("FCM refused the message (HTTP %s)", resp.status_code)
        return "error"


def from_settings(settings) -> Optional[Pusher]:
    if not settings.firebase_service_account_json:
        return None
    try:
        return Pusher(settings.firebase_service_account_json)
    except Exception as exc:  # malformed key: report the type only, never its content
        log.error("FIREBASE_SERVICE_ACCOUNT_JSON is not usable (%s)", type(exc).__name__)
        return None
