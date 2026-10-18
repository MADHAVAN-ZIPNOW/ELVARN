"""Authentication and rate limiting.

Single-owner design: every request must carry the owner key in ``X-Owner-Key`` (the same pattern ELVARN already uses).
It fails CLOSED: if no key (or a short one) is configured the API refuses everything instead of being open.
"""
from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

MIN_KEY_LENGTH = 16


class SlidingWindow:
    """In-memory limiter: at most ``limit`` events per ``window`` seconds per key (one Render instance, so enough)."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic) -> None:
        self.limit, self.window, self._clock = limit, window, clock
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def hit(self, key: str) -> bool:
        """Record an event; False if the key is over its limit."""
        now = self._clock()
        q = self._events[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        if len(self._events) > 5000:  # never grow without bound
            for k in [k for k, v in self._events.items() if not v or now - v[-1] > self.window]:
                self._events.pop(k, None)
        return True

    def blocked(self, key: str) -> bool:
        now = self._clock()
        q = self._events.get(key)
        if not q:
            return False
        while q and now - q[0] > self.window:
            q.popleft()
        return len(q) >= self.limit


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "unknown")[:64]


class Guard:
    """FastAPI dependency: rate limit -> authenticate. Failed logins are throttled much harder than normal use."""

    def __init__(self, settings, per_minute: int) -> None:
        self.settings = settings
        self.requests = SlidingWindow(per_minute)
        self.failures = SlidingWindow(10)  # 10 wrong keys per minute per address, then locked out for the window

    def __call__(self, request: Request) -> None:
        ip = client_ip(request)
        if self.failures.blocked(ip):
            raise HTTPException(429, "Too many failed attempts. Try again in a minute.", headers={"Retry-After": "60"})
        if not self.requests.hit(ip):
            raise HTTPException(429, "Too many requests. Slow down.", headers={"Retry-After": "30"})
        key = self.settings.owner_key
        if len(key) < MIN_KEY_LENGTH:
            raise HTTPException(503, "Jarvis Memory is not configured: set JM_OWNER_KEY (16+ characters).")
        supplied = request.headers.get("x-owner-key", "")
        if not hmac.compare_digest(supplied.encode(), key.encode()):
            self.failures.hit(ip)
            raise HTTPException(401, "Invalid owner key.")
