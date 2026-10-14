"""Cache / memory store on Redis (the third store, next to Neo4j and Chroma).

Two jobs:
  * per-user "latest" memory: the last few turns and the people / cases just discussed, so a follow-up such as
    "what about his bail?" can be resolved to the right entity;
  * a short-lived retrieval cache, keyed by the question and a data version. Loading any new document bumps the version,
    which invalidates every cached retrieval at once (stale answers can never outlive new data).

Privacy: users are identified only by a hash (never the raw address); conversations expire after SESSION_TTL
(default 7 days) and `forget()` deletes one user's memory immediately.
Reliability: every method swallows Redis errors and returns a neutral value. If Redis is down, the chat simply has no memory.

Connection: REDIS_URL (e.g. redis://:password@localhost:6379/0); unset = disabled.
"""
import hashlib
import json
import logging
import os
import re

log = logging.getLogger("elvarn.memory")
SESSION_TTL = int(os.environ.get("MEMORY_SESSION_TTL_SECONDS", 7 * 24 * 3600))
CACHE_TTL = int(os.environ.get("MEMORY_CACHE_TTL_SECONDS", 3600))
MAX_TURNS = 6
MAX_TEXT = 600


def session_key(raw):
    """Stable, anonymous id for a visitor or an explicit session id."""
    return hashlib.sha256(("elvarn|" + str(raw)).encode("utf-8")).hexdigest()[:20]


def normalise(question):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", question.lower())).strip()


class Memory:
    def __init__(self, client, session_ttl=SESSION_TTL, cache_ttl=CACHE_TTL):
        self.r, self.session_ttl, self.cache_ttl = client, session_ttl, cache_ttl

    @classmethod
    def from_env(cls):
        url = os.environ.get("REDIS_URL")
        if not url:
            return None
        import redis
        return cls(redis.Redis.from_url(url, decode_responses=True, socket_timeout=1.5, socket_connect_timeout=1.5))

    def _safe(self, fn, default=None):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - memory must never break the chat
            log.warning("memory unavailable: %s", exc.__class__.__name__)
            return default

    def ping(self):
        return bool(self._safe(lambda: self.r.ping(), False))

    # ---- the user's latest context --------------------------------------------------------------
    def recall(self, session):
        """{"turns": [{"q", "a"}] oldest first, "entities": [{"id", "type", "name"}] most recent first}."""
        def work():
            raw = self.r.lrange(f"mem:{session}:turns", 0, MAX_TURNS - 1)
            ents = self.r.get(f"mem:{session}:entities")
            return {"turns": [json.loads(t) for t in reversed(raw)], "entities": json.loads(ents) if ents else []}
        return self._safe(work, {"turns": [], "entities": []})

    def remember(self, session, question, answer, entities=()):
        """Store this turn. `entities` are the people/cases it was about; they become the most recent ones."""
        def work():
            p = self.r.pipeline()
            p.lpush(f"mem:{session}:turns", json.dumps({"q": question[:MAX_TEXT], "a": (answer or "")[:MAX_TEXT]}, ensure_ascii=False))
            p.ltrim(f"mem:{session}:turns", 0, MAX_TURNS - 1)
            p.expire(f"mem:{session}:turns", self.session_ttl)
            if entities:
                old = self.recall(session)["entities"]
                merged = {e["id"]: e for e in [*[{"id": e["id"], "type": e["type"], "name": e["name"]} for e in entities], *old]}
                p.set(f"mem:{session}:entities", json.dumps(list(merged.values())[:6], ensure_ascii=False), ex=self.session_ttl)
            p.execute()
        self._safe(work)

    def forget(self, session):
        self._safe(lambda: self.r.delete(f"mem:{session}:turns", f"mem:{session}:entities"))

    # ---- retrieval cache ------------------------------------------------------------------------
    def version(self):
        return int(self._safe(lambda: self.r.get("kb:version"), 0) or 0)

    def bump_version(self):
        """Called after any new data is stored: every cached retrieval becomes unreachable."""
        return self._safe(lambda: self.r.incr("kb:version"), None)

    def _ckey(self, question, entity_ids):
        digest = hashlib.sha1((normalise(question) + "|" + ",".join(sorted(entity_ids))).encode("utf-8")).hexdigest()
        return f"ctx:{self.version()}:{digest}"

    def get_context(self, question, entity_ids=()):
        raw = self._safe(lambda: self.r.get(self._ckey(question, entity_ids)))
        return json.loads(raw) if raw else None

    def put_context(self, question, entity_ids, payload):
        self._safe(lambda: self.r.set(self._ckey(question, entity_ids), json.dumps(payload, ensure_ascii=False), ex=self.cache_ttl))
