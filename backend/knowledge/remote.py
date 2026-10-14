"""Client for the hosted knowledge service (knowledge_service/): the chat site asks it for the GraphRAG context over HTTPS.

    KNOWLEDGE_URL     https://<your-knowledge-service>.onrender.com
    KNOWLEDGE_TOKEN   exactly the value of KNOWLEDGE_TOKEN on the service

Same rules as the search client: the token is only ever sent over https:// (or to localhost); with plain http to another host the
client refuses to start. Every failure (service asleep, token wrong, store down, too slow) becomes KnowledgeUnavailable, which the
chat turns into an honest message; it never lets a corruption question be guessed by the model.
Only the standard `httpx` package is needed, so the chat site stays small (no neo4j / chromadb / redis packages).
"""
import os
from urllib.parse import urlsplit

import httpx

from backend.knowledge.graphrag import Context

LOOPBACK = ("localhost", "127.0.0.1", "::1")


class KnowledgeUnavailable(Exception):
    """The knowledge service could not give a context (the message is safe to log: no question text, no token)."""


class RemoteKnowledge:
    def __init__(self, base_url, token, timeout=15.0, transport=None):
        parts = urlsplit((base_url or "").strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("KNOWLEDGE_URL must be an http:// or https:// address")
        if len(token or "") < 24:
            raise ValueError("KNOWLEDGE_TOKEN is missing or shorter than 24 characters")
        if parts.scheme != "https" and parts.hostname not in LOOPBACK:
            raise ValueError("KNOWLEDGE_TOKEN would be sent over plain http; use an https:// KNOWLEDGE_URL")
        self.base_url = f"{parts.scheme}://{parts.netloc}"
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport, follow_redirects=False,
                                 headers={"Authorization": f"Bearer {token}", "User-Agent": "ELVARN-site/1.0"})

    @classmethod
    def from_env(cls):
        """The configured client, or None when KNOWLEDGE_URL is not set. A bad configuration raises ValueError."""
        url = os.environ.get("KNOWLEDGE_URL", "").strip()
        return cls(url, os.environ.get("KNOWLEDGE_TOKEN", "")) if url else None

    def _call(self, method, path, **kw):
        try:
            r = self.http.request(method, path, **kw)
        except httpx.TimeoutException:
            raise KnowledgeUnavailable("the knowledge service did not answer in time (it may be waking up)")
        except httpx.HTTPError as exc:
            raise KnowledgeUnavailable(f"the knowledge service could not be reached ({exc.__class__.__name__})")
        if r.status_code in (401, 403):
            raise KnowledgeUnavailable("the knowledge service refused the token")
        if r.status_code != 200:
            raise KnowledgeUnavailable(f"the knowledge service answered {r.status_code}")
        try:
            return r.json()
        except ValueError:
            raise KnowledgeUnavailable("the knowledge service sent an unreadable answer")

    def context(self, question, session=None):
        """-> (Context, populated). `populated` says whether the stores hold any documents at all (used to refuse unanswerable questions)."""
        body = {"question": question[:1000]}
        if session:
            body["session"] = session
        data = self._call("POST", "/v1/context", json=body)
        try:
            return Context.from_dict(data), bool(data.get("populated"))
        except (KeyError, TypeError):
            raise KnowledgeUnavailable("the knowledge service sent an unexpected answer")

    def remember(self, session, question, answer, entities):
        try:
            self._call("POST", "/v1/remember", json={"session": session, "question": question[:1000], "answer": (answer or "")[:4000],
                                                     "entities": [{"id": e["id"], "type": e["type"], "name": e["name"]} for e in entities][:6]})
        except KnowledgeUnavailable:
            pass                                                # memory is a convenience; never fail a chat because of it

    def forget(self, session):
        try:
            self._call("DELETE", "/v1/memory", params={"session": session})
        except KnowledgeUnavailable:
            pass
