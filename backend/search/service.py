"""The search service: calls a provider, cleans the results, never raises into the chat flow."""
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import List

from .providers import SearchError, SearchProvider, SearchRateLimited, SearchTimeout, SearchUnavailable, get_provider
from .router import MAX_QUERY_CHARS, keywords
from .sanitize import clean_text, clean_url

log = logging.getLogger("elvarn.search")

# search_status values (machine readable)
USED, SKIPPED, UNAVAILABLE, NO_RESULTS, ERROR = "used", "skipped", "unavailable", "no_results", "error"

MESSAGES = {
    SKIPPED: "",
    UNAVAILABLE: "Live web search is not available right now, so this answer was not checked against live web results.",
    NO_RESULTS: "The web search found nothing that clearly answers this, so this answer was not checked against live web results.",
    ERROR: "The web search did not work this time, so this answer was not checked against live web results.",
    USED: "Answer written from live web search results. The small model can still be wrong; check the sources.",
}
# Live results were found, but none fitted into the model's small context window: be honest that the text was NOT written from them.
USED_NO_EVIDENCE_MESSAGE = ("Live web results were found (see the sources), but they did not fit into the small model's memory window, "
                            "so the answer text was NOT written from them. Read the sources instead.")


def _stem(word):
    """Very rough word stem so 'India' matches 'Indian' and 'article' matches 'articles'."""
    return word[:5]


def _relevant(query, wanted, result_text):
    """A result is kept only if it shares enough words with the question AND contains every number in it.

    Numbers matter most for legal questions ("Article 21", "Section 420"): a page about Article 246 is not an answer,
    even though it shares the word "article". Words shorter than 3 letters are ignored by keywords(), so numbers are checked here.
    """
    numbers = re.findall(r"\b\d+\b", query)
    if numbers:
        have = set(re.findall(r"\b\d+\b", result_text))
        if not all(n in have for n in numbers):
            return False
    want = {_stem(w) for w in wanted}
    got = {_stem(w) for w in keywords(result_text)}
    need = 1 if len(want) <= 2 else 2                        # one shared word is enough for a very short question; otherwise two
    return len(want & got) >= need


@dataclass
class Source:
    title: str
    url: str
    domain: str
    snippet: str

    def as_dict(self):
        return {"title": self.title, "url": self.url, "domain": self.domain, "snippet": self.snippet}


@dataclass
class SearchOutcome:
    status: str
    sources: List[Source] = field(default_factory=list)
    message: str = ""
    duration_ms: int = 0

    @property
    def searched(self):  # True only when a search really ran AND gave usable results
        return self.status == USED


def _number(env, name, default, cast):
    """A numeric setting; a missing or broken value falls back to the default (the service clamps the range)."""
    try:
        return cast(str(env.get(name, "")).strip() or default)
    except ValueError:
        return default


class SearchService:
    def __init__(self, provider: SearchProvider = None, max_results=5, timeout=5.0, max_snippet=220):
        self.provider = provider
        self.max_results = max(1, min(int(max_results), 8))
        self.timeout = max(0.5, min(float(timeout), 15.0))
        self.max_snippet = max(40, min(int(max_snippet), 600))

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        return cls(get_provider(environ=env),
                   max_results=_number(env, "SEARCH_RESULT_LIMIT", 5, int),
                   timeout=_number(env, "SEARXNG_TIMEOUT_SECONDS", 5.0, float),
                   max_snippet=_number(env, "MAX_SNIPPET_LENGTH", 220, int))

    @property
    def available(self):
        return self.provider is not None and self.provider.is_configured()

    def search(self, query):
        started = time.perf_counter()
        status, sources = self._run(query)
        ms = int((time.perf_counter() - started) * 1000)
        # minimal metadata only: never the query text, results or any key
        log.info("web_search status=%s results=%d ms=%d", status, len(sources), ms)
        return SearchOutcome(status, sources, MESSAGES[status], ms)

    def _run(self, query):
        query = (query or "").strip()[:MAX_QUERY_CHARS]
        if not query:
            return SKIPPED, []
        if not self.available:
            return UNAVAILABLE, []
        try:
            raw = self.provider.search(query, self.max_results, self.timeout)
        except SearchTimeout:
            return ERROR, []
        except (SearchRateLimited, SearchUnavailable):
            return UNAVAILABLE, []
        except SearchError:
            return ERROR, []
        except Exception:  # a provider bug must never break chat
            return ERROR, []
        wanted = keywords(query)
        sources, seen, per_domain = [], set(), {}
        for item in list(raw or [])[: self.max_results * 2]:
            checked = clean_url(getattr(item, "url", ""))
            title = clean_text(getattr(item, "title", ""), 120)
            snippet = clean_text(getattr(item, "snippet", ""), self.max_snippet)
            if not checked or not (title or snippet) or checked[0] in seen or per_domain.get(checked[1], 0) >= 2:
                continue  # unsafe URL, empty, duplicate, or a third hit from one site (spam)
            if wanted and not _relevant(query, wanted, f"{title} {snippet} {checked[1]}"):
                continue  # too little in common with the question (or a number from the question is missing): not relevant evidence
            seen.add(checked[0])
            per_domain[checked[1]] = per_domain.get(checked[1], 0) + 1
            sources.append(Source(title or checked[1], checked[0], checked[1], snippet))
            if len(sources) >= self.max_results:
                break
        return (USED, sources) if sources else (NO_RESULTS, [])
