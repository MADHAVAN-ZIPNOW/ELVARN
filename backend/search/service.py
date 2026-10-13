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


# Words that mark adult content in a result's address, title or snippet. A match drops the result completely.
_ADULT = re.compile(r"(porn|xxx|hentai|nsfw|erotic|escort|onlyfans|camgirl|nude|nudes|sex ?cam|sexcam|adult video|fetish|xvideos|xnxx|redtube|youporn|pornhub|eporner|brazzers)", re.I)


def is_adult(*texts):
    return any(_ADULT.search(t or "") for t in texts)


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
    # The distinctive word of the question must appear: a capitalised word ("Collegium", "Paris") or, for a very short
    # lowercase question, its longest word. "What is the Collegium system?" must not return a page about Linux
    # just because it also contains the common word "system".
    anchors = {_stem(w.lower()) for w in re.findall(r"[A-Z][a-z]{3,}", query) if w.lower() in wanted}
    if not anchors and len(want) <= 2 and wanted:
        anchors = {_stem(max(sorted(wanted), key=len))}
    if anchors and not (anchors & got):
        return False
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
    def __init__(self, provider: SearchProvider = None, max_results=5, timeout=5.0, max_snippet=220, retry_delay=0.8):
        self.provider = provider
        self.retry_delay = max(0.0, min(float(retry_delay), 3.0))      # pause before the single retry when a search returns nothing
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

    def search(self, query, core=None):
        started = time.perf_counter()
        status, sources = self._run(query, core)
        if status == NO_RESULTS and core and core.strip() and core.strip() != (query or "").strip():
            # The wording we added ("... under Indian law") sometimes sends the search engines the wrong way: try the plain question once.
            status, sources = self._run(core, core)
        ms = int((time.perf_counter() - started) * 1000)
        # minimal metadata only: never the query text, results or any key
        log.info("web_search status=%s results=%d ms=%d", status, len(sources), ms)
        return SearchOutcome(status, sources, MESSAGES[status], ms)

    def _run(self, query, core=None):
        query = (query or "").strip()[:MAX_QUERY_CHARS]
        if not query:
            return SKIPPED, []
        if not self.available:
            return UNAVAILABLE, []
        try:
            raw = self.provider.search(query, self.max_results, self.timeout)
            if not raw:                                  # the search engines behind SearXNG are flaky from cloud servers: try once more
                time.sleep(self.retry_delay)
                raw = self.provider.search(query, self.max_results, self.timeout)
        except SearchTimeout:
            return ERROR, []
        except (SearchRateLimited, SearchUnavailable):
            return UNAVAILABLE, []
        except SearchError:
            return ERROR, []
        except Exception:  # a provider bug must never break chat
            return ERROR, []
        base = (core or query)                  # judge relevance by the user's own words, not by context we added ("under Indian law")
        wanted = keywords(base)
        sources, seen, per_domain = [], set(), {}
        for item in list(raw or [])[: self.max_results * 2]:
            checked = clean_url(getattr(item, "url", ""))
            title = clean_text(getattr(item, "title", ""), 120)
            snippet = clean_text(getattr(item, "snippet", ""), self.max_snippet)
            if not checked or not (title or snippet) or checked[0] in seen or per_domain.get(checked[1], 0) >= 2:
                continue  # unsafe URL, empty, duplicate, or a third hit from one site (spam)
            if is_adult(checked[0], title, snippet):
                continue  # adult content: never shown, whatever the search engines return
            if wanted and not _relevant(base, wanted, f"{title} {snippet} {checked[1]}"):
                continue  # too little in common with the question (or a number from the question is missing): not relevant evidence
            seen.add(checked[0])
            per_domain[checked[1]] = per_domain.get(checked[1], 0) + 1
            sources.append(Source(title or checked[1], checked[0], checked[1], snippet))
            if len(sources) >= self.max_results:
                break
        return (USED, sources) if sources else (NO_RESULTS, [])
