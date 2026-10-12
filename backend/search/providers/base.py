"""The small interface every search provider implements.

Chat orchestration only ever talks to this interface, so a provider can be
swapped without touching the chat code.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


class SearchError(Exception):
    """Any provider failure. The message is safe to log (no query text, no results)."""


class SearchTimeout(SearchError):
    pass


class SearchRateLimited(SearchError):
    pass


class SearchUnavailable(SearchError):
    """The search server cannot be reached at all (not running, refused, DNS failure)."""


@dataclass
class RawResult:
    """One result exactly as the provider sent it (NOT yet sanitised)."""
    title: str
    url: str
    snippet: str


class SearchProvider(ABC):
    name = "base"

    @abstractmethod
    def is_configured(self) -> bool:
        """True when everything needed (for example an API key) is present."""

    @abstractmethod
    def search(self, query: str, count: int, timeout: float) -> List[RawResult]:
        """Return up to `count` results or raise SearchError (SearchTimeout / SearchRateLimited)."""
