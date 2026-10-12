"""Search providers. Add a new one by subclassing SearchProvider and returning it from get_provider()."""
import logging
import os

from .base import RawResult, SearchError, SearchProvider, SearchRateLimited, SearchTimeout, SearchUnavailable  # noqa: F401
from .searxng import DEFAULT_BASE_URL, SearxngProvider, validate_base_url  # noqa: F401

log = logging.getLogger("elvarn.search")


def get_provider(environ=None):
    """The self-hosted SearXNG provider, using SEARXNG_BASE_URL (default http://127.0.0.1:8080).

    The address is operator configuration, checked once here. If it is not a valid http(s) address
    the provider is disabled (returns None) and chat keeps working without web search.
    """
    env = os.environ if environ is None else environ
    try:
        return SearxngProvider(env.get("SEARXNG_BASE_URL", "") or DEFAULT_BASE_URL,
                               auth_token=env.get("SEARXNG_AUTH_TOKEN", ""))
    except ValueError as e:
        log.warning("web search disabled: SEARXNG_BASE_URL / SEARXNG_AUTH_TOKEN setting is not usable (%s)", e)
        return None
