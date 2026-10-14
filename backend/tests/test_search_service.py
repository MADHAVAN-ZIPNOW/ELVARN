from search.providers.base import RawResult
from search.service import SearchService




def test_an_idle_search_service_gets_a_long_first_timeout_then_a_short_one():
    seen = []

    class P:
        def is_configured(self):
            return True

        def search(self, query, count, timeout):
            seen.append(timeout)
            return [RawResult("Result title", "https://example.org/a", "snippet about world cup winner")]
    svc = SearchService(P(), timeout=5.0, cold_timeout=70.0)
    svc.search("world cup winner")
    svc.search("world cup winner")
    assert seen == [70.0, 5.0]
