"""Cache / memory store (Redis), tested with fakeredis: no server needed."""
import fakeredis
import pytest

from usermemory.store import Memory, normalise, session_key


@pytest.fixture()
def mem():
    return Memory(fakeredis.FakeRedis(decode_responses=True))


class Broken:
    """A Redis client whose every call fails (server down)."""
    def __getattr__(self, name):
        def fail(*a, **k):
            raise ConnectionError("down")
        return fail


def test_session_keys_are_anonymous_and_stable():
    a = session_key("203.0.113.7")
    assert a == session_key("203.0.113.7") and a != session_key("203.0.113.8") and "203" not in a and len(a) == 20


def test_recall_returns_recent_turns_oldest_first_and_caps_them(mem):
    for i in range(9):
        mem.remember("u1", f"question {i}", f"answer {i}")
    turns = mem.recall("u1")["turns"]
    assert [t["q"] for t in turns] == [f"question {i}" for i in range(3, 9)]       # the last 6, oldest first
    assert mem.recall("someone else") == {"turns": [], "entities": []}


def test_latest_entities_come_first_without_duplicates(mem):
    mem.remember("u1", "q1", "a1", entities=[{"id": "Person:a", "type": "Person", "name": "A"}])
    mem.remember("u1", "q2", "a2", entities=[{"id": "Case:x", "type": "Case", "name": "X"}, {"id": "Person:a", "type": "Person", "name": "A"}])
    assert [e["id"] for e in mem.recall("u1")["entities"]] == ["Case:x", "Person:a"]


def test_users_are_isolated_and_forget_deletes(mem):
    mem.remember("u1", "mine", "x", entities=[{"id": "Person:a", "type": "Person", "name": "A"}])
    mem.remember("u2", "yours", "y")
    assert mem.recall("u2")["entities"] == []
    mem.forget("u1")
    assert mem.recall("u1") == {"turns": [], "entities": []} and mem.recall("u2")["turns"][0]["q"] == "yours"


def test_sessions_expire(mem):
    mem.session_ttl = 5
    mem.remember("u1", "q", "a", entities=[{"id": "Person:a", "type": "Person", "name": "A"}])
    assert 0 < mem.r.ttl("mem:u1:turns") <= 5 and 0 < mem.r.ttl("mem:u1:entities") <= 5


def test_retrieval_cache_hits_and_new_data_invalidates_it(mem):
    assert mem.get_context("Who is X?", ["Person:x"]) is None
    mem.put_context("Who is X?", ["Person:x"], {"items": [1, 2]})
    assert mem.get_context("who is x", ["Person:x"]) == {"items": [1, 2]}            # same question, different case / punctuation
    assert mem.get_context("Who is X?", ["Person:y"]) is None                          # different entities: different entry
    mem.bump_version()                                                                 # a document was loaded
    assert mem.get_context("Who is X?", ["Person:x"]) is None
    assert normalise("  Who   is X?? ") == "who is x"


def test_cache_entries_expire(mem):
    mem.cache_ttl = 3
    mem.put_context("q", [], {"a": 1})
    key = mem._ckey("q", [])
    assert 0 < mem.r.ttl(key) <= 3


def test_redis_down_never_raises():
    m = Memory(Broken())
    assert m.ping() is False
    assert m.recall("u") == {"turns": [], "entities": []} and m.get_context("q") is None and m.version() == 0
    m.remember("u", "q", "a", entities=[{"id": "x", "type": "Person", "name": "X"}])
    m.put_context("q", [], {}), m.forget("u"), m.bump_version()                       # all silently ignored
