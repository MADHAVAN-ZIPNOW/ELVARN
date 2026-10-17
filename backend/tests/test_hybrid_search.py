"""Keyword (BM25) search, fusion with the vectors, the exact section match and the cross-encoder reranker. All text is FICTIONAL."""
import pytest

from knowledge import graphrag
from knowledge.lexical import Lexical, chroma_rows, exact_matcher, match_query
from knowledge.reranker import Reranker, sigmoid


def test_match_query_adds_the_exact_section_as_phrases_in_every_spelling():
    q = match_query("What does Section 83(1)(a) of the Representation of the People Act require?")
    assert '"83 1 a"' in q and '"section 83 1 a"' in q and '"s 83 1 a"' in q and '"sec 83 1 a"' in q
    assert '"representation"' in q and '"people"' in q
    assert '"what"' not in q and '"does"' not in q and '"act"' not in q                 # question filler and very common words are not search terms


def test_match_query_with_nothing_usable_is_empty():
    assert match_query("what does the") == "" and match_query("") == ""


def test_exact_matcher_needs_the_section_number_and_most_words_of_the_act():
    m = exact_matcher("What does Section 12(6) of the Karnataka Sales Tax Act provide?")
    assert m("The dealer relied on Section 12(6) of the Karnataka Sales Tax Act to claim the refund.")
    assert m("Under s.12(6) of the Karnataka Sales Tax Act the tax is levied.")
    assert not m("Section 12(7) of the Karnataka Sales Tax Act is different.")             # another sub-section
    assert not m("Section 12(6) of the Evidence Act deals with something else.")           # another Act
    assert not m("The Karnataka Sales Tax Act has twelve parts.")                          # no section number
    assert exact_matcher("tell me about corruption") is None                               # a question that names no section


def test_exact_matcher_does_not_treat_a_longer_number_as_the_section():
    m = exact_matcher("What is Section 5 of the Limitation Act?")
    assert m("Section 5 of the Limitation Act allows delay to be condoned.")
    assert not m("Section 50 of the Limitation Act is about something else.")
    assert not m("Section 51 of the Limitation Act is about something else.")


@pytest.fixture()
def passages():
    return [("p1", "Section 83(1)(a) of the Representation of the People Act requires an election petition to contain a concise statement of material facts."),
            ("p2", "Section 80 of the Representation of the People Act says no election shall be questioned except by an election petition."),
            ("p3", "The appellant was arrested on the allegation of money laundering and later released on bail."),
            ("p4", "An election petition must be presented within forty five days of the date of election of the returned candidate.")]


def test_lexical_index_finds_the_exact_section_first(tmp_path, passages):
    path = tmp_path / "lex.db"
    assert Lexical.build(path, passages) == 4
    lex = Lexical(path)
    assert lex.count() == 4
    ids = [pid for pid, _ in lex.search("What does Section 83(1)(a) of the Representation of the People Act require?", k=4)]
    assert ids[0] == "p1" and "p3" not in ids[:2]
    assert lex.search("completely unrelated zebra xylophone") == []
    assert lex.search("") == []


def test_lexical_build_never_overwrites_an_index(tmp_path, passages):
    path = tmp_path / "lex.db"
    Lexical.build(path, passages)
    with pytest.raises(FileExistsError):
        Lexical.build(path, passages)


def test_lexical_from_env_is_none_without_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LEXICAL_DB", str(tmp_path / "missing.db"))
    assert Lexical.from_env() is None


def fill(rag_store, tmp_path, passages):
    rag_store.add([{"id": pid, "text": text, "meta": {"title": "Case " + pid, "doc_type": "judgment", "source_id": "s" + pid}} for pid, text in passages])
    Lexical.build(tmp_path / "lex.db", chroma_rows(rag_store))
    rag_store.lexical = Lexical(tmp_path / "lex.db")


def test_hybrid_query_puts_the_exact_section_passage_first(rag_store, tmp_path, passages):
    fill(rag_store, tmp_path, passages)
    hits = rag_store.hybrid_query("What does Section 83(1)(a) of the Representation of the People Act require?", k=4)
    assert hits[0]["id"] == "p1" and hits[0]["exact"] is True and "rrf" in hits[0]
    assert [h["id"] for h in hits if h["exact"]] == ["p1"]                                   # only the passage with that exact section is flagged


def test_hybrid_query_without_a_keyword_index_is_the_plain_vector_query(rag_store, passages):
    rag_store.add([{"id": pid, "text": text, "meta": {"title": pid, "doc_type": "judgment", "source_id": pid}} for pid, text in passages])
    assert rag_store.lexical is None
    plain = rag_store.query("election petition", k=3)
    assert [h["id"] for h in rag_store.hybrid_query("election petition", k=3)] == [h["id"] for h in plain]


def test_exact_boost_can_be_switched_off(rag_store, tmp_path, passages, monkeypatch):
    fill(rag_store, tmp_path, passages)
    monkeypatch.setenv("ELVARN_EXACT_BOOST", "0")
    assert all(h["exact"] is False for h in rag_store.hybrid_query("Section 83(1)(a) of the Representation of the People Act", k=4))


class FakeReranker:
    """Scores by a fixed table so the test controls the order."""
    def __init__(self, table):
        self.table, self.asked = table, []

    def scores(self, question, texts):
        self.asked.append(list(texts))
        return [next((v for key, v in self.table.items() if key in t), -5.0) for t in texts]


def understanding(question):
    return graphrag.Understanding(question, question, [], "general", graphrag.rag.terms(question))


def test_vector_items_follow_the_reranker_and_keep_an_exact_match_ahead(rag_store, tmp_path, passages):
    fill(rag_store, tmp_path, passages)
    q = "election petition Section 83(1)(a) Representation of the People Act"
    rag_store.reranker = FakeReranker({"forty five days": 6.0, "Section 80": 2.0, "Section 83(1)(a)": 3.0})
    items = graphrag.vector_items(rag_store, understanding(q))
    texts = [i["text"] for i in items]
    assert "forty five days" in texts[0] or "Section 83(1)(a)" in texts[0]
    assert any("Section 83(1)(a)" in t for t in texts[:2])                                  # the exact section keeps its lead over a plain related passage
    assert rag_store.reranker.asked and all("_exact" not in i for i in items)


def test_reranker_can_be_switched_off(rag_store, tmp_path, passages, monkeypatch):
    fill(rag_store, tmp_path, passages)
    rag_store.reranker = FakeReranker({})
    monkeypatch.setenv("ELVARN_RERANK", "0")
    graphrag.vector_items(rag_store, understanding("election petition Section 83(1)(a) Representation of the People Act"))
    assert rag_store.reranker.asked == []


def test_the_real_reranker_ranks_the_matching_sentence_above_a_neighbour():
    rr = Reranker.from_env()
    if rr is None:
        pytest.skip("the reranker files are not present")
    q = "What does Section 83(1)(a) of the Representation of the People Act require?"
    right = "s.83(1)(a) mandates that an Election petition shall contain a concise statement of material facts on which the petitioner relies."
    near = "Section 80 of the RP Act states that no election shall be called in question except by an Election Petition presented in accordance with Part-VI."
    far = "The appellant was a former minister and was arrested on the allegation of money laundering."
    s_right, s_near, s_far = rr.scores(q, [right, near, far])
    assert s_right > s_near > s_far


def test_reranker_from_env_is_none_without_files(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANKER_DIR", str(tmp_path))
    assert Reranker.from_env() is None


def test_sigmoid_is_bounded_and_monotone():
    assert 0 < sigmoid(-100) < sigmoid(0) == 0.5 < sigmoid(100) < 1.0 + 1e-12
