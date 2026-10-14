"""Text helpers behind the retrieval pipeline (no database needed). The pipeline itself is tested in test_graphrag.py."""
from knowledge import rag


def test_terms_drop_stopwords_and_keep_numbers():
    assert rag.terms("What did the High Court say about disproportionate assets?") == ["high", "court", "disproportionate", "assets"]
    assert rag.terms("2G scam of 2008") == ["2g", "scam", "2008"] or rag.terms("2G scam of 2008") == ["scam", "2008"]
    assert rag.terms("hii") == ["hii"] and rag.terms("what is the") == []


def test_overlap_matches_word_stems():
    assert rag.overlap(["arrested", "officers"], "The officer was arrest by the police") == 2
    assert rag.overlap(["bribe"], "nothing to see") == 0


def test_snippet_picks_the_relevant_sentence_and_a_neighbour_only_if_on_topic():
    text = "The weather was pleasant that day. The department alleged disproportionate assets of the officer. Lunch was served."
    snippet = rag.best_snippet(text, ["disproportionate", "assets"])
    assert "disproportionate" in snippet and "weather" not in snippet and "Lunch" not in snippet
    on_topic = "The officer owned a house. The assets were disproportionate to income. The assets were seized later."
    assert "seized" in rag.best_snippet(on_topic, ["assets", "disproportionate"])
    assert len(rag.best_snippet("word " * 200, ["word"])) <= rag.SNIPPET_CHARS


def test_labels_cover_every_document_type():
    from knowledge.kb import DOC_TYPES
    assert all(t in rag.LABELS for t in DOC_TYPES)
