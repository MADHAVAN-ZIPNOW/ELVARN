"""Every reply must say where it came from. The page decides this from fields the server already sends."""
from pathlib import Path

HTML = (Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html").read_text(encoding="utf-8")


def test_every_origin_case_has_a_visible_label():
    for label in ("Answered from: stored records.", "Answered from: live web search.", "Answered by the model alone.", "Answered from: nothing."):
        assert label in HTML


def test_origin_is_drawn_for_new_and_saved_replies_and_uses_text_only():
    assert HTML.count("renderAnswerInfo(b.srcHolder") == 2          # a reply that just finished, and a reply reloaded from history
    start = HTML.index("function renderAnswerInfo")
    block = HTML[start:start + 700]
    assert "innerHTML" not in block and "createTextNode" in block   # never raw HTML


def test_origin_follows_the_fields_the_server_sends():
    start = HTML.index("function originInfo")
    block = HTML[start:start + 1500]
    for field in ("stop_reason===\"knowledge_base\"", "s.kb", "s.searched"):
        assert field in block
