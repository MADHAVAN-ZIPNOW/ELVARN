"""extend_vocab --scripts: whole characters (Tamil, Hindi, curly quotes) become single tokens and everything old stays."""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "training"))
sys.path.insert(0, str(BACKEND / "tests"))

import extend_vocab as ev  # noqa: E402
from test_extend_vocab import tiny_tokenizer  # noqa: E402


def test_script_characters_are_real_assigned_characters():
    tamil = ev.script_characters(["tamil"])
    assert "ந" in tamil and "ீ" in tamil and len(tamil) > 60 and all("஀" <= c <= "௿" for c in tamil)
    assert "“" in ev.script_characters(["punct"]) and "₹" in ev.script_characters(["punct"])


def test_a_tamil_letter_and_curly_quotes_become_one_token_each_and_text_roundtrips():
    base = tiny_tokenizer()
    chars = ["ந", "ீ", "“", "”"]
    assert all(len(base.encode(c).ids) >= 2 for c in chars)                          # today: several byte tokens each
    new_json, steps = ev.extend_tokenizer(json.loads(base.to_str()), [], base, chars)
    new = Tokenizer.from_str(json.dumps(new_json))
    assert all(len(new.encode(c).ids) == 1 for c in chars)
    text = "நீதி “quoted” and plain english stays the same"
    assert new.decode(new.encode(text).ids) == text
    plain = "a b c d e f x y z"
    assert new.encode(plain).ids == base.encode(plain).ids                           # old text: the very same ids
    assert len(steps) >= len(chars) and new.get_vocab_size() == base.get_vocab_size() + len(steps)
