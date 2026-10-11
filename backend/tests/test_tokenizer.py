import pytest

from tokenizer.bpe_tokenizer import BPETokenizer


def test_special_token_ids(tok):
    assert (tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id) == (0, 1, 2, 3)
    assert tok.token_id("<CASUAL_CONVERSATION>") == 6


def test_roundtrip_preserves_text(tok):
    for text in ["hii", "What is the CAG?", "Line one\nLine two", "Rs 1,000 in 2023.", "a  b"]:
        assert tok.decode(tok.encode(text)) == text


def test_unknown_words_split_into_pieces_not_unk(tok):
    word = "Pneumonoultramicroscopic"
    ids = tok.encode(word)
    assert tok.unk_id not in ids
    assert len(ids) > 1                      # broken into subword pieces
    assert tok.decode(ids) == word


def test_punctuation_numbers_and_unicode(tok):
    text = "Hello, world! 12345 தமிழ் (v2.0)"
    ids = tok.encode(text)
    assert tok.unk_id not in ids
    assert tok.decode(ids) == text


def test_long_word(tok):
    word = "x" * 500
    assert tok.decode(tok.encode(word)) == word


def test_bos_eos_flags(tok):
    ids = tok.encode("hii", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id


def test_vocabulary_is_fixed(tok):
    before = tok.vocab_size
    tok.encode("brand new words never seen zzzqqq")
    assert tok.vocab_size == before


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_bpe"):
        BPETokenizer(tmp_path / "nope.json")


def test_hash_ignores_windows_line_endings(tok_file, tmp_path):
    """A git checkout on Windows can turn LF into CRLF; the tokenizer must still match its checkpoints."""
    crlf = tmp_path / "tokenizer.json"
    crlf.write_bytes(tok_file.read_bytes().replace(b"\n", b"\r\n"))
    assert crlf.read_bytes() != tok_file.read_bytes()
    assert BPETokenizer(crlf).sha256 == BPETokenizer(tok_file).sha256
    assert BPETokenizer(crlf).encode("What is the CAG?") == BPETokenizer(tok_file).encode("What is the CAG?")
