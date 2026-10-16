"""vocab_sync: new words go into the main vocabulary (append only), the new text is stored for training, an older model is adapted to the newer vocabulary."""
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "training"))
sys.path.insert(0, str(BACKEND / "tests"))

import extend_vocab as ev  # noqa: E402
from test_extend_vocab import tiny_tokenizer  # noqa: E402

from backend.pipeline import vocab_sync  # noqa: E402
from model.transformer import build_model  # noqa: E402

LONG = ("The petitioner moved the High Court. The respondent filed a counter affidavit. Jurisprudence on adjudication is settled. " * 6)


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def setup(tmp_path):
    tok = tiny_tokenizer()
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    main = tok_dir / "tokenizer.json"
    main.write_text(tok.to_str(), encoding="utf-8")
    row = lambda text, topic="judgment": {"text": text, "title": "A vs B, No. 1/2020, Supreme Court of India", "source_url": "http://x/1.pdf", "topic": topic}  # noqa: E731
    write_parquet(tmp_path / "corpus" / "passages" / "source=pdf" / "date=2026-10-16" / "part-0.parquet", [row(LONG) for _ in range(6)] + [row("Wikipedia style text about a person.", "person")])
    return tok, main


def run(tmp_path, main, **kw):
    args = dict(corpus_dir=tmp_path / "corpus", tokenizer=main, text_out=tmp_path / "raw_v2" / "kb_judgments.txt", state_db=tmp_path / "state.sqlite", min_count=5,
                scripts=("punct",), say=lambda m: None)
    args.update(kw)
    return vocab_sync.run_once(**args)


def test_new_words_grow_the_main_vocabulary_append_only_and_text_is_stored(tmp_path):
    base, main = setup(tmp_path)
    before = json.loads(main.read_text(encoding="utf-8"))
    result = run(tmp_path, main)
    after = json.loads(main.read_text(encoding="utf-8"))
    assert result["new_files"] == 1 and result["new_tokens"] > 0 and result["vocab_size"] == len(after["model"]["vocab"]) > len(before["model"]["vocab"])
    assert ev.is_extension(before, after)                                          # old ids never moved, old merges are the first merges
    grown = Tokenizer.from_str(json.dumps(after))
    assert len(grown.encode(" petitioner").ids) == 1 and len(grown.encode(LONG).ids) < len(base.encode(LONG).ids)
    assert grown.decode(grown.encode(LONG).ids) == LONG
    text = (tmp_path / "raw_v2" / "kb_judgments.txt").read_text(encoding="utf-8")
    assert text.count("<DOMAIN_KNOWLEDGE>") >= 6 and "Topic: A vs B, No. 1/2020" in text and "Wikipedia style" not in text      # only judgments and court records
    log = [json.loads(l) for l in (main.parent / "vocab_versions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[-1]["version"] == 1 and log[-1]["vocab_before"] == len(before["model"]["vocab"]) and (main.parent / "tokenizer.original.json").is_file()


def test_a_second_pass_with_nothing_new_changes_nothing_and_a_new_file_grows_it_again(tmp_path):
    base, main = setup(tmp_path)
    run(tmp_path, main)
    first = main.read_text(encoding="utf-8")
    text_size = (tmp_path / "raw_v2" / "kb_judgments.txt").stat().st_size
    again = run(tmp_path, main)
    assert again["new_files"] == 0 and again["new_tokens"] == 0 and main.read_text(encoding="utf-8") == first
    assert (tmp_path / "raw_v2" / "kb_judgments.txt").stat().st_size == text_size              # the same text is never stored twice
    write_parquet(tmp_path / "corpus" / "passages" / "source=pdf" / "date=2026-10-17" / "part-1.parquet",
                  [{"text": "The assessee challenged the impugned assessment order. " * 12, "title": "T", "source_url": "u", "topic": "judgment"} for _ in range(5)])
    grew = run(tmp_path, main)
    assert grew["new_files"] == 1 and grew["new_tokens"] > 0
    assert ev.is_extension(json.loads(first), json.loads(main.read_text(encoding="utf-8")))     # version 2 still extends version 1
    assert len((main.parent / "vocab_versions.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_the_vocabulary_never_passes_the_16_bit_limit(tmp_path):
    base, main = setup(tmp_path)
    size = len(json.loads(main.read_text(encoding="utf-8"))["model"]["vocab"])
    result = run(tmp_path, main, max_vocab=size + 3)
    assert result["vocab_size"] <= size + 3


def test_an_older_model_is_adapted_to_the_newer_vocabulary(tmp_path):
    base, main = setup(tmp_path)
    old_path = tmp_path / "old_tokenizer.json"
    old_path.write_text(main.read_text(encoding="utf-8"), encoding="utf-8")
    run(tmp_path, main)
    config = {"embedding_size": 16, "hidden_size": 32, "sequence_length": 12, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "tie_weights": True}
    torch.manual_seed(0)
    old_model = build_model(base.get_vocab_size(), config).eval()
    ck = {"model_state_dict": old_model.state_dict(), "config": config, "vocab_size": base.get_vocab_size(), "tokenizer_sha256": "x"}
    added = ev.adapt_checkpoint(ck, old_path, main)
    assert added > 0 and ck["vocab_size"] == base.get_vocab_size() + added
    new_model = build_model(ck["vocab_size"], config).eval()
    new_model.load_state_dict(ck["model_state_dict"], strict=True)
    ids = torch.tensor([[2, 5, 9, 14, 3, 7, 11, 20]])
    with torch.no_grad():
        a, b = old_model(ids), new_model(ids)
    assert torch.allclose(a, b[..., :a.shape[-1]], atol=1e-6, rtol=0)                      # old tokens score exactly as before
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"model": {"vocab": {"zzz": 0}, "merges": []}}), encoding="utf-8")
    try:
        ev.adapt_checkpoint(dict(ck, vocab_size=1), old_path, other)
        raise AssertionError("a different vocabulary must be refused")
    except ValueError:
        pass


def test_training_start_adapts_a_checkpoint_to_the_grown_main_vocabulary(tmp_path, monkeypatch):
    import continue_train
    from tokenizer.bpe_tokenizer import BPETokenizer
    base, main = setup(tmp_path)
    ck_dir = tmp_path / "ckpt"
    ck_dir.mkdir()
    (ck_dir / "tokenizer.json").write_text(main.read_text(encoding="utf-8"), encoding="utf-8")
    old = BPETokenizer(ck_dir / "tokenizer.json")
    config = {"embedding_size": 16, "hidden_size": 32, "sequence_length": 12, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "tie_weights": True}
    model = build_model(old.vocab_size, config)
    ck = {"model_state_dict": model.state_dict(), "config": config, "vocab_size": old.vocab_size, "tokenizer_sha256": old.sha256}
    run(tmp_path, main)                                                         # the main vocabulary grows
    grown = BPETokenizer(main)
    monkeypatch.setattr(continue_train, "MAIN_TOKENIZER", main)
    tok = continue_train.check_checkpoint(ck, ck_dir / "best.pt", {"tokenizer_sha256": grown.sha256})      # a dataset built with the grown vocabulary
    assert tok.sha256 == grown.sha256 and ck["vocab_size"] == grown.vocab_size and ck["tokenizer_sha256"] == grown.sha256
    assert ck["model_state_dict"]["token_embedding.weight"].shape[0] == grown.vocab_size
    same = continue_train.check_checkpoint({**ck, "model_state_dict": model.state_dict(), "vocab_size": old.vocab_size, "tokenizer_sha256": old.sha256},
                                            ck_dir / "best.pt", {"tokenizer_sha256": old.sha256})              # an old dataset still works untouched
    assert same.sha256 == old.sha256
