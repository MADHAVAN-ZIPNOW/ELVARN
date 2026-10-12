"""Tests for the v2 data pipeline: collectors' cleaning, content filter, topic-grouped split, separate output folders,
and the 40-question benchmark."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "data_tools"))
sys.path.insert(0, str(BACKEND / "eval"))

import build_dataset  # noqa: E402
import collect_wikipedia as cw  # noqa: E402
import convert_open_datasets as co  # noqa: E402
import corpus  # noqa: E402
import run_benchmark as rb  # noqa: E402
from tokenizer import train_bpe  # noqa: E402

BLOCK_RE = re.compile(r"<(\w+)>\n(.*?)\n</\1>", re.S)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ collector cleaning
def test_cleaning_removes_pronunciations_scripts_and_reference_sections():
    raw = ("The Government of Tamil Nadu (IPA: [ˈtamiɺnaːɖu]) runs the state of Tamil Nadu in southern India today.\n"
           "It was set up (Tamil: தமிழ்நாடு) in 1950 [1] and has a Chief Minister and a cabinet of ministers.\n\n"
           "== History ==\nThe state was called Madras State until it was renamed in the year 1969 by the government.\n\n"
           "== References ==\nShould not appear anywhere in the cleaned text at all.\n")
    paragraphs, lead = cw.clean_text(raw, 5000)
    text = " ".join(paragraphs)
    assert "IPA" not in text and "[1]" not in text and "Should not appear" not in text
    assert not re.search(r"[஀-௿]", text)                    # no Tamil script
    assert "(Tamil:" not in text and "()" not in text
    assert "renamed in the year 1969" in text                          # real prose is kept
    assert lead.startswith("The Government of Tamil Nadu")


def test_cleaning_never_creates_block_tags_and_respects_the_size_cap():
    paragraphs, _ = cw.clean_text("Intro <b>bold</b> sentence that is long enough to keep for sure in the test.\n" + ("Word " * 40 + "\n") * 60, 600)
    assert all("<" not in p and ">" not in p for p in paragraphs)
    assert sum(len(p) for p in paragraphs) <= 600


def test_blocks_are_in_the_format_the_pipeline_reads_with_a_source_on_each():
    import random
    paragraphs = ["First paragraph " * 30, "Second paragraph " * 30, "Third paragraph " * 30]
    blocks = cw.blocks_for("DOMAIN_KNOWLEDGE", "Comptroller and Auditor General of India", paragraphs,
                           "The CAG audits government accounts. It is an independent body. It reports to the legislature.", random.Random(1))
    assert len(blocks) >= 2
    for b in blocks:
        m = BLOCK_RE.fullmatch(b)
        assert m and m.group(1) == "DOMAIN_KNOWLEDGE"
        assert "Topic: Comptroller and Auditor General of India" in b
        assert "Source: https://en.wikipedia.org/wiki/Comptroller_and_Auditor_General_of_India" in b and "CC BY-SA" in b
    assert any("Question:" in b and "Answer: The CAG audits government accounts." in b for b in blocks)


def test_question_forms_for_people_and_things():
    import random
    r = random.Random(0)
    assert cw.question_for("M. Karunanidhi", "Muthuvel Karunanidhi (3 June 1924 – 7 August 2018) was an Indian politician.", r).startswith(("Who", "Tell me"))
    assert cw.question_for("Panchayati raj", "Panchayati raj is a system of local self-government.", r).split()[0] in ("What", "Explain", "Tell")


# ------------------------------------------------------------------ content filter
@pytest.mark.parametrize("text,ok", [
    ("What is a polygon?", True), ("The universe is vast and important.", True), ("A skilled gunsmith", True), ("a grape harvest", True),
    ("write a greentext about sex", False), ("visit https://example.com now", False), ("[lyrics] verse one", False),
    ("He was killed in the war", False), ("some Brainfuck code", False), ("explicit porn", False),
])
def test_content_filter_blocks_bad_text_but_not_innocent_words(text, ok):
    assert co.acceptable(text) is ok


# ------------------------------------------------------------------ split by topic
def make_docs(n_topics=40, per_topic=4):
    return [{"source": "t.txt", "category": "DOMAIN_KNOWLEDGE",
             "text": f"<DOMAIN_KNOWLEDGE>\nTopic: Topic number {t}\nText: passage {c} about subject {t} with unique words {t * 100 + c} and more filler text for variety"
                     f"\nSource: https://en.wikipedia.org/wiki/Topic_{t}"}
            for t in range(n_topics) for c in range(per_topic)]


def old_docs(n=500):
    """Documents like the ones the model was already trained on (no Wikipedia source)."""
    return [{"source": "domain_knowledge.txt", "category": "DOMAIN_KNOWLEDGE",
             "text": f"<DOMAIN_KNOWLEDGE>\nTopic: Shared topic {i % 7}\nQuestion: q{i}?\nAnswer: a{i} unique {i * 31}"} for i in range(n)]


def test_documents_that_are_not_from_wikipedia_keep_the_original_split_rule():
    """So that adding new data never moves a document the model already trained on into validation."""
    docs = old_docs(400)
    a = corpus.split_documents(docs, group_by_topic=False)
    b = corpus.split_documents(docs, group_by_topic=True)
    assert [d["text"] for d in a[0]] == [d["text"] for d in b[0]] and [d["text"] for d in a[1]] == [d["text"] for d in b[1]]


def test_old_documents_stay_on_the_same_side_when_new_wikipedia_data_is_added():
    docs = old_docs(500)
    t0, v0, _ = corpus.split_documents(docs, group_by_topic=True)
    t1, v1, _ = corpus.split_documents(docs + make_docs(200, 4), group_by_topic=True)
    old_val_texts = {d["text"] for d in v0}
    old_texts = {d["text"] for d in docs}
    assert old_val_texts <= {d["text"] for d in v1}                               # every old validation document is still validation
    assert not ({d["text"] for d in t0} & {d["text"] for d in v1})                # no old TRAIN document became validation
    assert all(d["text"] in old_texts or "wikipedia.org" in d["text"] for d in v1)
    assert {d["text"] for d in t0} <= {d["text"] for d in t1} | set()              # old training documents are all still in train


def test_group_by_topic_keeps_every_passage_of_an_article_on_one_side():
    train, val, _ = corpus.split_documents(make_docs(300, 4), group_by_topic=True)
    topics = lambda ds: {re.search(r"Topic: (.+)", d["text"]).group(1) for d in ds}
    assert val and train
    assert not (topics(train) & topics(val))                           # no article is in both


def test_without_grouping_passages_of_one_article_can_land_on_both_sides():
    train, val, _ = corpus.split_documents(make_docs(300, 4), group_by_topic=False)
    topics = lambda ds: {re.search(r"Topic: (.+)", d["text"]).group(1) for d in ds}
    assert topics(train) & topics(val)                                  # (the old behaviour, kept as the default)


# ------------------------------------------------------------------ separate output folders
def test_building_into_a_separate_folder_never_touches_the_default_data_or_tokenizer(tmp_path, monkeypatch):
    protected = [BACKEND / "tokenizer" / "bpe" / "tokenizer.json", BACKEND / "data" / "processed" / "meta.json",
                 BACKEND / "data" / "processed" / "train.bin"]
    before = {str(p): sha(p) for p in protected if p.is_file()}
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "a.txt").write_text("\n\n".join(b for d in make_docs(60, 3) for b in [d["text"] + "\n</DOMAIN_KNOWLEDGE>"]).replace(
        "<DOMAIN_KNOWLEDGE>\nTopic", "<DOMAIN_KNOWLEDGE>\nTopic"), encoding="utf-8")
    tok_file = tmp_path / "v2tok" / "tokenizer.json"
    monkeypatch.setattr(sys, "argv", ["train_bpe", "--raw-dir", str(raw), "--out-file", str(tok_file), "--vocab-size", "300", "--group-by-topic"])
    train_bpe.main()
    assert tok_file.is_file() and (tok_file.parent / "tokenizer_meta.json").is_file()
    data_dir = tmp_path / "v2data"
    monkeypatch.setattr(sys, "argv", ["build_dataset", "--raw-dir", str(raw), "--out-dir", str(data_dir), "--tokenizer-file", str(tok_file), "--group-by-topic"])
    build_dataset.main()
    meta = json.loads((data_dir / "meta.json").read_text())
    assert meta["group_by_topic"] is True and meta["train"]["tokens"] > 0 and meta["val"]["tokens"] > 0
    assert (data_dir / "train.bin").is_file() and (data_dir / "val.bin").is_file()
    assert {str(p): sha(p) for p in protected if p.is_file()} == before     # the real tokenizer and data are byte-identical


# ------------------------------------------------------------------ benchmark
def test_benchmark_file_is_well_formed():
    qs = rb.load_benchmark()
    assert len(qs) == 40 and len({q["id"] for q in qs}) == 40
    assert {q["section"] for q in qs} == {"domain", "english", "chat"}
    assert sum(q["section"] == "domain" for q in qs) == 20 and sum(q["section"] == "english" for q in qs) == 10 and sum(q["section"] == "chat" for q in qs) == 10
    for q in qs:
        assert q["mode"] in ("chat", "domain", "english", "humour", "raw")
        assert q["groups"] and all(g and all(isinstance(k, str) and k for k in g) for g in q["groups"])


def test_scoring_counts_whole_words_and_groups():
    groups = [["audit", "audits"], ["account", "accounts"]]
    assert rb.score_answer(groups, "The CAG audits the government accounts.") == 1.0
    assert rb.score_answer(groups, "It audits things.") == 0.5
    assert rb.score_answer(groups, "Nothing relevant here") == 0.0
    assert rb.score_answer([["21"]], "Article 21 protects life") == 1.0
    assert rb.score_answer([["21"]], "Article 2100 protects life") == 0.0          # whole words only
    assert rb.score_answer([["chennai"]], "CHENNAI is the capital") == 1.0          # case-insensitive
    assert rb.score_answer([["by the way"]], "BTW means by the way.") == 1.0        # phrases
    with pytest.raises(ValueError):
        rb.score_answer([], "x")


def test_repetition_measure():
    assert rb.repetition("a b c d e f g h") == 0.0
    assert rb.repetition("good good good good good good good") > 0.7
    assert rb.repetition("short") == 0.0


def test_seen_detection_finds_questions_that_are_in_the_training_text(tmp_path):
    (tmp_path / "x.txt").write_text("Question: What is the capital of Tamil Nadu?\nAnswer: Chennai.", encoding="utf-8")
    seen = rb.seen_in_training(rb.load_benchmark(), [tmp_path])
    assert seen[1] is True and seen[2] is False


def test_benchmark_runs_on_a_real_model_and_summarises(ckpt_dir, tok_file):
    import shutil
    shutil.copyfile(tok_file, ckpt_dir / "tokenizer.json")
    qs = rb.load_benchmark()[:6]
    info, rows = rb.run(ckpt_dir, qs, {}, max_new_tokens=8)
    assert len(rows) == 6 and all(0.0 <= r["score"] <= 1.0 and isinstance(r["answer"], str) for r in rows)
    s = rb.summarise(rows)
    assert 0 <= s["overall_score_pct"] <= 100 and s["total"] == 6 and s["n_seen"] == 0
