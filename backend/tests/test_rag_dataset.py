"""The RAG dataset builder: live-prompt format, answers only from notes, topic split without leaks, token limits."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "data_tools"))

import build_rag_dataset as rag  # noqa: E402
import continue_train  # noqa: E402
from search.prompt import FOOTER, HEADER, build_evidence_text  # noqa: E402
from search.service import Source  # noqa: E402

SENT = [
    "{t} is a small research institute that studies river systems and the plants that grow near them.",
    "The institute was founded by Maria Okonkwo in 1987 after a long survey of the northern valley.",
    "In 2004 the team published a report with 312 pages describing the water quality of nine rivers.",
    "Visitors can see a public garden with more than 40 kinds of local trees and several walking paths.",
]


def write_raw(directory, n_topics=240):
    directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    for i in range(n_topics):
        t = f"Topic{i:03d} Institute"
        text = "\n".join(s.format(t=t) for s in SENT)
        blocks.append(f"<DOMAIN_KNOWLEDGE>\nTopic: {t}\nText: {text}\nSource: https://en.wikipedia.org/wiki/T{i}\n"
                      f"Verification status: Unverified\n</DOMAIN_KNOWLEDGE>")
    (directory / "wikipedia_domain.txt").write_text("\n\n".join(blocks), encoding="utf-8")
    qa = [f"<GENERAL_ENGLISH>\nQuestion: What is item number {i}?\nAnswer: Item number {i} is a thing.\n</GENERAL_ENGLISH>" for i in range(30)]
    (directory / "dolly_qa.txt").write_text("\n\n".join(qa), encoding="utf-8")
    return directory


@pytest.fixture()
def raw(tmp_path):
    return write_raw(tmp_path / "raw")


def test_sentence_splitting_keeps_abbreviations_together():
    s = rag.split_sentences("Dr. Smith met J. K. Rowling in the U.S. last year. They talked. Then they left at 5 p.m. sharp.")
    assert s[0].startswith("Dr. Smith met J. K. Rowling in the U.S. last year.")
    assert "They talked." in s


def test_prompt_format_is_the_live_one():
    notes = ["Paris is the capital of France and its largest city.", "The city sits on the river Seine in the north."]
    srcs = [Source("t", "https://a.com", "a.com", n) for n in notes]
    live, used = build_evidence_text("What is Paris?", srcs, lambda s: len(s.split()), 1000, 80)
    assert len(used) == 2
    assert rag.evidence_text(notes, "What is Paris?") == live
    assert live.startswith(HEADER) and f"\n{FOOTER} What is Paris?" in live
    text = rag.render("domain", live, "Paris. [1]")
    assert text.startswith("<DOMAIN_KNOWLEDGE>\nTopic: General\nQuestion: Web notes") and text.endswith("My question: What is Paris?\nAnswer: Paris. [1]")


def test_topic_rule_matches_corpus_group_key():
    from corpus import _bucket, group_key
    doc = {"text": "<DOMAIN_KNOWLEDGE>\nTopic: Physics\nText: x\nSource: https://en.wikipedia.org/wiki/Physics"}
    assert rag.is_val_topic("Physics") == (_bucket(group_key(doc)) < 500)


def build_small(tmp_path, tok, raw, **kw):
    out = tmp_path / "rag"
    meta, splits = rag.build(out, tok, seed=1, train_target=300, val_target=40, raw_dir=raw, dolly_file=None, **kw)
    return out, meta, splits


def test_build_files_meta_and_tokenizer_consistency(tmp_path, tok, raw):
    out, meta, splits = build_small(tmp_path, tok, raw)
    assert meta["tokenizer_sha256"] == tok.sha256 and meta["vocab_size"] == tok.vocab_size
    assert "format_description" in meta and meta["dolly"]["usable_examples"] == 0
    for name in ("train", "val"):
        arr = np.fromfile(out / f"{name}.bin", dtype=np.uint16)
        assert arr.size == meta[name]["tokens"] > 0
        assert int(arr.max()) < tok.vocab_size and not (arr == tok.pad_id).any()
        assert int((arr == tok.bos_id).sum()) == int((arr == tok.eos_id).sum()) == meta[name]["documents"]   # <BOS> ... <EOS>
        assert arr[0] == tok.bos_id and arr[-1] == tok.eos_id
        assert meta[name]["length_tokens"]["max"] <= rag.MAX_TOKENS
        assert sum(meta[name]["length_histogram"].values()) == meta[name]["documents"]
    # the trainer's own check accepts this dataset with a matching checkpoint
    ck = {"model_state_dict": {}, "config": {}, "vocab_size": tok.vocab_size, "tokenizer_sha256": tok.sha256}
    d = tmp_path / "ck"
    d.mkdir()
    (d / "tokenizer.json").write_bytes(tok.path.read_bytes())
    meta_on_disk = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert continue_train.check_checkpoint(ck, d / "m.pt", meta_on_disk).sha256 == tok.sha256


def test_refuses_to_overwrite(tmp_path, tok, raw):
    out, _, _ = build_small(tmp_path, tok, raw)
    with pytest.raises(SystemExit):
        rag.build(out, tok, raw_dir=raw, dolly_file=None)


def test_no_topic_leaks_and_val_topics_are_original_v2_val_topics(tmp_path, tok, raw):
    out, meta, splits = build_small(tmp_path, tok, raw)
    tr = {e["topic"] for e in splits["train"] if e["topic"] != "dolly"}
    va = {e["topic"] for e in splits["val"] if e["topic"] != "dolly"}
    assert tr and va and not (tr & va)
    assert all(rag.is_val_topic(t) for t in va) and not any(rag.is_val_topic(t) for t in tr)
    # also no distractor note of a validation example comes from a training topic
    train_notes = {n for e in splits["train"] for n in e["notes"]}
    val_notes = {n for e in splits["val"] for n in e["notes"]}
    shared_topic_names = [t for t in tr if any(t in n for n in val_notes)]
    assert not shared_topic_names


def test_answers_come_only_from_the_notes_and_types_are_right(tmp_path, tok, raw):
    out, meta, splits = build_small(tmp_path, tok, raw)
    kinds = {e["type"] for e in splits["train"]}
    assert {"single", "single_distractors", "double", "negative", "plain"} <= kinds
    for e in splits["train"]:
        if e["type"] in ("negative", "negative_same_passage"):
            assert e["answer"] in rag.REFUSALS and not e["answerable"]
            if e["type"] == "negative":
                assert not any(e["topic"].lower() in n.lower() for n in e["notes"])    # notes really are off-topic
        elif e["type"] in ("single", "single_distractors", "double"):
            body = e["answer"]
            used = [n for n in e["notes"] if n in body]
            assert used, "answer must contain a note sentence"
            rest = body
            for n in used:
                rest = rest.replace(n, "")
            import re
            rest = re.sub(r"\[\d\]|According to|and|From the notes:|:|\s", "", rest)
            assert rest == "", f"answer has text that is not from the notes: {rest!r}"
            for m in re.findall(r"\[(\d)\]", body):
                assert e["notes"][int(m) - 1] in body                                  # a citation points at a note that is quoted
        elif e["type"] == "plain":
            assert e["notes"] == []
    n = len(splits["train"])
    share = lambda k: sum(e["type"] in k for e in splits["train"]) / n
    assert 0.08 <= share({"negative", "negative_same_passage"}) <= 0.30
    assert 0.05 <= share({"plain"}) <= 0.30


def test_modes_and_wrapping(tmp_path, tok, raw):
    out, meta, splits = build_small(tmp_path, tok, raw)
    modes = {e["mode"] for e in splits["train"]}
    assert {"domain", "english"} <= modes
    for e in splits["train"][:50]:
        text = rag.example_text(e)
        assert text.split("\n", 1)[0] in ("<DOMAIN_KNOWLEDGE>", "<GENERAL_ENGLISH>", "<CASUAL_CONVERSATION>")
        assert ("Web notes" in text) == bool(e["notes"])


def test_long_examples_are_dropped(tok):
    ex = dict(type="single", mode="domain", topic="T", notes=["word " * 40], question="q?", answer="a " * 50, answerable=True)
    ids, kept, lengths, dropped = rag.encode_examples([ex], tok, max_tokens=20)
    assert dropped == 1 and not ids
    ids, kept, lengths, dropped = rag.encode_examples([ex], tok, max_tokens=512)
    assert dropped == 0 and lengths == [len(ids)] and ids[0] == tok.bos_id and ids[-1] == tok.eos_id


def test_dolly_context_examples_loaded_from_a_local_file(tmp_path):
    rows = [
        {"instruction": "Who founded the club?", "context": "The club was founded by Ana Silva in 1999. It has 200 members and a small library.",
         "response": "Ana Silva founded the club in 1999.", "category": "closed_qa"},
        {"instruction": "Say something", "context": "", "response": "Hello there friend", "category": "closed_qa"},
        {"instruction": "Tell me a joke please", "context": "Some context here that is long enough to count.", "response": "Banana split nonsense words", "category": "creative_writing"},
    ]
    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    got = rag.load_dolly_examples(f)
    assert len(got) == 1 and got[0]["question"] == "Who founded the club?" and 1 <= len(got[0]["notes"]) <= 3


def test_eval_cases_from_validation_topics_only(tmp_path, tok, raw):
    out = tmp_path / "rag2"
    ev = tmp_path / "eval.json"
    rag.build(out, tok, seed=1, train_target=100, val_target=20, raw_dir=raw, dolly_file=None, write_eval=ev, eval_cases=10)
    cases = json.loads(ev.read_text(encoding="utf-8"))["cases"]
    assert 5 <= len(cases) <= 10
    assert all(rag.is_val_topic(c["topic"]) for c in cases)
    assert any(not c["answerable"] for c in cases) or len(cases) < 5
    for c in cases:
        assert c["answerable"] == bool(c["groups"])
        assert c["reference_answer"]
    with pytest.raises(SystemExit):
        rag.build(tmp_path / "rag3", tok, raw_dir=raw, dolly_file=None, write_eval=ev)


def test_key_groups_prefer_numbers_and_names():
    g = rag.key_groups(["The institute was founded by Maria Okonkwo in 1987 after a survey."], "Who founded it?")
    assert ["1987"] in g and ["Maria"] in g
