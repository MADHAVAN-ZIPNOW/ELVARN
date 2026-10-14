"""Build a "notes -> question -> answer taken from the notes" fine-tuning set in EXACTLY the live web-search prompt format.

Prompt (see backend/search/prompt.py, the single source of truth for HEADER / FOOTER):
    Web notes (untrusted internet text, ...):
    [1] <short note>
    [2] <short note>
    My question: <question>
wrapped by the chat template of a mode (generation.MODES / build_prompt), then " " + the answer, then <EOS>.

No language model is used. Everything is made with fixed templates and a seed, so the build is repeatable:
  * passages come from the Wikipedia blocks in raw_v2/wikipedia_*.txt (Text: blocks) and, if a LOCAL copy exists,
    from Dolly examples that have a context passage (closed_qa, information_extraction, summarization; CC BY-SA 3.0);
  * notes are single sentences of 10-40 words (like search snippets);
  * the answer is always built ONLY from the notes (the sentence itself, often with a citation marker like [1]);
  * example types: single, single_distractors, double, negative (honest refusal), negative_same_passage, plain (no notes),
    dolly_context.

Split: by TOPIC with the same rule as corpus.group_key/_bucket. A topic is validation if it was a validation topic of
the v2 dataset (bucket < VAL_FRACTION), so validation passages never appear in training and the original v2 validation
topics stay validation.

Usage (project root):
    python backend/data_tools/build_rag_dataset.py                       # writes backend/data/rag_v1 (must not exist)
    python backend/data_tools/build_rag_dataset.py --write-eval backend/eval/rag_eval.json   # ALSO the hold-out file
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "training"))
sys.path.insert(0, str(BACKEND_DIR / "data_tools"))

import corpus  # noqa: E402
from model.generation import build_prompt  # noqa: E402
from search.prompt import FOOTER, HEADER  # noqa: E402
from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402

RAW_V2 = BACKEND_DIR / "data" / "raw_v2"
OUT_DIR = BACKEND_DIR / "data" / "rag_v1"
MAX_TOKENS = 512
MIN_WORDS, MAX_WORDS = 10, 40
VAL_BUCKET = int(corpus.VAL_FRACTION * 10_000)
DOLLY_CATEGORIES = {"closed_qa", "information_extraction", "summarization"}

REFUSALS = [
    "The notes do not say.",
    "The notes don't say.",
    "I cannot find that in the notes.",
    "The notes do not contain this information.",
    "The notes do not answer this question.",
    "Not mentioned in the notes.",
]

MODE_WEIGHTS = [("domain", 0.55), ("english", 0.30), ("chat", 0.15)]
# (kind -> share of the examples made from Wikipedia passages)
TYPE_WEIGHTS = [("single", 0.25), ("single_distractors", 0.25), ("double", 0.20), ("negative", 0.13),
                ("negative_same_passage", 0.07), ("plain", 0.10)]
# Plain (no notes) examples keep the model's normal behaviour (about 15% of the total with the context-free Dolly Q/A).

DEF_Q = ["What is {t}?", "Tell me about {t}.", "Who or what is {t}?", "Explain {t} briefly.", "Can you describe {t}?",
         "What can you tell me about {t}?", "Give me a short description of {t}."]
OVERVIEW_Q = ["Give me two facts about {t}.", "Tell me more about {t}.", "What should I know about {t}?",
              "Give me a short overview of {t}.", "Share a couple of facts about {t}."]
YEAR_Q = ["What happened in {y} regarding {t}?", "What does the text say about {t} in {y}?",
          "What is mentioned about {y} in connection with {t}?", "Tell me about {t} in {y}."]
NUM_Q = ["What figures are mentioned about {t}?", "Which numbers are given for {t}?", "What numbers does the text give about {t}?"]
WHO_PATTERNS = [
    (r"\bfounded by\b", "Who founded {t}?"), (r"\bwritten by\b", "Who wrote {t}?"),
    (r"\bdirected by\b", "Who directed {t}?"), (r"\bdeveloped by\b", "Who developed {t}?"),
    (r"\bcreated by\b", "Who created {t}?"), (r"\binvented by\b", "Who invented {t}?"),
    (r"\bdiscovered by\b", "Who discovered {t}?"), (r"\bdesigned by\b", "Who designed {t}?"),
    (r"\bestablished by\b", "Who established {t}?"), (r"\bnamed after\b", "What is {t} named after?"),
    (r"\b(?:located|situated|based) in\b", "Where is {t} located?"),
]
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
NUM_RE = re.compile(r"\b\d[\d,\.]*\b")
ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "vs", "no", "inc", "ltd", "co", "mt", "gen", "col", "lt", "sen", "rep", "gov", "fig", "etc", "approx", "c", "ca"}
STOPWORDS = set("a an the of in on at to for and or but is are was were be been it its this that these those with as by from "
                "has have had not no do does did what who when where which how why tell me about give some more can you i my "
                "text notes say said mentioned regarding connection".split())


# ----------------------------------------------------------------- text helpers
def split_sentences(text):
    """Split into sentences without breaking on initials / common abbreviations (Dr. Smith, J. K. Rowling, U.S.)."""
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    out, start = [], 0
    for m in re.finditer(r"[.!?][\"')\]]*\s+(?=[A-Z0-9\"(])", text):
        piece = text[start:m.start() + 1].strip()
        last = re.findall(r"(\S+)$", text[start:m.start()])
        word = last[0].strip("(\"'").lower().rstrip(".") if last else ""
        if len(word) == 1 or word in ABBREV or re.fullmatch(r"(?:[a-z]\.)+[a-z]?", word + "."):
            continue
        out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def clean_sentence(s):
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def valid_note(s):
    n = len(s.split())
    return (MIN_WORDS <= n <= MAX_WORDS and s[0].isupper() and s[-1] in ".!" and not re.search(r"[<>\[\]{}]", s)
            and "\ufffd" not in s and "http" not in s and "Source:" not in s and "Topic:" not in s)


def passage_notes(text):
    """The usable one-sentence notes of a passage, in order (some neighbours are joined when both are short)."""
    notes = []
    for s in split_sentences(text):
        s = clean_sentence(s)
        if valid_note(s):
            notes.append(s)
    return notes


def content_words(text):
    return [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text) if w.lower() not in STOPWORDS and len(w) > 2]


def key_groups(answer_notes, question, limit=3):
    """Distinctive words of the answer notes that are not in the question: years/numbers first, then capitalised
    names, then long words. Each group has one keyword (the checking code accepts lists of alternatives)."""
    q = {w.lower() for w in content_words(question)}
    cands = []
    for note in answer_notes:
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", note)
        for i, w in enumerate(words):
            if w.lower() in STOPWORDS or w.lower() in q or len(w) < 3:
                continue
            rank = 0 if w[0].isdigit() else 1 if (w[0].isupper() and i > 0) else 2 if len(w) >= 7 else 9
            if rank < 9:
                cands.append((rank, -len(w), w))
    seen, groups = set(), []
    for _, _, w in sorted(cands):
        if w.lower() not in seen:
            seen.add(w.lower())
            groups.append([w])
        if len(groups) >= limit:
            break
    return groups


def evidence_text(notes, question):
    """The live prompt body. Same layout as search.prompt.build_evidence_text."""
    lines = [f"[{i + 1}] {n}" for i, n in enumerate(notes)]
    return f"{HEADER}\n" + "\n".join(lines) + f"\n{FOOTER} {question}"


def render(mode, body, answer):
    """Full training document text: mode template + answer (documents then become <BOS> text <EOS>)."""
    return f"{build_prompt(body, mode)} {answer}"


# ------------------------------------------------------------------ split
def is_val_topic(topic):
    return corpus._bucket(corpus.group_key({"text": f"Topic: {topic}\nSource: https://en.wikipedia.org/"})) < VAL_BUCKET


def load_wikipedia_passages(raw_dir=RAW_V2):
    """[{topic, notes, key}] for every Text: block of raw_v2/wikipedia_*.txt."""
    passages = []
    for path in sorted([*Path(raw_dir).glob("wikipedia_*.txt"), *Path(raw_dir).glob("kb_*.txt")]):    # kb_*.txt: exported from the knowledge base
        text = path.read_text(encoding="utf-8", errors="replace")
        for tag, body in corpus._BLOCK_RE.findall(text):
            m = re.search(r"^Topic: (.+)$", body, re.M)
            t = re.search(r"^Text: (.*?)(?=^Source: |\Z)", body, re.M | re.S)
            if not m or not t:
                continue
            topic = m.group(1).strip()
            notes = passage_notes(t.group(1))
            if len(notes) >= 2 and "<" not in topic and "\ufffd" not in topic:
                passages.append({"topic": topic, "notes": notes, "source": path.name,
                                 "val": is_val_topic(topic)})
    return passages


def find_dolly_file():
    for pattern in (os.path.expanduser("~/.cache/huggingface/hub/datasets--databricks--databricks-dolly-15k/snapshots/*/databricks-dolly-15k.jsonl"),
                    str(BACKEND_DIR / "data" / "raw_v2" / "databricks-dolly-15k.jsonl")):
        found = glob.glob(pattern)
        if found:
            return Path(found[0])
    return None


def overlap(answer, context):
    a = {w.lower() for w in content_words(answer)}
    c = {w.lower() for w in content_words(context)}
    return len(a & c) / len(a) if a else 0.0


def load_dolly_examples(path):
    """Dolly rows with a context: [{question, notes, answer, val}]. Only short, mostly extractive, clean ones."""
    from convert_open_datasets import acceptable, clean   # same filters as the context-free Dolly conversion
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["category"] not in DOLLY_CATEGORIES or not r["context"].strip():
            continue
        q, a, ctx = clean(r["instruction"]), clean(r["response"]), clean(r["context"]).replace("\n", " ")
        if not (8 <= len(q) <= 300 and 3 <= len(a.split()) <= 60 and len(ctx.split()) <= 3 * MAX_WORDS):
            continue
        if not acceptable(q, a, ctx) or "\n" in q or "\n" in a or re.search(r"[<>\[\]{}]", ctx + q + a):
            continue
        if overlap(a, ctx) < 0.6:
            continue
        sents, notes, cur = split_sentences(ctx), [], ""
        for s in sents:                                    # group sentences into notes of at most MAX_WORDS words
            if len(s.split()) > MAX_WORDS:
                notes = None
                break
            if cur and len((cur + " " + s).split()) > MAX_WORDS:
                notes.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if notes is None:
            continue
        if cur:
            notes.append(cur)
        if not 1 <= len(notes) <= 3 or any(len(n.split()) < 4 for n in notes):
            continue
        rows.append({"question": q, "notes": notes, "answer": a, "val": corpus._bucket(ctx) < VAL_BUCKET})
    return rows


# --------------------------------------------------------------- example makers
def _cite(rng, notes_used, idxs):
    """Answer text from chosen notes with citation markers, in one of several styles."""
    marks = [f"[{i + 1}]" for i in idxs]
    texts = [notes_used[i] for i in idxs]
    style = rng.choice(["end", "according", "from", "plain"])
    if len(texts) == 1:
        s, m = texts[0], marks[0]
        return {"end": f"{s} {m}", "according": f"According to {m}: {s}", "from": f"From the notes: {s} {m}", "plain": s}[style]
    if style == "according":
        return f"According to {' and '.join(marks)}: {' '.join(texts)}"
    if style == "plain":
        return " ".join(texts)
    return " ".join(f"{t} {m}" for t, m in zip(texts, marks))


def is_definition(topic, sentence):
    """True when the sentence starts with the topic and says what it is ("X is a ...", "X was ...", "X refers to ...")."""
    base = re.sub(r"\s*\(.*?\)", "", topic).strip()
    if not base or not re.match(re.escape(base), sentence, re.I):
        return False
    return re.search(r"(is|are|was|were|refers to|means)", sentence[len(base):len(base) + 60]) is not None


def _question_for(rng, passage, idx):
    """A question that note `idx` of the passage answers (or None)."""
    t, s = passage["topic"], passage["notes"][idx]
    options = []
    if is_definition(t, s):
        options.append(("def", rng.choice(DEF_Q).format(t=t)))
    y = YEAR_RE.findall(s)
    if y:
        options.append(("year", rng.choice(YEAR_Q).format(t=t, y=y[0])))
    for pat, q in WHO_PATTERNS:
        if re.search(pat, s):
            options.append(("who", q.format(t=t)))
            break
    nums = [n for n in NUM_RE.findall(s) if not YEAR_RE.fullmatch(n) and len(n.strip(".,")) >= 1]
    if nums:
        options.append(("num", rng.choice(NUM_Q).format(t=t)))
    return rng.choice(options) if options else None


def _distractors(rng, pool, topic, count):
    """Notes from OTHER topics (and not mentioning this topic)."""
    out, tries, low = [], 0, topic.lower()
    while len(out) < count and tries < 50:
        tries += 1
        other_topic, note = pool[rng.randrange(len(pool))]
        if other_topic.lower() != low and low not in note.lower() and note not in out:
            out.append(note)
    return out


def _place(rng, relevant, extra):
    """Shuffle relevant + distractor notes; return (notes, indexes of the relevant ones)."""
    items = [(n, True) for n in relevant] + [(n, False) for n in extra]
    rng.shuffle(items)
    return [n for n, _ in items], [i for i, (_, rel) in enumerate(items) if rel]


def make_example(rng, kind, passage, pool, mode):
    """One example dict or None. Keys: type, mode, topic, notes, question, answer, answerable."""
    notes_all, topic = passage["notes"], passage["topic"]
    if kind in ("single", "single_distractors"):
        idx = rng.randrange(len(notes_all))
        q = _question_for(rng, passage, idx)
        if q is None:
            return None
        extra = _distractors(rng, pool, topic, rng.choice([1, 2])) if kind == "single_distractors" else []
        notes, rel = _place(rng, [notes_all[idx]], extra)
        return dict(type=kind, mode=mode, topic=topic, notes=notes, question=q[1], answer=_cite(rng, notes, rel), answerable=True, sub=q[0])
    if kind == "double":
        i = rng.randrange(len(notes_all) - 1)
        if rng.random() < 0.5:
            j = i + 1
            q = rng.choice(DEF_Q if is_definition(topic, notes_all[i]) and rng.random() < 0.5 else OVERVIEW_Q).format(t=topic)
        else:
            j = rng.randrange(len(notes_all))
            if j == i:
                return None
            q = rng.choice(OVERVIEW_Q).format(t=topic)
        extra = _distractors(rng, pool, topic, 1) if rng.random() < 0.3 else []
        notes, rel = _place(rng, [notes_all[i], notes_all[j]], extra)
        if len(notes) > 3:
            return None
        return dict(type="double", mode=mode, topic=topic, notes=notes, question=q, answer=_cite(rng, notes, rel), answerable=True, sub="two")
    if kind == "negative":
        idx = rng.randrange(len(notes_all))
        q = _question_for(rng, passage, idx)
        if q is None:
            return None
        notes = _distractors(rng, pool, topic, rng.choice([1, 2, 3]))
        if not notes:
            return None
        return dict(type="negative", mode=mode, topic=topic, notes=notes, question=q[1], answer=rng.choice(REFUSALS), answerable=False, sub=q[0])
    if kind == "negative_same_passage":
        idx = rng.randrange(len(notes_all))
        years = YEAR_RE.findall(notes_all[idx])
        if not years:
            return None
        year = years[0]
        others = [n for k, n in enumerate(notes_all) if k != idx and year not in n]
        if not others:
            return None
        rng.shuffle(others)
        notes = others[:rng.choice([1, 2])]
        q = rng.choice(YEAR_Q).format(t=topic, y=year)
        return dict(type="negative_same_passage", mode=mode, topic=topic, notes=notes, question=q, answer=rng.choice(REFUSALS), answerable=False, sub="year")
    if kind == "plain":
        first = next((n for n in notes_all if is_definition(topic, n)), None)
        if first is not None and rng.random() < 0.6:
            return dict(type="plain", mode=mode, topic=topic, notes=[], question=rng.choice(DEF_Q).format(t=topic), answer=first,
                        answerable=True, sub="def")
        idx = rng.randrange(len(notes_all))
        q = _question_for(rng, passage, idx)
        if q is None:
            return None
        return dict(type="plain", mode=mode, topic=topic, notes=[], question=q[1], answer=notes_all[idx], answerable=True, sub=q[0])
    raise ValueError(kind)


def example_text(ex):
    body = evidence_text(ex["notes"], ex["question"]) if ex["notes"] else ex["question"]
    return render(ex["mode"], body, ex["answer"])


def pick(rng, weighted):
    r, acc = rng.random(), 0.0
    for name, w in weighted:
        acc += w
        if r < acc:
            return name
    return weighted[-1][0]


def build_split(passages, dolly, plain_qa, rng, target, dolly_share=0.12):
    """Make about `target` examples from the passages of ONE split (so topics never cross splits)."""
    pool = [(p["topic"], n) for p in passages for n in p["notes"]]
    examples, attempts = [], 0
    dolly = list(dolly)
    rng.shuffle(dolly)
    n_dolly = min(len(dolly), int(target * dolly_share))
    for row in dolly[:n_dolly]:
        mode = pick(rng, MODE_WEIGHTS[:2])
        examples.append(dict(type="dolly_context", mode=mode, topic="dolly", notes=row["notes"], question=row["question"],
                             answer=row["answer"], answerable=True, sub="dolly"))
    plain_extra = plain_qa[:]
    rng.shuffle(plain_extra)
    for text in plain_extra[:int(target * 0.05)]:               # plain Q/A without notes (context-free Dolly)
        q, a = text
        examples.append(dict(type="plain", mode="english", topic="dolly", notes=[], question=q, answer=a, answerable=True, sub="qa"))
    while len(examples) < target and attempts < target * 40 and passages:
        kind, mode = pick(rng, TYPE_WEIGHTS), pick(rng, MODE_WEIGHTS)
        for _ in range(40):                        # keep the chosen type (some passages cannot make some types): try others
            attempts += 1
            ex = make_example(rng, kind, passages[rng.randrange(len(passages))], pool, mode)
            if ex:
                examples.append(ex)
                break
    rng.shuffle(examples)
    return examples


def load_plain_qa(raw_dir=RAW_V2):
    """(question, answer, is_val) from the context-free Dolly Q/A blocks already on disk."""
    path = Path(raw_dir) / "dolly_qa.txt"
    out = []
    if path.is_file():
        for tag, body in corpus._BLOCK_RE.findall(path.read_text(encoding="utf-8", errors="replace")):
            m = re.match(r"Question: (.*)\nAnswer: (.*)", body, re.S)
            if m and "\n" not in m.group(1) and len(m.group(2).split()) <= 60 and "\n" not in m.group(2):
                out.append((m.group(1).strip(), m.group(2).strip(), corpus._bucket(f"<{tag}>\n{body.strip()}") < VAL_BUCKET))
    return out


# ------------------------------------------------------------------ tokens
def encode_examples(examples, tok, max_tokens=MAX_TOKENS):
    """(ids, kept examples, lengths, dropped). Every example is <BOS> ... <EOS>; longer than max_tokens -> dropped."""
    ids, kept, lengths, dropped = [], [], [], 0
    for ex in examples:
        enc = tok.encode(example_text(ex), add_bos=True, add_eos=True)
        if len(enc) > max_tokens:
            dropped += 1
            continue
        ids.extend(enc)
        kept.append(ex)
        lengths.append(len(enc))
    return ids, kept, lengths, dropped


def length_histogram(lengths):
    bins = [0, 64, 128, 192, 256, 320, 384, 448, 512]
    hist = {}
    for lo, hi in zip(bins, bins[1:]):
        hist[f"{lo + 1}-{hi}"] = int(sum(lo < n <= hi for n in lengths))
    return hist


def make_eval_cases(passages, dolly_unused, rng, n_total=60):
    """Hold-out cases from VALIDATION topics: notes + question + key facts (and 'not in notes' cases)."""
    pool = [(p["topic"], n) for p in passages for n in p["notes"]]
    cases, seen_topics, tries = [], set(), 0
    n_neg = max(1, n_total // 5)
    order = passages[:]
    rng.shuffle(order)
    for p in order:
        if len(cases) >= n_total or tries > 50 * n_total:
            break
        tries += 1
        if p["topic"] in seen_topics:
            continue
        want_neg = sum(not c["answerable"] for c in cases) < n_neg and len(cases) % 5 == 4
        kind = "negative" if want_neg else rng.choice(["single", "single_distractors", "double"])
        ex = make_example(rng, kind, p, pool, rng.choice(["domain", "english"]))
        if not ex:
            continue
        seen_topics.add(p["topic"])
        rel_notes = [n for n in ex["notes"]]
        groups = key_groups(rel_notes if ex["answerable"] else [], ex["question"]) if ex["answerable"] else []
        if ex["answerable"]:
            # only the notes that really carry the answer count for the key facts
            carrying = [n for n in ex["notes"] if n in ex["answer"]]
            groups = key_groups(carrying, ex["question"])
            if not groups:
                continue
        cases.append({"id": len(cases) + 1, "type": ex["type"], "mode": ex["mode"], "topic": ex["topic"], "notes": ex["notes"],
                      "question": ex["question"], "answerable": ex["answerable"], "groups": groups,
                      "reference_answer": ex["answer"]})
    return cases


def build(out_dir, tok, seed=0, train_target=24000, val_target=1200, raw_dir=RAW_V2, dolly_file="auto", max_tokens=MAX_TOKENS,
          write_eval=None, eval_cases=60):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; refusing to overwrite. Choose a new folder.")
    rng = random.Random(seed)
    passages = load_wikipedia_passages(raw_dir)
    train_p = [p for p in passages if not p["val"]]
    val_p = [p for p in passages if p["val"]]
    topics_train, topics_val = {p["topic"].lower() for p in train_p}, {p["topic"].lower() for p in val_p}
    if topics_train & topics_val:
        raise SystemExit("Internal error: a topic is in both splits.")
    dolly_path = find_dolly_file() if dolly_file == "auto" else (Path(dolly_file) if dolly_file else None)
    dolly = load_dolly_examples(dolly_path) if dolly_path and Path(dolly_path).is_file() else []
    plain = load_plain_qa(raw_dir)
    splits = {}
    for name, ps, target, is_val in (("train", train_p, train_target, False), ("val", val_p, val_target, True)):
        dl = [d for d in dolly if d["val"] == is_val]
        pl = [(q, a) for q, a, v in plain if v == is_val]
        examples = build_split(ps, dl, pl, random.Random(f"{seed}-{name}"), target)
        splits[name] = examples
    meta = {
        "tokenizer_sha256": tok.sha256, "vocab_size": tok.vocab_size, "format": "rag_v1",
        "format_description": ("Each document is <BOS> + mode template + ' ' + answer + <EOS>. Mode body = " + repr(HEADER) +
                               " + lines '[n] note' + '" + FOOTER + " question' (or just the question for 'plain' examples)."),
        "seed": seed, "max_tokens_per_example": max_tokens, "refusal_phrases": REFUSALS,
        "split_rule": f"topic bucket < {VAL_BUCKET} (corpus._bucket of corpus.group_key) = validation = the original v2 validation topics",
        "passages": {"train": len(train_p), "val": len(val_p)}, "topics": {"train": len(topics_train), "val": len(topics_val)},
        "dolly": {"file": str(dolly_path) if dolly else None, "usable_examples": len(dolly), "licence": "CC BY-SA 3.0 (databricks-dolly-15k)"},
    }
    out_dir.mkdir(parents=True)
    for name, examples in splits.items():
        ids, kept, lengths, dropped = encode_examples(examples, tok, max_tokens)
        arr = np.asarray(ids, dtype=np.uint16)
        if arr.size and int(arr.max()) >= tok.vocab_size:
            raise SystemExit("Token id outside the vocabulary.")
        arr.tofile(out_dir / f"{name}.bin")
        lens = np.asarray(lengths)
        meta[name] = {
            "documents": len(kept), "tokens": int(arr.size), "dropped_too_long": dropped,
            "unk_tokens": int((arr == tok.unk_id).sum()),
            "examples_by_type": dict(Counter(e["type"] for e in kept)),
            "examples_by_mode": dict(Counter(e["mode"] for e in kept)),
            "length_tokens": {"mean": round(float(lens.mean()), 1), "median": int(np.median(lens)), "max": int(lens.max()),
                              "fit_in_128": int((lens <= 128).sum()), "fit_in_128_percent": round(100 * float((lens <= 128).mean()), 1)},
            "length_histogram": length_histogram(lengths),
        }
        (out_dir / f"{name}_examples.jsonl").write_text(
            "\n".join(json.dumps({k: e[k] for k in ("type", "mode", "topic", "question", "answer")}, ensure_ascii=False) for e in kept[:300]),
            encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if write_eval:
        write_eval = Path(write_eval)
        if write_eval.exists():
            raise SystemExit(f"{write_eval} already exists; refusing to overwrite.")
        cases = make_eval_cases(val_p, None, random.Random(f"{seed}-eval"), eval_cases)
        write_eval.write_text(json.dumps({"description": "ELVARN RAG hold-out: notes + question; passages come only from validation topics.",
                                          "cases": cases}, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta, splits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="NEW folder (must not exist)")
    ap.add_argument("--tokenizer-file", type=Path, default=BACKEND_DIR / "data" / "v2" / "tokenizer" / "tokenizer.json")
    ap.add_argument("--raw-dir", type=Path, default=RAW_V2)
    ap.add_argument("--train-examples", type=int, default=24000)
    ap.add_argument("--val-examples", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dolly-file", default="auto", help="'auto' = look in the local Hugging Face cache; '' = skip Dolly")
    ap.add_argument("--write-eval", type=Path, default=None, help="also write the hold-out evaluation file (new file only)")
    ap.add_argument("--eval-cases", type=int, default=60)
    args = ap.parse_args(argv)
    tok = BPETokenizer(args.tokenizer_file)
    meta, splits = build(args.out_dir, tok, args.seed, args.train_examples, args.val_examples, args.raw_dir,
                         args.dolly_file or None, write_eval=args.write_eval, eval_cases=args.eval_cases)
    print(json.dumps(meta, indent=2))
    return meta, splits


if __name__ == "__main__":
    main()
