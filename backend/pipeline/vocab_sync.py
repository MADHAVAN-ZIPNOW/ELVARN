"""The vocabulary grows with the data: new words from the knowledge base are added to the MAIN vocabulary file, and the new text is stored for training.

Whenever data collection runs (the sync program starts with it), this step looks for NEW passage files in the corpus and does two things:

  1. VOCABULARY. It counts the words of the new passages (all topics, all languages that use Latin letters) and adds every word seen at least MIN_COUNT (100) times
     that still costs 2 or more tokens, as ONE token. On its first run it also adds one token for every Tamil, Hindi and punctuation character (curly quotes,
     dashes, section sign, rupee). There is no cap, except the one the training data format sets: token ids are stored as 16-bit numbers, so the vocabulary
     stops at 65,535 tokens.
       main file     backend/data/v2/tokenizer/tokenizer.json            the vocabulary every dataset builder uses; it only GROWS (new tokens at the end,
                                                                         old ids never move; see extend_vocab.py)
       version log   backend/data/v2/tokenizer/vocab_versions.jsonl      one line per growth: date, size before / after, tokens added, fingerprint
       original      backend/data/v2/tokenizer/tokenizer.original.json   the 16,384 vocabulary as it was before the first growth
     A model trained with an older version is made to fit the newer one automatically when training starts (continue_train.py): each new token starts as the
     average of the two pieces it joins. The live (deployed) model keeps the tokenizer saved next to it, so it keeps working untouched.

  2. TRAINING TEXT. Court judgments and court records are appended as blocks to backend/data/raw_v2/kb_judgments.txt, in the block format the project's
     pipeline already reads (the RAG builder and the language-model builder both pick up kb_*.txt). Training is NOT started here: you start it yourself, and it
     reads that file with the current vocabulary.

    python -m backend.pipeline.vocab_sync            # one pass now
"""
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "training"))

import extend_vocab  # noqa: E402
from backend.pipeline.registry import STATE_DIR  # noqa: E402

CORPUS = BACKEND / "data" / "corpus"
MAIN_TOKENIZER = BACKEND / "data" / "v2" / "tokenizer" / "tokenizer.json"
TEXT_OUT = BACKEND / "data" / "raw_v2" / "kb_judgments.txt"
STATE_DB = STATE_DIR / "vocab_sync.sqlite"
MIN_COUNT = 100                  # a word must be seen this often before it gets its own token (20 would fill the whole 65,535 vocabulary in one pass)
TEXT_TOPICS = ("judgment", "court_record")
MIN_CHARS, MAX_CHARS = 120, 1800
SCHEMA = """
CREATE TABLE IF NOT EXISTS files (name TEXT PRIMARY KEY, at TEXT);
CREATE TABLE IF NOT EXISTS words (word TEXT PRIMARY KEY, n INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS added (word TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""
_SENTENCE = re.compile(r"(?<=[.;:!?])\s+")


def versions_path(tokenizer):
    return Path(tokenizer).parent / "vocab_versions.jsonl"


def blocks_of(title, text, source):
    """The passage as training blocks of at most MAX_CHARS characters (cut at sentence ends), in the project's block format."""
    from backend.knowledge.export_training import block, clean
    text = clean(text)
    out, cur = [], ""
    for sentence in _SENTENCE.split(text):
        if cur and len(cur) + len(sentence) + 1 > MAX_CHARS:
            out.append(cur)
            cur = ""
        cur = (cur + " " + sentence).strip() if cur else sentence[:MAX_CHARS]
    if cur:
        out.append(cur)
    topic = clean(title or "Court judgment")[:200]
    return [block(topic, chunk, source) for chunk in out if len(chunk) >= MIN_CHARS]


def passage_files(corpus_dir):
    return sorted(Path(corpus_dir).glob("passages/**/*.parquet"))


def latest_version(tokenizer=MAIN_TOKENIZER):
    """The last line of the version log (None before the first growth)."""
    path = versions_path(tokenizer)
    if not path.is_file():
        return None
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def run_once(corpus_dir=CORPUS, tokenizer=MAIN_TOKENIZER, text_out=TEXT_OUT, state_db=STATE_DB, min_count=MIN_COUNT, max_vocab=extend_vocab.MAX_VOCAB,
             scripts=("tamil", "devanagari", "punct"), say=print):
    """One pass. -> {"new_files", "new_tokens", "vocab_size", "blocks"}; changes nothing when no new passage file has appeared."""
    import collections
    import pyarrow.parquet as pq
    from backend.pipeline.process import looks_garbled
    from tokenizers import Tokenizer
    tokenizer, text_out = Path(tokenizer), Path(text_out)
    Path(state_db).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(state_db), timeout=30)
    db.executescript(SCHEMA)
    done = {r[0] for r in db.execute("SELECT name FROM files")}
    files = [f for f in passage_files(corpus_dir) if f.relative_to(corpus_dir).as_posix() not in done]
    if not files:
        db.close()
        return {"new_files": 0, "new_tokens": 0, "blocks": 0, "vocab_size": len(json.loads(tokenizer.read_text(encoding="utf-8"))["model"]["vocab"])}
    text_out.parent.mkdir(parents=True, exist_ok=True)
    blocks_written = 0
    for f in files:
        counter, blocks = collections.Counter(), []
        for row in pq.read_table(f, columns=["text", "title", "source_url", "topic"]).to_pylist():
            text = row["text"] or ""
            counter.update(extend_vocab.WORD.findall(text))
            if row["topic"] in TEXT_TOPICS and not looks_garbled(text):
                blocks += blocks_of(row["title"], text, row["source_url"])
        if blocks:
            with open(text_out, "a", encoding="utf-8") as out:
                out.write("\n\n".join(blocks) + "\n\n")
        blocks_written += len(blocks)
        db.executemany("INSERT INTO words (word, n) VALUES (?, ?) ON CONFLICT(word) DO UPDATE SET n = n + excluded.n", list(counter.items()))
        db.execute("INSERT OR REPLACE INTO files (name, at) VALUES (?, ?)", (f.relative_to(corpus_dir).as_posix(), datetime.now(timezone.utc).isoformat(timespec="seconds")))
        db.commit()
    # which words are now frequent enough, and still cost 2 or more tokens
    base_json = json.loads(tokenizer.read_text(encoding="utf-8"))
    base = Tokenizer.from_str(json.dumps(base_json))
    added = {r[0] for r in db.execute("SELECT word FROM added")}
    frequent = collections.Counter({w: n for w, n in db.execute("SELECT word, n FROM words WHERE n >= ?", (min_count,)) if w not in added})
    words = extend_vocab.pick_words(frequent, base, 10 ** 9, min_count)
    first = db.execute("SELECT v FROM meta WHERE k = 'scripts'").fetchone() is None
    chars = extend_vocab.script_characters(list(scripts)) if first else []
    size = len(base_json["model"]["vocab"])
    new_json, steps = base_json, []
    while True:                                                                  # no cap of our own; only the 16-bit id limit, kept by dropping the least useful words
        new_json, steps = extend_vocab.extend_tokenizer(base_json, words, base, chars)
        if size + len(steps) <= max_vocab or not words:
            break
        words = words[:max(0, int(len(words) * (max_vocab - size) / max(len(steps), 1)) - 1)]
    if steps and size + len(steps) <= max_vocab:
        original = tokenizer.with_name("tokenizer.original.json")
        if not original.exists():
            original.write_text(tokenizer.read_text(encoding="utf-8"), encoding="utf-8")
        tmp = tokenizer.with_suffix(".part")
        tmp.write_text(json.dumps(new_json, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(tokenizer)                                                   # a half-written vocabulary is never visible
        from tokenizer.bpe_tokenizer import BPETokenizer
        fingerprint = BPETokenizer(tokenizer).sha256
        previous = latest_version(tokenizer)
        entry = {"version": (previous["version"] + 1) if previous else 1, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "vocab_before": size,
                 "vocab_after": size + len(steps), "new_tokens": len(steps), "words_added": len(words), "characters_added": len(chars), "sha256": fingerprint,
                 "new_files": len(files)}
        with open(versions_path(tokenizer), "a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")
        meta = tokenizer.with_name("tokenizer_meta.json")
        if meta.is_file():
            info = json.loads(meta.read_text(encoding="utf-8"))
            info.update({"vocab_size": entry["vocab_after"], "sha256": fingerprint})
            meta.write_text(json.dumps(info, indent=2), encoding="utf-8")
        db.executemany("INSERT OR IGNORE INTO added (word) VALUES (?)", [(w,) for w in words])
        db.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('scripts', ?)", (",".join(scripts),))
        db.commit()
        say(f"vocabulary grew: {size:,} -> {size + len(steps):,} tokens ({len(words):,} words, {len(chars)} characters); {blocks_written:,} training blocks added")
    else:
        steps = []
        if first:
            db.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('scripts', ?)", (",".join(scripts),))
            db.commit()
    db.close()
    return {"new_files": len(files), "new_tokens": len(steps), "blocks": blocks_written, "vocab_size": size + len(steps)}


def run_forever(interval=60, say=lambda m: print(m, flush=True)):
    while True:
        try:
            run_once(say=say)
        except Exception as exc:  # noqa: BLE001 - a failed pass is tried again next time; nothing was half-written
            say(f"vocab sync skipped: {exc.__class__.__name__}: {str(exc)[:160]}")
        time.sleep(interval)


if __name__ == "__main__":
    print(json.dumps(run_once(), indent=2))
