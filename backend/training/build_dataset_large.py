"""Tokenize a LARGE text corpus (hundreds of MB) into train.bin / val.bin, a document at a time, without holding the tokens as Python numbers.

build_dataset.py keeps every token id in one Python list (about 30 bytes per token): 135 million tokens would need more than 4 GB of memory. This builder
encodes in batches and writes the 16-bit ids straight to disk, so the memory needed is the size of the biggest text file plus a few hundred MB.

Same output as build_dataset.py (train.bin, val.bin, meta.json with the tokenizer fingerprint), so continue_train.py reads it unchanged. Differences:
  * a document goes to validation when its fingerprint says so (1 in --val-every, default 100), the same way on every run, and only ever to ONE side;
  * exact duplicates are dropped (the first copy is kept); the near-duplicate pass of build_dataset.py is not run (it cannot handle 350,000 documents).

Usage (project root):
    python backend/training/build_dataset_large.py --raw-dir backend/data/raw_v2 --out-dir backend/data/lm_legal_v1 \\
        --tokenizer-file backend/data/v2/tokenizer/tokenizer.json
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "training"))

from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402
import corpus  # noqa: E402

BATCH = 2000


def documents_of(path):
    """The documents of one text file, as the project's pipeline reads them: tagged blocks, or plain paragraphs."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = corpus._BLOCK_RE.findall(text)
    if blocks:
        for tag, body in blocks:
            yield tag, f"<{tag}>\n{body.strip()}"
    else:
        for body in corpus._split_plain_text(text):
            yield "PLAIN", body


def is_validation(text, every):
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % every == 0


def build(raw_dir, out_dir, tokenizer, val_every=100, say=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {name: open(out_dir / f"{name}.bin", "wb") for name in ("train", "val")}
    stats = {name: {"documents": 0, "tokens": 0, "unk_tokens": 0, "tokens_by_category": Counter()} for name in ("train", "val")}
    seen, duplicates = set(), 0
    pending = {"train": [], "val": []}
    raw = tokenizer._tok
    bos, eos, unk = tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id

    def flush(name):
        batch = pending[name]
        if not batch:
            return
        for (category, _), enc in zip(batch, raw.encode_batch([t for _, t in batch])):
            ids = np.asarray([bos] + enc.ids + [eos], dtype=np.int64)
            if ids.max() >= tokenizer.vocab_size:
                raise SystemExit("Token id outside vocabulary range.")
            handles[name].write(ids.astype(np.uint16).tobytes())
            s = stats[name]
            s["documents"] += 1
            s["tokens"] += int(ids.size)
            s["unk_tokens"] += int((ids == unk).sum())
            s["tokens_by_category"][category] += int(ids.size)
        batch.clear()

    files = corpus.raw_files(raw_dir)
    for path in files:
        say(f"reading {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
        for category, text in documents_of(path):
            key = hashlib.sha1(text.encode("utf-8")).digest()[:12]
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            name = "val" if is_validation(text, val_every) else "train"
            pending[name].append((category, text))
            if len(pending[name]) >= BATCH:
                flush(name)
        for name in pending:
            flush(name)
    for h in handles.values():
        h.close()
    meta = {"tokenizer_sha256": tokenizer.sha256, "vocab_size": tokenizer.vocab_size, "files": [p.name for p in files], "group_by_topic": False,
            "split_stats": {"exact_duplicates_removed": duplicates, "val_every": val_every}}
    for name, s in stats.items():
        meta[name] = {"documents": s["documents"], "tokens": s["tokens"], "unk_tokens": s["unk_tokens"],
                      "unk_percent": round(100 * s["unk_tokens"] / max(s["tokens"], 1), 4), "tokens_by_category": dict(s["tokens_by_category"])}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="NEW folder (must not exist)")
    ap.add_argument("--tokenizer-file", type=Path, default=BACKEND_DIR / "data" / "v2" / "tokenizer" / "tokenizer.json")
    ap.add_argument("--val-every", type=int, default=100, help="1 document in this many goes to validation")
    args = ap.parse_args(argv)
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} already exists; refusing to overwrite. Choose a new folder.")
    tokenizer = BPETokenizer(args.tokenizer_file)
    if tokenizer.vocab_size > 65535:
        raise SystemExit("Vocabulary too large for uint16 storage.")
    meta = build(args.raw_dir, args.out_dir, tokenizer, args.val_every)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
