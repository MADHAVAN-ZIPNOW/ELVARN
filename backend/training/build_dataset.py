"""Tokenize the train / validation documents into compact binary files.

Usage (from the project root):
    python backend/training/build_dataset.py

Reads  : backend/data/raw/*.txt|md  (same source as prepare_corpus.py)
Needs  : backend/tokenizer/bpe/tokenizer.json  (run train_bpe.py first)
Writes : backend/data/processed/train.bin, val.bin, meta.json

Each document becomes  <BOS> tokens <EOS>  and documents are concatenated
inside their own split only, so nothing crosses from train into validation.
"""
import argparse
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

PROCESSED_DIR = BACKEND_DIR / "data" / "processed"


def encode_documents(docs, tokenizer):
    ids, per_category, unk = [], Counter(), 0
    for doc in docs:
        doc_ids = tokenizer.encode(doc["text"], add_bos=True, add_eos=True)
        unk += doc_ids.count(tokenizer.unk_id)
        per_category[doc["category"]] += len(doc_ids)
        ids.extend(doc_ids)
    return ids, per_category, unk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=corpus.RAW_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR, help="where train.bin / val.bin / meta.json are written")
    ap.add_argument("--tokenizer-file", type=Path, default=None, help="tokenizer.json to use (default: the project tokenizer)")
    ap.add_argument("--group-by-topic", action="store_true", help="keep all blocks of one Topic on the same side of the split")
    args = ap.parse_args()
    out_dir = args.out_dir
    tokenizer = BPETokenizer(args.tokenizer_file) if args.tokenizer_file else BPETokenizer()
    if tokenizer.vocab_size > 65535:
        raise SystemExit("Vocabulary too large for uint16 storage.")

    documents = corpus.load_documents(args.raw_dir)
    train_docs, val_docs, stats = corpus.split_documents(documents, args.group_by_topic)
    if not train_docs or not val_docs:
        raise SystemExit(
            f"Empty split: {len(train_docs)} train / {len(val_docs)} val documents."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "tokenizer_sha256": tokenizer.sha256,
        "vocab_size": tokenizer.vocab_size,
        "files": [p.name for p in corpus.raw_files(args.raw_dir)],
        "group_by_topic": args.group_by_topic,
        "split_stats": stats,
    }

    for name, docs in (("train", train_docs), ("val", val_docs)):
        ids, per_category, unk = encode_documents(docs, tokenizer)
        arr = np.asarray(ids, dtype=np.uint16)
        if arr.size and int(arr.max()) >= tokenizer.vocab_size:
            raise SystemExit("Token id outside vocabulary range.")
        arr.tofile(out_dir / f"{name}.bin")
        meta[name] = {
            "documents": len(docs),
            "tokens": int(arr.size),
            "unk_tokens": unk,
            "unk_percent": round(100 * unk / max(arr.size, 1), 4),
            "tokens_by_category": dict(per_category),
        }

    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
