"""Train the fixed ELVARN BPE tokenizer on the TRAINING documents only.

Usage (from the project root):
    python backend/tokenizer/train_bpe.py [--vocab-size 4096]

Output: backend/tokenizer/bpe/tokenizer.json and tokenizer_meta.json.
Trained on training documents only, so the validation set stays unseen.
"""
import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

BACKEND_DIR = Path(__file__).resolve().parent.parent
# Running this file puts tokenizer/ first on sys.path, where the legacy
# tokenizer.py would shadow the `tokenizer` package. Replace it.
sys.path[0] = str(BACKEND_DIR)
sys.path.insert(1, str(BACKEND_DIR / "training"))

from tokenizer.bpe_tokenizer import BPE_DIR, SPECIAL_TOKENS, TOKENIZER_FILE, BPETokenizer  # noqa: E402
import corpus  # noqa: E402


def train(texts, vocab_size, out_file=TOKENIZER_FILE):
    tok = Tokenizer(models.BPE(unk_token="<UNK>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_file))
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=4096)
    args = parser.parse_args()

    documents = corpus.load_documents()
    train_docs, val_docs, stats = corpus.split_documents(documents)
    if not train_docs:
        raise SystemExit("No training documents found in " + str(corpus.RAW_DATA_DIR))

    out = train((d["text"] for d in train_docs), args.vocab_size)
    tokenizer = BPETokenizer(out)

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "requested_vocab_size": args.vocab_size,
        "special_tokens": {t: tokenizer.token_id(t) for t in SPECIAL_TOKENS},
        "sha256": tokenizer.sha256,
        "trained_on_documents": len(train_docs),
        "split_stats": stats,
        "library": "tokenizers (byte-level BPE)",
    }
    (BPE_DIR / "tokenizer_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("Saved:", out)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Special tokens:", meta["special_tokens"])
    print("sha256:", tokenizer.sha256)


if __name__ == "__main__":
    main()
