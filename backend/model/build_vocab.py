import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

VOCAB_FILE = BASE_DIR / "data" / "vocab" / "vocab.txt"
OUTPUT_FILE = BASE_DIR / "tokenizer" / "vocab.json"


SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]


def build_vocab():

    if not VOCAB_FILE.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found:\n{VOCAB_FILE}"
        )

    vocab = {}

    # Special tokens first
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    with open(VOCAB_FILE, "r", encoding="utf-8") as file:

        for line in file:

            token = line.strip()

            if not token:
                continue

            # Remove accidental [L123] line markers
            if token.startswith("[L") and "]" in token:
                token = token.split("]", 1)[1].strip()

            if not token:
                continue

            # Avoid duplicate tokens
            if token not in vocab:
                vocab[token] = len(vocab)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            vocab,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("Vocabulary built successfully.")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_vocab()