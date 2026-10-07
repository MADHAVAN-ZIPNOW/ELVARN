import json
from pathlib import Path


# =========================================================
# PATH SETUP
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent

VOCAB_FILE = BACKEND_DIR / "tokenizer" / "vocab.json"
TRAIN_FILE = BACKEND_DIR / "data" / "train.txt"


# =========================================================
# SPECIAL TOKENS
# =========================================================

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]


# =========================================================
# LOAD EXISTING VOCABULARY
# =========================================================

with open(VOCAB_FILE, "r", encoding="utf-8") as file:
    old_vocab = json.load(file)


print("===================================")
print("REBUILDING VOCABULARY")
print("===================================")

print("\nOld vocabulary size:")
print(len(old_vocab))


# =========================================================
# LOAD TRAINING CORPUS
# =========================================================

with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


print("\nTraining characters:")
print(len(text))


# =========================================================
# BASIC WORD / PUNCTUATION TOKENIZATION
# =========================================================

import re

corpus_tokens = re.findall(
    r"\w+|[^\w\s]",
    text,
    re.UNICODE
)


corpus_token_set = set(
    corpus_tokens
)


print("\nCorpus unique tokens:")
print(len(corpus_token_set))


# =========================================================
# BUILD NEW VOCABULARY
# =========================================================

combined_tokens = []


# ---------------------------------------------------------
# SPECIAL TOKENS FIRST
# ---------------------------------------------------------

for token in SPECIAL_TOKENS:

    if token not in combined_tokens:

        combined_tokens.append(token)


# ---------------------------------------------------------
# PRESERVE EXISTING DOMAIN VOCABULARY
# ---------------------------------------------------------

for token in old_vocab:

    if token in SPECIAL_TOKENS:

        continue

    if token not in combined_tokens:

        combined_tokens.append(token)


# ---------------------------------------------------------
# ADD CORPUS TOKENS
# ---------------------------------------------------------

added_tokens = []

for token in corpus_tokens:

    if token in SPECIAL_TOKENS:

        continue

    if token not in combined_tokens:

        combined_tokens.append(token)

        added_tokens.append(token)


# =========================================================
# CREATE NEW ID MAPPING
# =========================================================

new_vocab = {
    token: token_id
    for token_id, token in enumerate(combined_tokens)
}


# =========================================================
# SAVE
# =========================================================

with open(
    VOCAB_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        new_vocab,
        file,
        ensure_ascii=False,
        indent=2
    )


# =========================================================
# RESULTS
# =========================================================

print("\n===================================")
print("VOCABULARY REBUILT")
print("===================================")

print("\nOld vocabulary size:")
print(len(old_vocab))

print("\nNew vocabulary size:")
print(len(new_vocab))

print("\nVocabulary added:")
print(len(added_tokens))


print("\nSpecial tokens:")

for token in SPECIAL_TOKENS:

    print(
        new_vocab[token],
        "->",
        token
    )


print("\nAdded tokens:")

if added_tokens:

    for token in added_tokens:

        print(
            repr(token),
            "->",
            new_vocab[token]
        )

else:

    print("NONE")


print("\nSaved to:")
print(VOCAB_FILE)