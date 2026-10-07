import sys
from pathlib import Path


# =========================================================
# PATH SETUP
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BACKEND_DIR))


# =========================================================
# IMPORT TOKENIZER
# =========================================================

from tokenizer.tokenizer import (
    tokenizer,
    encode,
    vocab
)


# =========================================================
# TEST PROMPT
# =========================================================

text = "The case involved"


# =========================================================
# TOKENIZE
# =========================================================

tokens = tokenizer(text)

token_ids = encode(text)


# =========================================================
# DISPLAY
# =========================================================

print("===================================")
print("TOKENIZER DIAGNOSTIC")
print("===================================")

print("\nInput:")
print(text)

print("\nTokens:")

for token in tokens:
    print(
        repr(token),
        "->",
        vocab.get(token, vocab["<UNK>"])
    )


print("\nToken IDs:")
print(token_ids)


# =========================================================
# UNKNOWN TOKENS
# =========================================================

unknown_tokens = []

for token, token_id in zip(tokens, token_ids):

    if token_id == vocab["<UNK>"]:
        unknown_tokens.append(token)


print("\nUnknown tokens:")

if unknown_tokens:

    for token in unknown_tokens:
        print(repr(token))

else:

    print("NONE")


# =========================================================
# CHECK VOCAB DIRECTLY
# =========================================================

print("\n===================================")
print("DIRECT VOCABULARY CHECK")
print("===================================")

for word in ["The", "case", "involved"]:

    print(
        repr(word),
        "exists:",
        word in vocab,
        "ID:",
        vocab.get(word, "NOT FOUND")
    )


print("\n===================================")
print("DIAGNOSTIC COMPLETE")
print("===================================")