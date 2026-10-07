import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from tokenizer.tokenizer import tokenizer, encode, vocab

DATA_FILE = BACKEND_DIR / "data" / "train.txt"

with open(DATA_FILE, "r", encoding="utf-8") as file:
    text = file.read()

tokens = tokenizer(text)
token_ids = encode(text)

unknown_tokens = []

for token, token_id in zip(tokens, token_ids):

    if token_id == vocab["<UNK>"]:
        unknown_tokens.append(token)


print("Total tokens:", len(tokens))
print("Unknown tokens:", len(unknown_tokens))

unknown_percentage = (
    len(unknown_tokens) / len(tokens) * 100
    if tokens
    else 0
)

print(
    f"Unknown percentage: {unknown_percentage:.2f}%"
)

print("\nUnknown token list:")

for token in unknown_tokens:
    print(repr(token))