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

from tokenizer.tokenizer import encode


# =========================================================
# SETTINGS
# =========================================================

DATA_FILE = BACKEND_DIR / "data" / "train.txt"

SEQUENCE_LENGTH = 128


# =========================================================
# LOAD TEXT
# =========================================================

with open(DATA_FILE, "r", encoding="utf-8") as file:
    text = file.read()


print("Training text characters:")
print(len(text))


# =========================================================
# TOKENIZE ENTIRE CORPUS
# =========================================================

token_ids = encode(text)


print("\nTotal token count:")
print(len(token_ids))


# =========================================================
# CREATE NEXT-TOKEN SEQUENCES
# =========================================================

inputs = []
targets = []

for i in range(
    len(token_ids) - SEQUENCE_LENGTH
):

    input_sequence = token_ids[
        i:i + SEQUENCE_LENGTH
    ]

    target_sequence = token_ids[
        i + 1:i + SEQUENCE_LENGTH + 1
    ]

    inputs.append(input_sequence)
    targets.append(target_sequence)


# =========================================================
# DISPLAY DATASET INFORMATION
# =========================================================

print("\nSequence length:")
print(SEQUENCE_LENGTH)

print("\nNumber of training sequences:")
print(len(inputs))


# =========================================================
# SHOW FIRST EXAMPLE
# =========================================================

if len(inputs) > 0:

    print("\nFirst input:")
    print(inputs[0])

    print("\nFirst target:")
    print(targets[0])
else:

    print("\nNot enough tokens to create a training sequence.")