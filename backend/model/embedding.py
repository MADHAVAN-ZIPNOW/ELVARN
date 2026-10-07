import json
from pathlib import Path

import torch
import torch.nn as nn


# =========================================================
# LOAD VOCABULARY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VOCAB_FILE = BASE_DIR / "tokenizer" / "vocab.json"

with open(VOCAB_FILE, "r", encoding="utf-8") as file:
    vocab = json.load(file)


# =========================================================
# MODEL SETTINGS
# =========================================================

vocab_size = len(vocab)
embedding_size = 128


# =========================================================
# EMBEDDING LAYER
# =========================================================

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_size
)


# =========================================================
# TEST TOKEN IDs
# =========================================================

token_ids = torch.tensor(
    [3972, 3974, 4, 5],
    dtype=torch.long
)


# =========================================================
# GET EMBEDDINGS
# =========================================================

vectors = embedding(token_ids)


# =========================================================
# PRINT RESULTS
# =========================================================

print("Vocabulary size :", vocab_size)
print("Embedding size  :", embedding_size)

print("\nToken IDs:")
print(token_ids)

print("\nEmbedding shape:")
print(vectors.shape)

print("\nEmbedding vectors:")
print(vectors)