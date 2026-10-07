import torch
import torch.nn as nn
import sys
from pathlib import Path


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))


# =========================================================
# IMPORT TOKENIZER
# =========================================================

from tokenizer.tokenizer import encode


# =========================================================
# MODEL SETTINGS
# =========================================================

vocab_size = 5299
embedding_size = 128
sequence_length = 4


# =========================================================
# EMBEDDINGS
# =========================================================

token_embedding = nn.Embedding(
    vocab_size,
    embedding_size
)

position_embedding = nn.Embedding(
    sequence_length,
    embedding_size
)


# =========================================================
# INPUT TEXT
# =========================================================

text = "Tamil Nadu Legislative Assembly"


# =========================================================
# TOKENIZE + ENCODE
# =========================================================

token_ids = encode(text)

print("Input text:")
print(text)

print("\nToken IDs:")
print(token_ids)


# =========================================================
# CONVERT TO TENSOR
# =========================================================

token_ids = torch.tensor(
    token_ids,
    dtype=torch.long
)

print("\nToken ID tensor:")
print(token_ids)


# =========================================================
# TOKEN EMBEDDING
# =========================================================

token_vectors = token_embedding(
    token_ids
)

print("\nToken embedding shape:")
print(token_vectors.shape)


# =========================================================
# POSITION IDS
# =========================================================

position_ids = torch.arange(
    len(token_ids)
)

print("\nPosition IDs:")
print(position_ids)


# =========================================================
# POSITION EMBEDDING
# =========================================================

position_vectors = position_embedding(
    position_ids
)

print("\nPosition embedding shape:")
print(position_vectors.shape)


# =========================================================
# INITIAL HIDDEN STATES
# =========================================================

H = token_vectors + position_vectors

print("\nInitial hidden states H⁰ shape:")
print(H.shape)