import torch
import torch.nn as nn


# =========================================================
# SETTINGS
# =========================================================

vocab_size = 5299
embedding_size = 128
max_sequence_length = 256


# =========================================================
# TOKEN EMBEDDING
# =========================================================

token_embedding = nn.Embedding(
    vocab_size,
    embedding_size
)


# =========================================================
# POSITION EMBEDDING
# =========================================================

position_embedding = nn.Embedding(
    max_sequence_length,
    embedding_size
)


# =========================================================
# TEST INPUT
# =========================================================

token_ids = torch.tensor(
    [3972, 3974, 4, 5],
    dtype=torch.long
)


# =========================================================
# TOKEN EMBEDDINGS
# =========================================================

token_vectors = token_embedding(token_ids)


# =========================================================
# POSITION IDs
# =========================================================

sequence_length = token_ids.shape[0]

position_ids = torch.arange(
    sequence_length,
    dtype=torch.long
)


# =========================================================
# POSITION VECTORS
# =========================================================

position_vectors = position_embedding(position_ids)


# =========================================================
# ADD TOKEN + POSITION
# =========================================================

hidden_states = token_vectors + position_vectors


# =========================================================
# OUTPUT
# =========================================================

print("Token IDs:")
print(token_ids)

print("\nPosition IDs:")
print(position_ids)

print("\nToken embedding shape:")
print(token_vectors.shape)

print("\nPosition embedding shape:")
print(position_vectors.shape)

print("\nInitial hidden states H⁰ shape:")
print(hidden_states.shape)