import sys
from pathlib import Path

# Add backend directory to Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import torch
import torch.nn as nn
# =========================================================
# MODEL SETTINGS
# =========================================================

embedding_size = 128
hidden_size = 512
sequence_length = 4
num_layers = 100
vocab_size = 5299


# =========================================================
# DECODER BLOCK
# =========================================================

class DecoderBlock(nn.Module):

    def __init__(self, embedding_size, hidden_size):

        super().__init__()

        # -------------------------------------------------
        # Q, K, V projections
        # -------------------------------------------------

        self.W_Q = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.W_K = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.W_V = nn.Linear(
            embedding_size,
            embedding_size
        )

        # -------------------------------------------------
        # Output projection
        # -------------------------------------------------

        self.W_O = nn.Linear(
            embedding_size,
            embedding_size
        )

        # -------------------------------------------------
        # Layer Normalization
        # -------------------------------------------------

        self.layer_norm1 = nn.LayerNorm(
            embedding_size
        )

        self.layer_norm2 = nn.LayerNorm(
            embedding_size
        )

        # -------------------------------------------------
        # Feed-Forward Network
        # -------------------------------------------------

        self.fc1 = nn.Linear(
            embedding_size,
            hidden_size
        )

        self.fc2 = nn.Linear(
            hidden_size,
            embedding_size
        )

    def forward(self, H):

        # =================================================
        # SELF-ATTENTION
        # =================================================

        Q = self.W_Q(H)
        K = self.W_K(H)
        V = self.W_V(H)

        # -------------------------------------------------
        # Attention scores
        # -------------------------------------------------

        scores = Q @ K.transpose(-2, -1)

        # -------------------------------------------------
        # Scale
        # -------------------------------------------------

        scale = embedding_size ** 0.5

        scaled_scores = scores / scale

        # -------------------------------------------------
        # Causal mask
        # -------------------------------------------------

        mask = torch.triu(
            torch.ones(
                H.size(0),
                H.size(0),
                device=H.device
            ),
            diagonal=1
        )

        scaled_scores = scaled_scores.masked_fill(
            mask == 1,
            float("-inf")
        )

        # -------------------------------------------------
        # Softmax
        # -------------------------------------------------

        attention_weights = torch.softmax(
            scaled_scores,
            dim=-1
        )

        # -------------------------------------------------
        # Attention output
        # -------------------------------------------------

        attention_output = attention_weights @ V

        # -------------------------------------------------
        # Output projection
        # -------------------------------------------------

        projected_output = self.W_O(
            attention_output
        )

        # =================================================
        # FIRST RESIDUAL + LAYER NORM
        # =================================================

        residual_output = H + projected_output

        normalized_output = self.layer_norm1(
            residual_output
        )

        # =================================================
        # FEED-FORWARD NETWORK
        # =================================================

        mlp_hidden = self.fc1(
            normalized_output
        )

        mlp_hidden = torch.nn.functional.gelu(
            mlp_hidden
        )

        mlp_output = self.fc2(
            mlp_hidden
        )

        # =================================================
        # SECOND RESIDUAL + LAYER NORM
        # =================================================

        block_output = normalized_output + mlp_output

        block_output = self.layer_norm2(
            block_output
        )

        return block_output


# =========================================================
# INPUT + EMBEDDINGS
# =========================================================

from tokenizer.tokenizer import encode


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
# TOKEN ID TENSOR
# =========================================================

token_ids = torch.tensor(
    token_ids,
    dtype=torch.long
)


# =========================================================
# TOKEN EMBEDDING
# =========================================================

token_embedding = nn.Embedding(
    vocab_size,
    embedding_size
)

token_vectors = token_embedding(
    token_ids
)


# =========================================================
# POSITION EMBEDDING
# =========================================================

position_embedding = nn.Embedding(
    sequence_length,
    embedding_size
)

position_ids = torch.arange(
    len(token_ids)
)

position_vectors = position_embedding(
    position_ids
)


# =========================================================
# INITIAL HIDDEN STATES H⁰
# =========================================================

H = token_vectors + position_vectors

print("\nInitial hidden states H⁰ shape:")
print(H.shape)

print("\nInput H shape:")
print(H.shape)


# =========================================================
# CREATE 100 DECODER BLOCKS
# =========================================================

decoder_blocks = nn.ModuleList([
    DecoderBlock(
        embedding_size,
        hidden_size
    )
    for _ in range(num_layers)
])


# =========================================================
# PASS THROUGH 100 DECODER BLOCKS
# =========================================================

x = H

for layer_number, block in enumerate(
    decoder_blocks,
    start=1
):

    x = block(x)

    print(
        f"\nDecoder Block {layer_number} output shape:"
    )

    print(x.shape)


# =========================================================
# FINAL TRANSFORMER OUTPUT
# =========================================================

print("\nFinal Transformer output shape:")
print(x.shape)


# =========================================================
# FINAL LAYER NORMALIZATION
# =========================================================

final_layer_norm = nn.LayerNorm(
    embedding_size
)

x = final_layer_norm(x)

print("\nAfter Final LayerNorm:")
print(x.shape)


# =========================================================
# LANGUAGE MODEL HEAD
# =========================================================

lm_head = nn.Linear(
    embedding_size,
    vocab_size
)

logits = lm_head(x)

print("\nLM Head output shape:")
print(logits.shape)
# =========================================================
# NEXT-TOKEN PREDICTION
# =========================================================

# Convert logits into probabilities
probabilities = torch.softmax(
    logits,
    dim=-1
)

print("\nProbabilities shape:")
print(probabilities.shape)


# =========================================================
# PREDICT TOKEN WITH HIGHEST PROBABILITY
# =========================================================

predicted_token_ids = torch.argmax(
    probabilities,
    dim=-1
)

print("\nPredicted token IDs:")
print(predicted_token_ids)


# =========================================================
# CONVERT PREDICTED IDs BACK TO TOKENS
# =========================================================

from tokenizer.tokenizer import id_to_token

predicted_tokens = [
    id_to_token.get(
        token_id.item(),
        "<UNK>"
    )
    for token_id in predicted_token_ids
]

print("\nPredicted tokens:")
print(predicted_tokens)


# =========================================================
# PARAMETER COUNT
# =========================================================

total_parameters = sum(
    parameter.numel()
    for parameter in decoder_blocks.parameters()
)

total_parameters += sum(
    parameter.numel()
    for parameter in final_layer_norm.parameters()
)

total_parameters += sum(
    parameter.numel()
    for parameter in lm_head.parameters()
)

print("\nDecoder + Final LayerNorm + LM Head parameters:")
print(f"{total_parameters:,}")


# =========================================================
# COMPLETE MODEL PARAMETER COUNT
# =========================================================

total_model_parameters = (
    sum(
        parameter.numel()
        for parameter in token_embedding.parameters()
    )

    + sum(
        parameter.numel()
        for parameter in position_embedding.parameters()
    )

    + sum(
        parameter.numel()
        for parameter in decoder_blocks.parameters()
    )

    + sum(
        parameter.numel()
        for parameter in final_layer_norm.parameters()
    )

    + sum(
        parameter.numel()
        for parameter in lm_head.parameters()
    )
)

print("\nComplete model parameters:")
print(f"{total_model_parameters:,}")

