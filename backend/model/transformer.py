
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# PATHS AND IMPORTS
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from tokenizer.tokenizer import vocab
from model.attention import CausalSelfAttention

# =========================================================
# MODEL CONFIGURATION
# =========================================================

vocab_size = len(vocab)
embedding_size = 128
hidden_size = 256
sequence_length = 64
num_layers = 2


# =========================================================
# DECODER BLOCK
# =========================================================

class DecoderBlock(nn.Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()

        self.attention = CausalSelfAttention(
            embedding_size
        )

        self.layer_norm1 = nn.LayerNorm(
            embedding_size
        )

        self.layer_norm2 = nn.LayerNorm(
            embedding_size
        )

        self.fc1 = nn.Linear(
            embedding_size,
            hidden_size
        )

        self.fc2 = nn.Linear(
            hidden_size,
            embedding_size
        )

    def forward(self, H):

        # Self-attention + residual connection
        attention_output = self.attention(H)

        H = self.layer_norm1(
            H + attention_output
        )

        # Feed-forward network
        mlp_output = self.fc1(H)
        mlp_output = F.gelu(mlp_output)
        mlp_output = self.fc2(mlp_output)

        # Feed-forward residual connection
        H = self.layer_norm2(
            H + mlp_output
        )

        return H


# =========================================================
# TRANSFORMER MODEL
# =========================================================

class TransformerModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        sequence_length,
        num_layers
    ):
        super().__init__()

        self.sequence_length = sequence_length

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            sequence_length,
            embedding_size
        )

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                embedding_size,
                hidden_size
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )

        # Predict vocabulary scores
        self.lm_head = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, token_ids):

        # Token IDs -> token vectors
        token_vectors = self.token_embedding(
            token_ids
        )

        # Create position IDs
        current_sequence_length = token_ids.size(-1)

        if current_sequence_length > self.sequence_length:
            raise ValueError(
                "Input exceeds the configured sequence length."
            )

        position_ids = torch.arange(
            current_sequence_length,
            device=token_ids.device
        )

        # Position IDs -> position vectors
        position_vectors = self.position_embedding(
            position_ids
        )

        # Initial hidden states
        H = token_vectors + position_vectors

        # Process through decoder blocks
        for block in self.decoder_blocks:
            H = block(H)

        # Final hidden states
        H = self.final_layer_norm(H)

        # Vocabulary logits
        logits = self.lm_head(H)

        return logits


# =========================================================
# BASIC MODEL TEST
# =========================================================

if __name__ == "__main__":

    model = TransformerModel(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        num_layers=num_layers
    )

    # Example input: four valid token IDs
    token_ids = torch.tensor(
        [1, 2, 3, 4],
        dtype=torch.long
    )

    logits = model(token_ids)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("ELVARN Transformer Test")
    print("Vocabulary size:", vocab_size)
    print("Input shape:", token_ids.shape)
    print("Logits shape:", logits.shape)
    print(f"Total parameters: {total_parameters:,}")

