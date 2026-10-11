import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from model.attention import CausalSelfAttention

# =========================================================
# DEFAULT MODEL CONFIGURATION
# =========================================================
# vocab_size is NOT fixed here: it comes from the trained BPE tokenizer.

DEFAULT_CONFIG = {
    "embedding_size": 128,
    "hidden_size": 256,
    "sequence_length": 128,
    "num_layers": 2,
    "num_heads": 4,
    "dropout": 0.1,
    "tie_weights": True,
}


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_heads=1, dropout=0.0):
        super().__init__()

        self.attention = CausalSelfAttention(
            embedding_size, num_heads=num_heads, dropout=dropout
        )
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H):
        H = self.layer_norm1(H + self.dropout(self.attention(H)))

        mlp_output = self.fc2(F.gelu(self.fc1(H)))
        H = self.layer_norm2(H + self.dropout(mlp_output))
        return H


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        sequence_length,
        num_layers,
        num_heads=1,
        dropout=0.0,
        tie_weights=False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(sequence_length, embedding_size)
        self.embedding_dropout = nn.Dropout(dropout)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embedding_size, hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

        # Small-scale init: with PyTorch's default N(0,1) embeddings the
        # starting loss is far above ln(vocab_size) (especially when tied).
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        if tie_weights:
            # Output projection shares the token embedding matrix.
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids):
        current_sequence_length = token_ids.size(-1)

        if current_sequence_length > self.sequence_length:
            raise ValueError(
                f"Input length {current_sequence_length} exceeds the "
                f"configured sequence length {self.sequence_length}."
            )

        position_ids = torch.arange(
            current_sequence_length, device=token_ids.device
        )

        H = self.token_embedding(token_ids) + self.position_embedding(position_ids)
        H = self.embedding_dropout(H)

        for block in self.decoder_blocks:
            H = block(H)

        H = self.final_layer_norm(H)
        return self.lm_head(H)


def build_model(vocab_size, config=None):
    """Create a TransformerModel from a config dict (defaults + overrides)."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    return TransformerModel(
        vocab_size=vocab_size,
        embedding_size=cfg["embedding_size"],
        hidden_size=cfg["hidden_size"],
        sequence_length=cfg["sequence_length"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        tie_weights=cfg["tie_weights"],
    )


def count_parameters(model):
    # parameters() already counts a tied matrix only once.
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = build_model(vocab_size=4096)
    token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    logits = model(token_ids)

    print("ELVARN Transformer Test")
    print("Input shape:", tuple(token_ids.shape))
    print("Logits shape:", tuple(logits.shape))
    print(f"Total parameters: {count_parameters(model):,}")
