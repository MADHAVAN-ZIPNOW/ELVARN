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


# Named size presets. "tiny" is the model that is trained on this CPU.
# "1b" is about 1.0 billion parameters: it needs roughly 16 GB for training
# state alone (weights + gradients + AdamW), so it needs a bigger machine.
PRESETS = {
    "tiny": {},  # DEFAULT_CONFIG as is: ~0.8M parameters
    "7m": {"embedding_size": 256, "hidden_size": 1024, "num_layers": 4, "num_heads": 8},
    "1b": {"embedding_size": 2048, "hidden_size": 8192, "num_layers": 20,
           "num_heads": 16, "sequence_length": 512},
}


def preset_config(name):
    if name not in PRESETS:
        raise ValueError(f"Unknown size '{name}'. Choose from: {', '.join(PRESETS)}")
    return {**DEFAULT_CONFIG, **PRESETS[name]}


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

    def forward(self, H, past=None, use_cache=False):
        if past is None and not use_cache:
            attn_out = self.attention(H)  # original training path, untouched
            present = None
        else:
            attn_out, present = self.attention(H, past=past, use_cache=True)
        H = self.layer_norm1(H + self.dropout(attn_out))

        mlp_output = self.fc2(F.gelu(self.fc1(H)))
        H = self.layer_norm2(H + self.dropout(mlp_output))
        if use_cache:
            return H, present
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

    def forward_cached(self, token_ids, past=None, last_only=False):
        """Incremental decoding. Same function as forward(), but only the NEW tokens are computed.

        token_ids: (batch, new_len) the new tokens only.
        past:      None (start fresh) or the `presents` returned by the previous call (one (key, value)
                   per layer). Positions of the new tokens start at past_len.
        last_only: return logits for the last new position only (saves the big output projection).
        Returns (logits, presents). Kept separate from forward() so training/eval calls never change.
        """
        new_len = token_ids.size(-1)
        past_len = 0 if past is None else past[0][0].size(2)
        if past is not None and len(past) != len(self.decoder_blocks):
            raise ValueError(f"past has {len(past)} layers, the model has {len(self.decoder_blocks)}.")
        if past_len + new_len > self.sequence_length:
            raise ValueError(
                f"Cached length {past_len} + new length {new_len} exceeds the "
                f"configured sequence length {self.sequence_length}."
            )

        position_ids = torch.arange(past_len, past_len + new_len, device=token_ids.device)
        H = self.token_embedding(token_ids) + self.position_embedding(position_ids)
        H = self.embedding_dropout(H)

        presents = []
        for i, block in enumerate(self.decoder_blocks):
            H, present = block(H, past=None if past is None else past[i], use_cache=True)
            presents.append(present)

        if last_only:
            H = H[:, -1:, :]
        H = self.final_layer_norm(H)
        return self.lm_head(H), presents


REQUIRED_CONFIG_KEYS = tuple(DEFAULT_CONFIG)


def config_from_checkpoint(ck):
    """The model config stored in a checkpoint, with NOTHING filled in silently.

    The number of attention heads cannot be recovered from the weights (W_Q, W_K, W_V, W_O are
    embedding_size x embedding_size for ANY head count), so loading a checkpoint with the wrong head
    count would raise no shape error and just give different answers. Therefore the head count must be
    written in the checkpoint.

    * New checkpoints have a "config" dict: it must contain every key, including num_heads.
    * Old word-level checkpoints (before heads existed) have the sizes at the top level and no "config":
      those were trained with ONE head, untied embeddings and no dropout, and are loaded as such.
    """
    if "config" in ck:
        cfg = dict(ck["config"])
        missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
        if missing:
            hint = (" The head count is NOT stored in the weights, so it cannot be guessed; "
                    "re-save the checkpoint with num_heads in its config.") if "num_heads" in missing else ""
            raise ValueError(f"Checkpoint config is missing {missing}.{hint}")
        return cfg
    legacy = ("embedding_size", "hidden_size", "num_layers", "sequence_length")
    if all(k in ck for k in legacy):
        return {"embedding_size": ck["embedding_size"], "hidden_size": ck["hidden_size"],
                "sequence_length": ck["sequence_length"], "num_layers": ck["num_layers"],
                "num_heads": 1, "dropout": 0.0, "tie_weights": False}
    raise ValueError("This file does not look like an ELVARN checkpoint (no model config found).")


def model_from_checkpoint(ck):
    """Build the model a checkpoint describes and load its weights STRICTLY (no silently ignored or
    mismatched tensors). Returns (model, config)."""
    cfg = config_from_checkpoint(ck)
    model = build_model(ck["vocab_size"], cfg)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model, cfg


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
