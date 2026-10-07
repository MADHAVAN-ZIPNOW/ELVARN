import torch
import torch.nn as nn


# =========================================================
# MODEL CONFIGURATION
# =========================================================

embedding_size = 128
hidden_size = 512
sequence_length = 128
num_layers = 100
vocab_size = 5515


# =========================================================
# DECODER BLOCK
# =========================================================

class DecoderBlock(nn.Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()

        # Self-attention projections
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

        self.W_O = nn.Linear(
            embedding_size,
            embedding_size
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(
            embedding_size
        )

        self.layer_norm2 = nn.LayerNorm(
            embedding_size
        )

        # Feed-forward network
        self.fc1 = nn.Linear(
            embedding_size,
            hidden_size
        )

        self.fc2 = nn.Linear(
            hidden_size,
            embedding_size
        )

    def forward(self, H):

        # -------------------------------------------------
        # H shape:
        #
        # [sequence, embedding]
        # OR
        # [batch, sequence, embedding]
        # -------------------------------------------------

        Q = self.W_Q(H)
        K = self.W_K(H)
        V = self.W_V(H)

        # -------------------------------------------------
        # Attention scores
        # -------------------------------------------------

        if H.dim() == 2:

            # [sequence, embedding]
            scores = Q @ K.transpose(-2, -1)

            sequence_length_current = H.size(0)

        else:

            # [batch, sequence, embedding]
            scores = Q @ K.transpose(-2, -1)

            sequence_length_current = H.size(1)

        # -------------------------------------------------
        # Scale attention scores
        # -------------------------------------------------

        scale = self.W_Q.in_features ** 0.5

        scaled_scores = scores / scale

        # -------------------------------------------------
        # Causal mask
        #
        # Prevents a token from seeing future tokens.
        # -------------------------------------------------

        mask = torch.triu(
            torch.ones(
                sequence_length_current,
                sequence_length_current,
                device=H.device
            ),
            diagonal=1
        )

        if H.dim() == 3:

            # Add batch dimension
            mask = mask.unsqueeze(0)

        scaled_scores = scaled_scores.masked_fill(
            mask == 1,
            float("-inf")
        )

        # -------------------------------------------------
        # Attention weights
        # -------------------------------------------------

        attention_weights = torch.softmax(
            scaled_scores,
            dim=-1
        )

        # -------------------------------------------------
        # Attention output
        # -------------------------------------------------

        attention_output = attention_weights @ V

        projected_output = self.W_O(
            attention_output
        )

        # -------------------------------------------------
        # First residual connection + LayerNorm
        # -------------------------------------------------

        normalized_output = self.layer_norm1(
            H + projected_output
        )

        # -------------------------------------------------
        # Feed-forward network
        # -------------------------------------------------

        mlp_hidden = self.fc1(
            normalized_output
        )

        mlp_hidden = torch.nn.functional.gelu(
            mlp_hidden
        )

        mlp_output = self.fc2(
            mlp_hidden
        )

        # -------------------------------------------------
        # Second residual connection + LayerNorm
        # -------------------------------------------------

        block_output = self.layer_norm2(
            normalized_output + mlp_output
        )

        return block_output


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

        # -------------------------------------------------
        # Token embedding
        # -------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # -------------------------------------------------
        # Position embedding
        # -------------------------------------------------

        self.position_embedding = nn.Embedding(
            sequence_length,
            embedding_size
        )

        # -------------------------------------------------
        # Decoder blocks
        # -------------------------------------------------

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                embedding_size,
                hidden_size
            )
            for _ in range(num_layers)
        ])

        # -------------------------------------------------
        # Final normalization
        # -------------------------------------------------

        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )

        # -------------------------------------------------
        # Language-model head
        # -------------------------------------------------

        self.lm_head = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, token_ids):

        # =================================================
        # TOKEN EMBEDDINGS
        # =================================================

        token_vectors = self.token_embedding(
            token_ids
        )

        # =================================================
        # POSITION IDS
        # =================================================

        if token_ids.dim() == 1:

            # ---------------------------------------------
            # Input:
            # [sequence]
            # ---------------------------------------------

            current_sequence_length = token_ids.size(0)

            position_ids = torch.arange(
                current_sequence_length,
                device=token_ids.device
            )

        else:

            # ---------------------------------------------
            # Input:
            # [batch, sequence]
            # ---------------------------------------------

            current_sequence_length = token_ids.size(1)

            position_ids = torch.arange(
                current_sequence_length,
                device=token_ids.device
            )

        # =================================================
        # POSITION EMBEDDINGS
        # =================================================

        position_vectors = self.position_embedding(
            position_ids
        )

        # =================================================
        # INITIAL HIDDEN STATES
        #
        # H⁰ = Token Embedding + Position Embedding
        # =================================================

        H = token_vectors + position_vectors

        # =================================================
        # DECODER BLOCKS
        # =================================================

        x = H

        for block in self.decoder_blocks:

            x = block(x)

        # =================================================
        # FINAL LAYER NORMALIZATION
        # =================================================

        x = self.final_layer_norm(x)

        # =================================================
        # LANGUAGE MODEL HEAD
        # =================================================

        logits = self.lm_head(x)

        return logits


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("===================================")
    print("TRANSFORMER MODEL TEST")
    print("===================================")

    model = TransformerModel(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        num_layers=num_layers
    )

    # -----------------------------------------------------
    # Single sequence test
    # -----------------------------------------------------

    token_ids = torch.tensor(
        [3972, 3974, 4, 5],
        dtype=torch.long
    )

    logits = model(token_ids)

    print("\nSingle sequence:")
    print("Input shape:")
    print(token_ids.shape)

    print("\nLogits shape:")
    print(logits.shape)

    # -----------------------------------------------------
    # Batch test
    # -----------------------------------------------------

    batch_token_ids = torch.tensor(
        [
            [3972, 3974, 4, 5],
            [3972, 3974, 4, 5],
            [3972, 3974, 4, 5],
            [3972, 3974, 4, 5]
        ],
        dtype=torch.long
    )

    batch_logits = model(
        batch_token_ids
    )

    print("\nBatch:")
    print("Input shape:")
    print(batch_token_ids.shape)

    print("\nBatch logits shape:")
    print(batch_logits.shape)

    # -----------------------------------------------------
    # Parameter count
    # -----------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nComplete model parameters:")
    print(f"{total_parameters:,}")

    print("\n===================================")
    print("TEST COMPLETE")
    print("===================================")