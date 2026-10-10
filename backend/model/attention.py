
import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()

        self.W_Q = nn.Linear(embedding_size, embedding_size)
        self.W_K = nn.Linear(embedding_size, embedding_size)
        self.W_V = nn.Linear(embedding_size, embedding_size)
        self.W_O = nn.Linear(embedding_size, embedding_size)

        self.embedding_size = embedding_size

    def forward(self, H):

        # Create Query, Key, and Value vectors
        Q = self.W_Q(H)
        K = self.W_K(H)
        V = self.W_V(H)

        # Calculate attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale scores
        scores = scores / math.sqrt(self.embedding_size)

        # Determine sequence length
        sequence_length = H.size(-2)

        # Create causal mask: block future tokens
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=H.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        # Hide future-token scores
        scores = scores.masked_fill(mask, float("-inf"))

        # Convert scores into attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Combine Value vectors
        attention_output = attention_weights @ V

        # Final output projection
        output = self.W_O(attention_output)

        return output
