import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    The embedding is split into `num_heads` smaller heads. Each head looks at
    the past with its own attention pattern. Parameter count is the same as
    single-head attention (4 square matrices); only the reshaping differs.
    With num_heads=1 this is the original single-head attention.
    """

    def __init__(self, embedding_size, num_heads=1, dropout=0.0):
        super().__init__()
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads < 1:
            raise ValueError(f"num_heads must be a positive whole number, got {num_heads!r}.")
        if embedding_size % num_heads != 0:
            raise ValueError(
                f"embedding_size ({embedding_size}) must be divisible by num_heads ({num_heads}): "
                f"each head gets embedding_size / num_heads dimensions. "
                f"Pick a num_heads that divides {embedding_size} exactly."
            )
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        self.dropout = dropout

        self.W_Q = nn.Linear(embedding_size, embedding_size)
        self.W_K = nn.Linear(embedding_size, embedding_size)
        self.W_V = nn.Linear(embedding_size, embedding_size)
        self.W_O = nn.Linear(embedding_size, embedding_size)

    def _split_heads(self, x):
        # (batch, seq, embedding) -> (batch, heads, seq, head_size)
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.head_size).transpose(1, 2)

    def forward(self, H, past=None, use_cache=False):
        """With past=None and use_cache=False this is the original computation (training path).

        past:      optional (key, value), each (batch, heads, past_len, head_size), from earlier calls.
                   H then holds only the NEW tokens; they attend to past + themselves.
        use_cache: also return the (key, value) covering past + new tokens, to pass in next time.
        Returns the output, or (output, (key, value)) when use_cache=True.
        """
        Q = self._split_heads(self.W_Q(H))
        K = self._split_heads(self.W_K(H))
        V = self._split_heads(self.W_V(H))

        if past is None:
            # is_causal=True hides future positions: token t only sees tokens <= t.
            out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            past_len = past[0].size(2)
            new_len = Q.size(2)
            K = torch.cat([past[0], K], dim=2)
            V = torch.cat([past[1], V], dim=2)
            if new_len == 1:
                mask = None  # the single new token may see everything before it
            else:
                # new token i (0-based) sees the past and new tokens 0..i
                mask = torch.ones(new_len, past_len + new_len, dtype=torch.bool, device=Q.device)
                mask = torch.tril(mask, diagonal=past_len)
            out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        batch, _, seq, _ = out.shape
        out = out.transpose(1, 2).reshape(batch, seq, self.embedding_size)
        out = self.W_O(out)
        if use_cache:
            return out, (K, V)
        return out
