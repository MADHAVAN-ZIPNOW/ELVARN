import torch
import torch.nn as nn


# =========================================================
# VOCABULARY
# =========================================================

vocab_size = 5299


# =========================================================
# EXAMPLE MODEL OUTPUT
# =========================================================
# In the real model this will come from the LM Head.
#
# Shape:
# [sequence_length, vocab_size]
#
# For our example:
# [3, 5299]

logits = torch.randn(
    3,
    vocab_size
)


print("Logits shape:")
print(logits.shape)


# =========================================================
# TARGET TOKENS
# =========================================================
# Example:
#
# Tamil       -> Nadu
# Nadu        -> Legislative
# Legislative -> Assembly
#
# These are the correct next tokens.

target_tokens = torch.tensor(
    [
        3974,   # Nadu
        4,      # Legislative
        5       # Assembly
    ],
    dtype=torch.long
)


print("\nTarget token IDs:")
print(target_tokens)


# =========================================================
# CROSS-ENTROPY LOSS
# =========================================================

loss_function = nn.CrossEntropyLoss()


loss = loss_function(
    logits,
    target_tokens
)


print("\nCross-Entropy Loss:")
print(loss.item())