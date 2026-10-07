import sys
from pathlib import Path

# =========================================================
# PATH SETUP
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# =========================================================
# IMPORTS
# =========================================================

import torch
import torch.nn as nn

from transformer import TransformerModel
from tokenizer.tokenizer import encode


# =========================================================
# MODEL SETTINGS
# =========================================================

vocab_size = 5299
embedding_size = 128
hidden_size = 512
sequence_length = 4
num_layers = 100


# =========================================================
# CREATE MODEL
# =========================================================

model = TransformerModel(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    hidden_size=hidden_size,
    sequence_length=sequence_length,
    num_layers=num_layers
)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)


# =========================================================
# LOSS FUNCTION
# =========================================================

loss_function = nn.CrossEntropyLoss()


# =========================================================
# TRAINING EXAMPLE
# =========================================================

text = "Tamil Nadu Legislative Assembly"

token_ids = encode(text)

token_ids = torch.tensor(
    token_ids,
    dtype=torch.long
)


print("Original token IDs:")
print(token_ids)


# =========================================================
# NEXT-TOKEN SHIFTING
# =========================================================

input_tokens = token_ids[:-1]

target_tokens = token_ids[1:]


print("\nInput token IDs:")
print(input_tokens)

print("\nTarget token IDs:")
print(target_tokens)


# =========================================================
# CHECK INITIAL WEIGHT
# =========================================================

weight_before = model.token_embedding.weight[
    3972
].detach().clone()


# =========================================================
# FORWARD PASS
# =========================================================

logits = model(input_tokens)


print("\nInput shape:")
print(input_tokens.shape)

print("\nLogits shape:")
print(logits.shape)


# =========================================================
# CALCULATE LOSS
# =========================================================

loss = loss_function(
    logits,
    target_tokens
)


print("\nLoss BEFORE BACKPROPAGATION:")
print(loss.item())


# =========================================================
# BACKPROPAGATION
# =========================================================

optimizer.zero_grad()

loss.backward()


# =========================================================
# CHECK GRADIENT
# =========================================================
gradient = model.token_embedding.weight.grad

print("\nGradient exists:")
print(gradient is not None)

if gradient is not None:

    print("\nFull embedding gradient shape:")
    print(gradient.shape)

    selected_gradient = gradient[3972]

    print("\nGradient for token 3972:")
    print(selected_gradient)

    print("\nGradient norm for token 3972:")
    print(selected_gradient.norm().item())


print("\nGradient exists:")
print(gradient is not None)

if gradient is not None:
    print("\nGradient shape:")
    print(gradient.shape)

    print("\nGradient norm:")
    print(gradient.norm().item())


# =========================================================
# UPDATE MODEL WEIGHTS
# =========================================================

optimizer.step()


# =========================================================
# CHECK WEIGHT CHANGE
# =========================================================

weight_after = model.token_embedding.weight[
    3972
].detach().clone()


weight_change = torch.norm(
    weight_after - weight_before
)


print("\nWeight change after optimizer step:")
print(weight_change.item())


# =========================================================
# SECOND FORWARD PASS
# =========================================================

new_logits = model(input_tokens)

new_loss = loss_function(
    new_logits,
    target_tokens
)


print("\nLoss AFTER WEIGHT UPDATE:")
print(new_loss.item())