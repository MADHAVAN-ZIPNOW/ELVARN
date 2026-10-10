
import sys
from pathlib import Path

import torch

# =========================================================
# PATHS
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from tokenizer.tokenizer import encode, decode, vocab
from model.transformer import TransformerModel


# =========================================================
# CONFIGURATION
# =========================================================

CHECKPOINT_FILE = (
    BACKEND_DIR
    / "training"
    / "checkpoints"
    / "checkpoint_epoch_5.pt"
)

VOCAB_SIZE = len(vocab)
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256
SEQUENCE_LENGTH = 64
NUM_LAYERS = 2
MAX_NEW_TOKENS = 50

TEMPERATURE = 0.8
TOP_K = 20

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# INFORMATION
# =========================================================

print("===================================")
print("MODEL GENERATION TEST")
print("===================================")

print("\nDevice:")
print(device)

print("\nVocabulary size:")
print(VOCAB_SIZE)

print("\nCheckpoint:")
print(CHECKPOINT_FILE)


# =========================================================
# LOAD MODEL
# =========================================================

model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    embedding_size=EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    num_layers=NUM_LAYERS
).to(device)


# =========================================================
# LOAD CHECKPOINT
# =========================================================

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device
)

checkpoint_vocab_size = checkpoint.get(
    "vocab_size",
    None
)

print("\nCheckpoint vocabulary size:")
print(checkpoint_vocab_size)

if checkpoint_vocab_size != VOCAB_SIZE:
    raise ValueError(
        "\nERROR: Vocabulary size mismatch.\n"
        f"Checkpoint vocabulary: {checkpoint_vocab_size}\n"
        f"Current vocabulary: {VOCAB_SIZE}\n"
    )

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nCheckpoint loaded successfully.")

print("\nCheckpoint epoch:")
print(checkpoint["epoch"])

print("\nTraining loss:")
print(checkpoint["train_loss"])

print("\nValidation loss:")
print(checkpoint["validation_loss"])


# =========================================================
# TOP-K SAMPLING
# =========================================================

def sample_next_token(logits, temperature=1.0, top_k=20):

    if temperature <= 0:
        temperature = 1.0

    logits = logits / temperature

    if top_k is not None and top_k > 0:

        top_k = min(
            top_k,
            logits.size(-1)
        )

        values, indices = torch.topk(
            logits,
            top_k
        )

        filtered_logits = torch.full_like(
            logits,
            float("-inf")
        )

        filtered_logits.scatter_(
            0,
            indices,
            values
        )

        logits = filtered_logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return next_token.item()


# =========================================================
# GENERATION
# =========================================================

def generate(prompt):

    token_ids = encode(prompt)

    if not token_ids:
        return prompt

    generated_ids = token_ids.copy()

    for _ in range(MAX_NEW_TOKENS):

        context_ids = generated_ids[
            -SEQUENCE_LENGTH:
        ]

        input_tensor = torch.tensor(
            context_ids,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():

            logits = model(
                input_tensor
            )

        next_token_logits = logits[
            0,
            -1,
            :
        ]

        next_token_id = sample_next_token(
            next_token_logits,
            TEMPERATURE,
            TOP_K
        )

        generated_ids.append(
            next_token_id
        )

        if next_token_id == vocab["<EOS>"]:
            break

    generated_text = decode(
        generated_ids
    )

    return generated_text


# =========================================================
# TEST PROMPTS
# =========================================================

prompts = [
    "Tamil Nadu corruption",
    "The case involved",
    "political",
    "J. Jayalalithaa",
    "Prevention of Corruption Act",
]


# =========================================================
# RUN TESTS
# =========================================================

print("\n===================================")
print("GENERATION TESTS")
print("===================================")

for prompt in prompts:

    print("\n-----------------------------------")
    print("PROMPT:")
    print(prompt)

    result = generate(prompt)

    print("\nGENERATED:")
    print(result)


print("\n===================================")
print("GENERATION TEST COMPLETE")
print("===================================")

