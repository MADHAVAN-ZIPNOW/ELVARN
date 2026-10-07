import sys
from pathlib import Path
import math
import torch
import torch.nn.functional as F

# =========================================================
# PATHS
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from tokenizer.tokenizer import (
    encode,
    decode,
    vocab
)

from model.transformer import TransformerModel


# =========================================================
# CONFIGURATION
# =========================================================

VOCAB_SIZE = len(vocab)

EMBEDDING_SIZE = 128
HIDDEN_SIZE = 512
SEQUENCE_LENGTH = 128
NUM_LAYERS = 100

CHECKPOINT_FILE = (
    BACKEND_DIR
    / "training"
    / "checkpoints"
    / "checkpoint_epoch_5.pt"
)

DATA_FILE = (
    BACKEND_DIR
    / "data"
    / "train.txt"
)

EVALUATION_SEQUENCE_LENGTH = 128


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# HEADER
# =========================================================

print("===================================")
print("MODEL EVALUATION")
print("===================================")

print("\nDevice:")
print(device)

print("\nVocabulary size:")
print(VOCAB_SIZE)

print("\nSequence length:")
print(SEQUENCE_LENGTH)

print("\nNumber of layers:")
print(NUM_LAYERS)

print("\nCheckpoint:")
print(CHECKPOINT_FILE)


# =========================================================
# LOAD CORPUS
# =========================================================

print("\n===================================")
print("LOADING EVALUATION DATA")
print("===================================")

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as file:
    text = file.read()

print("\nCorpus characters:")
print(len(text))


token_ids = encode(text)

print("\nCorpus tokens:")
print(len(token_ids))


# =========================================================
# CREATE EVALUATION SEQUENCES
# =========================================================

evaluation_inputs = []
evaluation_targets = []

for i in range(
    len(token_ids) - EVALUATION_SEQUENCE_LENGTH
):

    input_sequence = token_ids[
        i:
        i + EVALUATION_SEQUENCE_LENGTH
    ]

    target_sequence = token_ids[
        i + 1:
        i + EVALUATION_SEQUENCE_LENGTH + 1
    ]

    evaluation_inputs.append(
        input_sequence
    )

    evaluation_targets.append(
        target_sequence
    )


print("\nEvaluation sequences:")
print(len(evaluation_inputs))


# =========================================================
# LOAD MODEL
# =========================================================

print("\n===================================")
print("LOADING MODEL")
print("===================================")

model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    embedding_size=EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    num_layers=NUM_LAYERS
)


checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

else:

    model.load_state_dict(
        checkpoint
    )


model.to(device)
model.eval()


print("\nModel loaded successfully.")


# =========================================================
# MODEL PARAMETER COUNT
# =========================================================

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel parameters:")
print(parameter_count)


# =========================================================
# EVALUATION
# =========================================================

print("\n===================================")
print("RUNNING EVALUATION")
print("===================================")


total_loss = 0.0
total_tokens = 0
correct_tokens = 0


with torch.no_grad():

    for index in range(
        len(evaluation_inputs)
    ):

        input_ids = torch.tensor(
            evaluation_inputs[index],
            dtype=torch.long,
            device=device
        )

        target_ids = torch.tensor(
            evaluation_targets[index],
            dtype=torch.long,
            device=device
        )

        logits = model(
            input_ids
        )

        loss = F.cross_entropy(
            logits,
            target_ids,
            reduction="sum"
        )

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        correct = (
            predictions == target_ids
        ).sum().item()

        correct_tokens += correct

        total_tokens += target_ids.numel()


# =========================================================
# FINAL METRICS
# =========================================================

average_loss = (
    total_loss / total_tokens
    if total_tokens > 0
    else 0.0
)

accuracy = (
    correct_tokens / total_tokens
    if total_tokens > 0
    else 0.0
)

perplexity = math.exp(
    min(average_loss, 20)
)


print("\n===================================")
print("EVALUATION RESULTS")
print("===================================")

print("\nTotal evaluated tokens:")
print(total_tokens)

print("\nCorrect next-token predictions:")
print(correct_tokens)

print("\nNext-token accuracy:")
print(f"{accuracy * 100:.4f}%")

print("\nAverage cross-entropy loss:")
print(f"{average_loss:.6f}")

print("\nPerplexity:")
print(f"{perplexity:.6f}")


# =========================================================
# SAMPLE NEXT-TOKEN TESTS
# =========================================================

print("\n===================================")
print("NEXT-TOKEN PREDICTION TEST")
print("===================================")


sample_positions = [
    0,
    100,
    200,
    300,
    400
]


with torch.no_grad():

    for position in sample_positions:

        if position >= len(token_ids) - 1:
            continue

        start = max(
            0,
            position - SEQUENCE_LENGTH + 1
        )

        context = token_ids[
            start:
            position + 1
        ]

        input_tensor = torch.tensor(
            context,
            dtype=torch.long,
            device=device
        )

        logits = model(
            input_tensor
        )

        next_token_logits = logits[-1]

        predicted_id = torch.argmax(
            next_token_logits
        ).item()

        actual_id = token_ids[
            position + 1
        ]

        predicted_token = decode(
            [predicted_id]
        )

        actual_token = decode(
            [actual_id]
        )

        print("\n-----------------------------------")

        print("Position:")
        print(position)

        print("Context:")
        print(
            decode(context[-20:])
        )

        print("\nActual next token:")
        print(
            repr(actual_token)
        )

        print("Predicted next token:")
        print(
            repr(predicted_token)
        )

        if predicted_id == actual_id:
            print("Result: CORRECT")
        else:
            print("Result: WRONG")


# =========================================================
# FINAL
# =========================================================

print("\n===================================")
print("EVALUATION COMPLETE")
print("===================================")