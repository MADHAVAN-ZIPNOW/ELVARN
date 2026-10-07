import sys
from pathlib import Path

import torch ,os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(2)
# =========================================================
# PATH SETUP
# ==========================================+++++===============

BACKEND_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BACKEND_DIR))

from tokenizer.tokenizer import encode, vocab
from model.transformer import TransformerModel


# =========================================================
# CONFIGURATION
# =========================================================

DATA_FILE = BACKEND_DIR / "data" / "train.txt"

CHECKPOINT_DIR = (
    BACKEND_DIR
    / "training"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

# IMPORTANT:
# Always take vocabulary size directly from vocab.json.
# Do NOT hardcode 5515 or 5605.

VOCAB_SIZE = len(vocab)

EMBEDDING_SIZE = 128
HIDDEN_SIZE = 512
NUM_LAYERS = 100


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

SEQUENCE_LENGTH = 128

BATCH_SIZE = 32

LEARNING_RATE = 0.001

NUM_EPOCHS = 5

TRAIN_RATIO = 0.90

RANDOM_SEED = 42


# =========================================================
# DEVICE
# =========================================================

device = torch.device("cpu")


print("===================================")
print("TRAIN / VALIDATION TRAINING")
print("===================================")

print("\nDevice:")
print(device)

print("\nVocabulary size:")
print(VOCAB_SIZE)


# =========================================================
# LOAD CORPUS
# =========================================================

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


print("\nTraining file:")
print(DATA_FILE)

print("\nTraining text characters:")
print(len(text))


# =========================================================
# TOKENIZATION
# =========================================================

token_ids = encode(text)

print("\nTotal tokens:")
print(len(token_ids))


# =========================================================
# SAFETY CHECK
# =========================================================

max_token_id = max(token_ids)

print("\nMaximum token ID:")
print(max_token_id)

print("\nVocabulary maximum ID:")
print(VOCAB_SIZE - 1)


if max_token_id >= VOCAB_SIZE:

    raise ValueError(
        "\nERROR: Token ID is outside vocabulary range.\n"
        f"Maximum token ID: {max_token_id}\n"
        f"Vocabulary size: {VOCAB_SIZE}\n"
        f"Valid IDs: 0 to {VOCAB_SIZE - 1}"
    )


print("\nToken ID range check:")
print("PASSED")


# =========================================================
# CREATE SEQUENCES
# =========================================================

all_inputs = []
all_targets = []


for i in range(
    len(token_ids) - SEQUENCE_LENGTH
):

    input_sequence = token_ids[
        i:i + SEQUENCE_LENGTH
    ]

    target_sequence = token_ids[
        i + 1:i + SEQUENCE_LENGTH + 1
    ]

    all_inputs.append(
        torch.tensor(
            input_sequence,
            dtype=torch.long
        )
    )

    all_targets.append(
        torch.tensor(
            target_sequence,
            dtype=torch.long
        )
    )


print("\nTotal sequences:")
print(len(all_inputs))


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

total_sequences = len(all_inputs)

train_size = int(
    total_sequences * TRAIN_RATIO
)

validation_size = (
    total_sequences - train_size
)


# =========================================================
# REPRODUCIBLE SHUFFLE
# =========================================================

generator = torch.Generator()

generator.manual_seed(
    RANDOM_SEED
)

indices = torch.randperm(
    total_sequences,
    generator=generator
).tolist()


train_indices = indices[
    :train_size
]

validation_indices = indices[
    train_size:
]


train_inputs = [
    all_inputs[i]
    for i in train_indices
]

train_targets = [
    all_targets[i]
    for i in train_indices
]

validation_inputs = [
    all_inputs[i]
    for i in validation_indices
]

validation_targets = [
    all_targets[i]
    for i in validation_indices
]


print("\n===================================")
print("DATA SPLIT")
print("===================================")

print("\nTraining sequences:")
print(len(train_inputs))

print("\nValidation sequences:")
print(len(validation_inputs))


# =========================================================
# DATASET CLASS
# =========================================================

class LanguageModelDataset(Dataset):

    def __init__(
        self,
        inputs,
        targets
    ):

        self.inputs = inputs
        self.targets = targets

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        return (
            self.inputs[index],
            self.targets[index]
        )


# =========================================================
# CREATE DATASETS
# =========================================================

train_dataset = LanguageModelDataset(
    train_inputs,
    train_targets
)

validation_dataset = LanguageModelDataset(
    validation_inputs,
    validation_targets
)


# =========================================================
# DATA LOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nTraining batches:")
print(len(train_loader))

print("\nValidation batches:")
print(len(validation_loader))


# =========================================================
# CREATE MODEL
# =========================================================

model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    embedding_size=EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    num_layers=NUM_LAYERS
)

model = model.to(device)


# =========================================================
# PARAMETER COUNT
# =========================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel parameters:")
print(f"{total_parameters:,}")


# =========================================================
# LOSS FUNCTION
# =========================================================

loss_function = nn.CrossEntropyLoss()


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# TRAINING LOOP
# =========================================================

print("\n===================================")
print("STARTING TRAINING")
print("===================================")


for epoch in range(NUM_EPOCHS):

    # =====================================================
    # TRAINING MODE
    # =====================================================

    model.train()

    total_train_loss = 0.0

    train_batch_count = 0


    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print("-----------------------------------")


    # =====================================================
    # TRAINING BATCHES
    # =====================================================

    for batch_index, (
        inputs,
        targets
    ) in enumerate(train_loader):

        inputs = inputs.to(device)

        targets = targets.to(device)


        # -------------------------------------------------
        # CLEAR GRADIENTS
        # -------------------------------------------------

        optimizer.zero_grad()


        # -------------------------------------------------
        # FORWARD PASS
        # -------------------------------------------------

        logits = model(inputs)


        # -------------------------------------------------
        # SAFETY CHECK
        # -------------------------------------------------

        if logits.size(-1) != VOCAB_SIZE:

            raise ValueError(
                "\nERROR: Model output vocabulary size "
                "does not match vocabulary."
                f"\nModel output: {logits.size(-1)}"
                f"\nVocabulary: {VOCAB_SIZE}"
            )


        # -------------------------------------------------
        # RESHAPE FOR LOSS
        # -------------------------------------------------

        logits_for_loss = logits.reshape(
            -1,
            VOCAB_SIZE
        )

        targets_for_loss = targets.reshape(
            -1
        )


        # -------------------------------------------------
        # LOSS
        # -------------------------------------------------

        loss = loss_function(
            logits_for_loss,
            targets_for_loss
        )


        # -------------------------------------------------
        # BACKPROPAGATION
        # -------------------------------------------------

        loss.backward()


        # -------------------------------------------------
        # UPDATE WEIGHTS
        # -------------------------------------------------

        optimizer.step()


        # -------------------------------------------------
        # TRACK LOSS
        # -------------------------------------------------

        total_train_loss += loss.item()

        train_batch_count += 1


        # -------------------------------------------------
        # FIRST BATCH
        # -------------------------------------------------

        if batch_index == 0:

            print(
                f"First training batch loss: "
                f"{loss.item():.6f}"
            )


    # =====================================================
    # AVERAGE TRAINING LOSS
    # =====================================================

    average_train_loss = (
        total_train_loss
        / train_batch_count
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    total_validation_loss = 0.0

    validation_batch_count = 0


    with torch.no_grad():

        for (
            inputs,
            targets
        ) in validation_loader:

            inputs = inputs.to(device)

            targets = targets.to(device)


            # ---------------------------------------------
            # FORWARD PASS
            # ---------------------------------------------

            logits = model(inputs)


            # ---------------------------------------------
            # RESHAPE
            # ---------------------------------------------

            logits_for_loss = logits.reshape(
                -1,
                VOCAB_SIZE
            )

            targets_for_loss = targets.reshape(
                -1
            )


            # ---------------------------------------------
            # VALIDATION LOSS
            # ---------------------------------------------

            loss = loss_function(
                logits_for_loss,
                targets_for_loss
            )


            total_validation_loss += (
                loss.item()
            )

            validation_batch_count += 1


    # =====================================================
    # AVERAGE VALIDATION LOSS
    # =====================================================

    average_validation_loss = (
        total_validation_loss
        / validation_batch_count
    )


    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print(
        f"Average training loss: "
        f"{average_train_loss:.6f}"
    )

    print(
        f"Average validation loss: "
        f"{average_validation_loss:.6f}"
    )


    # =====================================================
    # SAVE CHECKPOINT
    # =====================================================

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"checkpoint_epoch_{epoch + 1}.pt"
    )


    torch.save(
        {
            "epoch": epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                average_train_loss,

            "validation_loss":
                average_validation_loss,

            "vocab_size":
                VOCAB_SIZE,

            "embedding_size":
                EMBEDDING_SIZE,

            "hidden_size":
                HIDDEN_SIZE,

            "num_layers":
                NUM_LAYERS,

            "sequence_length":
                SEQUENCE_LENGTH
        },
        checkpoint_path
    )


    print(
        f"Checkpoint saved: "
        f"{checkpoint_path.name}"
    )


# =========================================================
# COMPLETE
# =========================================================

print("\n===================================")
print("TRAINING COMPLETE")
print("===================================")

print("\nFinal training loss:")
print(
    f"{average_train_loss:.6f}"
)

print("\nFinal validation loss:")
print(
    f"{average_validation_loss:.6f}"
)

print("\nFinal vocabulary size:")
print(VOCAB_SIZE)

print("\nCheckpoint directory:")
print(CHECKPOINT_DIR)