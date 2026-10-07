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
from torch.utils.data import Dataset, DataLoader

from tokenizer.tokenizer import encode


# =========================================================
# SETTINGS
# =========================================================

DATA_FILE = BACKEND_DIR / "data" / "train.txt"

SEQUENCE_LENGTH = 4
BATCH_SIZE = 4


# =========================================================
# DATASET
# =========================================================

class TextDataset(Dataset):

    def __init__(self, text, sequence_length):

        token_ids = encode(text)

        self.inputs = []
        self.targets = []

        for i in range(
            len(token_ids) - sequence_length
        ):

            input_sequence = token_ids[
                i:i + sequence_length
            ]

            target_sequence = token_ids[
                i + 1:i + sequence_length + 1
            ]

            self.inputs.append(
                torch.tensor(
                    input_sequence,
                    dtype=torch.long
                )
            )

            self.targets.append(
                torch.tensor(
                    target_sequence,
                    dtype=torch.long
                )
            )

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        return (
            self.inputs[index],
            self.targets[index]
        )


# =========================================================
# LOAD TRAINING TEXT
# =========================================================

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


# =========================================================
# CREATE DATASET
# =========================================================

dataset = TextDataset(
    text,
    SEQUENCE_LENGTH
)


print("Dataset size:")
print(len(dataset))


# =========================================================
# CREATE DATALOADER
# =========================================================

data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


print("\nBatch size:")
print(BATCH_SIZE)


# =========================================================
# TEST FIRST BATCH
# =========================================================

for batch_inputs, batch_targets in data_loader:

    print("\nBatch input shape:")
    print(batch_inputs.shape)

    print("\nBatch target shape:")
    print(batch_targets.shape)

    print("\nBatch inputs:")
    print(batch_inputs)

    print("\nBatch targets:")
    print(batch_targets)

    break