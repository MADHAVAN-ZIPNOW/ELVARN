"""Prepare everything needed before training (BPE pipeline).

Steps:
  1. prepare_corpus.py  - combine data/raw/* into data/train.txt (legacy, unchanged)
  2. train_bpe.py       - train the fixed BPE tokenizer on the training documents
  3. build_dataset.py   - split by document, tokenize, write data/processed/*.bin

The old word-vocabulary steps (rebuild_vocab.py, check_unknown.py) are no
longer run: the BPE tokenizer has a fixed vocabulary and no <UNK> growth.
"""
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


def run_step(title, script_path):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_DIR)
    if result.returncode != 0:
        print("\nERROR:")
        print(f"Step failed: {script_path}")
        sys.exit(result.returncode)


run_step("STEP 1 - PREPARING CORPUS", BACKEND_DIR / "training" / "prepare_corpus.py")
run_step("STEP 2 - TRAINING BPE TOKENIZER", BACKEND_DIR / "tokenizer" / "train_bpe.py")
run_step("STEP 3 - BUILDING TRAIN / VALIDATION DATA", BACKEND_DIR / "training" / "build_dataset.py")

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)
print("\nNext: python backend/training/train_batch.py")
