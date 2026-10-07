import subprocess
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


def run_step(title, script_path):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR
    )

    if result.returncode != 0:
        print("\nERROR:")
        print(f"Step failed: {script_path}")
        sys.exit(result.returncode)


# ---------------------------------------------------------
# STEP 1 — Prepare corpus
# ---------------------------------------------------------

run_step(
    "STEP 1 — PREPARING CORPUS",
    BACKEND_DIR / "training" / "prepare_corpus.py"
)


# ---------------------------------------------------------
# STEP 2 — Rebuild vocabulary
# ---------------------------------------------------------

run_step(
    "STEP 2 — REBUILDING VOCABULARY",
    BACKEND_DIR / "tokenizer" / "rebuild_vocab.py"
)


# ---------------------------------------------------------
# STEP 3 — Check unknown tokens
# ---------------------------------------------------------

run_step(
    "STEP 3 — CHECKING UNKNOWN TOKENS",
    BACKEND_DIR / "training" / "check_unknown.py"
)


# ---------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)

print("\nRaw files are combined.")
print("Vocabulary is updated.")
print("Unknown-token coverage is checked.")

print("\nReady for model training.")