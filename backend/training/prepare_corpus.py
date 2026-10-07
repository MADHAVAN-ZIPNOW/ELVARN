import sys
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BACKEND_DIR / "data" / "raw"
TRAIN_FILE = BACKEND_DIR / "data" / "train.txt"


# =========================================================
# SUPPORTED FILE TYPES
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md"
}


# =========================================================
# FIND FILES
# =========================================================

files = []

for file_path in RAW_DATA_DIR.rglob("*"):

    if (
        file_path.is_file()
        and file_path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    ):
        files.append(file_path)


files.sort()


print("Raw data directory:")
print(RAW_DATA_DIR)

print("\nFiles found:")
print(len(files))

for file_path in files:
    print(" -", file_path.name)


# =========================================================
# READ FILES
# =========================================================

all_text = []

for file_path in files:

    print(
        f"\nReading: {file_path.name}"
    )

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

    except UnicodeDecodeError:

        print(
            "UTF-8 failed. Trying UTF-8 with replacement."
        )

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as file:

            text = file.read()

    print(
        "Characters:",
        len(text)
    )

    all_text.append(text)


# =========================================================
# COMBINE
# =========================================================

combined_text = "\n\n".join(
    all_text
)


# =========================================================
# SAVE
# =========================================================

with open(
    TRAIN_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(combined_text)


# =========================================================
# SUMMARY
# =========================================================

print("\n===================================")
print("CORPUS PREPARATION COMPLETE")
print("===================================")

print(
    "Documents:",
    len(files)
)

print(
    "Total characters:",
    len(combined_text)
)

print(
    "Output:",
    TRAIN_FILE
)