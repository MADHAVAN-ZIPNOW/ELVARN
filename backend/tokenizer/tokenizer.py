import re
import json
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
VOCAB_FILE = BASE_DIR / "vocab.json"


# =========================================================
# LOAD VOCABULARY
# =========================================================

with open(VOCAB_FILE, "r", encoding="utf-8") as file:
    vocab = json.load(file)


id_to_token = {
    token_id: token
    for token, token_id in vocab.items()
}


# =========================================================
# SPECIAL TOKENS
# =========================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

PAD_ID = vocab[PAD_TOKEN]
UNK_ID = vocab[UNK_TOKEN]
BOS_ID = vocab[BOS_TOKEN]
EOS_ID = vocab[EOS_TOKEN]


# =========================================================
# PREPARE MULTI-WORD VOCABULARY
# =========================================================

multi_word_tokens = [
    token
    for token in vocab
    if " " in token
    and not token.startswith("<")
]

# Longest phrases first.
# Example:
#
# Tamil Nadu Legislative Assembly
# Tamil Nadu Legislative
# Tamil Nadu
#
# The longest matching phrase gets priority.

multi_word_tokens.sort(
    key=lambda token: len(token),
    reverse=True
)


# =========================================================
# TOKENIZER
# =========================================================

def tokenizer(text):

    if not text:
        return []

    remaining_text = text
    tokens = []

    while remaining_text:

        # -------------------------------------------------
        # Remove leading whitespace
        # -------------------------------------------------

        whitespace_match = re.match(
            r"\s+",
            remaining_text,
            re.UNICODE
        )

        if whitespace_match:
            remaining_text = remaining_text[
                whitespace_match.end():
            ]
            continue

        # -------------------------------------------------
        # Try longest multi-word vocabulary token
        # -------------------------------------------------

        matched_phrase = None

        for phrase in multi_word_tokens:

            if remaining_text.startswith(phrase):

                end_position = len(phrase)

                # Make sure phrase does not match only the
                # beginning of a larger word.

                if (
                    end_position == len(remaining_text)
                    or remaining_text[end_position].isspace()
                    or not (
                        phrase[-1].isalnum()
                        and remaining_text[end_position].isalnum()
                    )
                ):
                    matched_phrase = phrase
                    break

        if matched_phrase is not None:

            tokens.append(matched_phrase)

            remaining_text = remaining_text[
                len(matched_phrase):
            ]

            continue

        # -------------------------------------------------
        # Otherwise tokenize one word
        # or one punctuation symbol
        # -------------------------------------------------

        match = re.match(
            r"\w+|[^\w\s]",
            remaining_text,
            re.UNICODE
        )

        if match:

            token = match.group(0)

            tokens.append(token)

            remaining_text = remaining_text[
                match.end():
            ]

        else:

            # Safety fallback
            tokens.append(
                remaining_text[0]
            )

            remaining_text = remaining_text[1:]

    return tokens


# =========================================================
# ENCODE
# =========================================================

def encode(text):

    tokens = tokenizer(text)

    token_ids = []

    for token in tokens:

        if token in vocab:
            token_ids.append(
                vocab[token]
            )

        else:
            token_ids.append(
                UNK_ID
            )

    return token_ids


# =========================================================
# DECODE
# =========================================================

def decode(token_ids):

    tokens = []

    for token_id in token_ids:

        if token_id in id_to_token:

            tokens.append(
                id_to_token[token_id]
            )

        else:

            tokens.append(
                UNK_TOKEN
            )

    return " ".join(tokens)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        "Vocabulary size:",
        len(vocab)
    )

    print(
        "Multi-word vocabulary entries:",
        len(multi_word_tokens)
    )

    print(
        "\nType your text."
    )

    print(
        "Press Enter on an empty line to exit."
    )

    while True:

        text = input(
            "\nLet's send token: "
        )

        if text == "":
            break

        tokens = tokenizer(text)

        token_ids = encode(text)

        decoded_text = decode(
            token_ids
        )

        print(
            "\nTokens    :",
            tokens
        )

        print(
            "Token IDs :",
            token_ids
        )

        print(
            "Decoded   :",
            decoded_text
        )