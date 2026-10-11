"""ELVARN byte-level BPE tokenizer (fixed vocabulary).

The vocabulary is trained once by train_bpe.py and saved to
tokenizer/bpe/tokenizer.json. This module only loads and uses it; it never
adds ids. Byte-level BPE can represent any text, so unseen words are split
into smaller pieces instead of becoming <UNK>.
"""
import hashlib
from pathlib import Path

from tokenizers import Tokenizer

BPE_DIR = Path(__file__).resolve().parent / "bpe"
TOKENIZER_FILE = BPE_DIR / "tokenizer.json"

# Ids 0-3 match the legacy word tokenizer.
SPECIAL_TOKENS = [
    "<PAD>", "<UNK>", "<BOS>", "<EOS>",
    "<DOMAIN_KNOWLEDGE>", "<GENERAL_ENGLISH>",
    "<CASUAL_CONVERSATION>", "<HUMOUR>",
]


class BPETokenizer:
    def __init__(self, path=TOKENIZER_FILE):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"BPE tokenizer not found: {path}. "
                "Run: python backend/tokenizer/train_bpe.py"
            )
        self.path = path
        self._tok = Tokenizer.from_file(str(path))
        # Hash with line endings normalised, so a checkout that turned LF into CRLF (Windows git) still matches.
        self.sha256 = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        self.pad_id = self._tok.token_to_id("<PAD>")
        self.unk_id = self._tok.token_to_id("<UNK>")
        self.bos_id = self._tok.token_to_id("<BOS>")
        self.eos_id = self._tok.token_to_id("<EOS>")
        if None in (self.pad_id, self.unk_id, self.bos_id, self.eos_id):
            raise ValueError("Tokenizer file is missing a required special token.")

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def token_id(self, token):
        return self._tok.token_to_id(token)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def tokens(self, text):
        return self._tok.encode(text).tokens
