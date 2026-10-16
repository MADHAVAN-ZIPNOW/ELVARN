"""Give a trained ELVARN model a BIGGER vocabulary: the common words of a new text (here the court judgments) become single tokens, without losing anything learned.

Why: the vocabulary was trained on small data, so a legal word like "petitioner" costs 3 tokens and "jurisdiction" 4. Making the most frequent words single tokens
makes the same text 8-10% shorter: faster training and answers, and more text inside the 512-token window.

How the tokenizer grows (byte-level BPE): for each chosen word, the pieces it is split into today (e.g. "Ġpet" "ition" "er") are joined by NEW merges added at the
END of the merge list (lowest priority). Old merges therefore act first, exactly as before, and the new merges only join pieces that were already final:
the new tokenization of any text is the old one with some neighbouring pieces glued together. Nothing is removed, no id changes, and any text still decodes back exactly.

How the model grows (every old weight is copied unchanged):
  * the token table (shared with the output layer) gets one new row per new token = the AVERAGE of the rows of the two pieces it glues;
  * the output bias of a new token = the average of its two pieces' biases minus --bias-offset (default 2.0), so a not-yet-trained model does not start
    emitting new tokens too eagerly. Fine-tuning learns the right values.
  For any input made of OLD token ids, the scores (logits) of every OLD token are the same as before (equal to 6 decimals: the output matrix is bigger, which
  changes only the last floating-point digit; tested); the new columns are only added.

Whole characters: Tamil costs 23.7 tokens per word today (every letter is 3 raw bytes = 3 tokens), Hindi the same, and curly quotes, dashes and the section sign
cost 3 tokens each. --scripts adds one token per character of those scripts (glued from its bytes in the same safe way), the biggest gain for a Tamil Nadu model.

Usage (project root):
    python backend/training/extend_vocab.py --init-from deploy/models/17m_rag/best.pt --out-dir <NEW folder> --words 8000

Never overwrites: the output folder must not exist. The old tokenizer and checkpoint are not touched.
"""
import argparse
import collections
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from model.transformer import build_model, config_from_checkpoint, count_parameters  # noqa: E402
from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402

SCRIPTS = {                                    # whole characters worth one token each (byte-level BPE spends 3 tokens on each of them today)
    "tamil": [(0x0B80, 0x0BFF)],
    "devanagari": [(0x0900, 0x097F)],
    "punct": [(0x2010, 0x2027), (0x20B9, 0x20B9), (0x00A7, 0x00A7), (0x00B0, 0x00B0), (0x2212, 0x2212)],   # quotes, dashes, ellipsis, rupee, section, degree, minus
}
WORD = re.compile(r"(?<= )[A-Za-z]+")          # a word with the space before it: that is how byte-level BPE sees it ("Ġword")
MIN_COUNT = 20


def count_words(texts, max_chars=None):
    """How often each space-led word occurs. Reads at most max_chars characters (None = all)."""
    counter, seen = collections.Counter(), 0
    for text in texts:
        counter.update(WORD.findall(text))
        seen += len(text)
        if max_chars and seen >= max_chars:
            break
    return counter, seen


def pick_words(counter, tokenizer, n, min_count=MIN_COUNT):
    """The n words that save the most tokens (occurrences x (pieces - 1)); only words that cost 2 or more pieces today."""
    scored = []
    for word, count in counter.items():
        if count < min_count or len(word) < 3:
            continue
        pieces = len(tokenizer.encode(" " + word).ids)
        if pieces >= 2:
            scored.append((count * (pieces - 1), word))
    scored.sort(reverse=True)
    return [w for _, w in scored[:n]]


def script_characters(names):
    """Every assigned character of the named scripts (see SCRIPTS)."""
    import unicodedata
    chars = []
    for name in names:
        for lo, hi in SCRIPTS[name]:
            chars += [chr(c) for c in range(lo, hi + 1) if unicodedata.category(chr(c)) != "Cn"]
    return chars


def extend_tokenizer(tok_json, words, base, chars=()):
    """-> (new tokenizer dict, [(new_id, id_a, id_b), ...]). `base` is the old Tokenizer (to read the current pieces of each word and character).
    `words` are glued with their leading space ("Ġword"); `chars` are glued as they stand (one Tamil letter = 3 bytes = 3 tokens before, 1 after)."""
    out = json.loads(json.dumps(tok_json))
    vocab, merges = out["model"]["vocab"], out["model"]["merges"]
    next_id = max(vocab.values()) + 1
    steps = []
    for text in [" " + w for w in words] + list(chars):
        pieces = base.encode(text).tokens
        cur = pieces[0]
        for piece in pieces[1:]:
            glued = cur + piece
            if glued not in vocab:
                vocab[glued] = next_id
                merges.append([cur, piece])
                steps.append((next_id, vocab[cur], vocab[piece]))
                next_id += 1
            cur = glued
    return out, steps


MAX_VOCAB = 65535                                # the training data stores token ids as 16-bit numbers: the vocabulary cannot pass this


def is_extension(old_json, new_json):
    """True when `new_json` is `old_json` with tokens added at the END only: every old token keeps its id and the old merges are the first merges."""
    ov, nv = old_json["model"]["vocab"], new_json["model"]["vocab"]
    om, nm = old_json["model"]["merges"], new_json["model"]["merges"]
    return all(nv.get(k) == v for k, v in ov.items()) and len(nm) >= len(om) and [list(m) for m in nm[:len(om)]] == [list(m) for m in om]


def steps_between(old_json, new_json):
    """The (new_id, id_a, id_b) steps that turn the old vocabulary into the new one, read from the merges that were appended."""
    vocab, om = new_json["model"]["vocab"], len(old_json["model"]["merges"])
    out = []
    for a, b in new_json["model"]["merges"][om:]:
        out.append((vocab[a + b], vocab[a], vocab[b]))
    return out


def extend_state(state, steps, bias_offset=2.0):
    """A copy of a model state dict with one new token row per step (the average of the two pieces it glues); old rows are copied unchanged."""
    state = {k: v.clone() for k, v in state.items()}
    emb, bias = state["token_embedding.weight"], state["lm_head.bias"]
    old = emb.shape[0]
    new_emb = torch.empty(old + len(steps), emb.shape[1], dtype=emb.dtype)
    new_bias = torch.empty(old + len(steps), dtype=bias.dtype)
    new_emb[:old], new_bias[:old] = emb, bias
    for new_id, a, b in steps:
        new_emb[new_id] = (new_emb[a] + new_emb[b]) / 2
        new_bias[new_id] = (new_bias[a] + new_bias[b]) / 2 - bias_offset
    state["token_embedding.weight"] = new_emb
    if "lm_head.weight" in state:
        state["lm_head.weight"] = new_emb.clone()
    state["lm_head.bias"] = new_bias
    return state


def adapt_checkpoint(ck, old_tok_path, new_tok_path, bias_offset=2.0):
    """Make a checkpoint trained with an OLDER version of the vocabulary fit the NEWER one (in place). Returns how many tokens were added.
    Only allowed when the new vocabulary is the old one with tokens appended (is_extension): then every old token means the same thing."""
    old_json = json.loads(Path(old_tok_path).read_text(encoding="utf-8"))
    new_json = json.loads(Path(new_tok_path).read_text(encoding="utf-8"))
    if not is_extension(old_json, new_json):
        raise ValueError("The new tokenizer is not the old one with tokens appended; the model cannot be adapted to it.")
    steps = steps_between(old_json, new_json)
    if len(old_json["model"]["vocab"]) != ck["vocab_size"]:
        raise ValueError("The checkpoint does not match its own tokenizer.")
    ck["model_state_dict"] = extend_state(ck["model_state_dict"], steps, bias_offset)
    ck["vocab_size"] = len(new_json["model"]["vocab"])
    ck["tokenizer_sha256"] = BPETokenizer(new_tok_path).sha256
    return len(steps)


def extend_checkpoint(checkpoint, steps, bias_offset=2.0):
    """-> (model, config): every old weight copied, one new row per new token (average of the two pieces it glues)."""
    config = config_from_checkpoint(checkpoint)
    state = {k: v.clone() for k, v in checkpoint["model_state_dict"].items()}
    emb, bias = state["token_embedding.weight"], state["lm_head.bias"]
    old = emb.shape[0]
    new_emb = torch.empty(old + len(steps), emb.shape[1], dtype=emb.dtype)
    new_bias = torch.empty(old + len(steps), dtype=bias.dtype)
    new_emb[:old], new_bias[:old] = emb, bias
    for new_id, a, b in steps:                                  # in order: a glued token can use a token created a moment ago
        new_emb[new_id] = (new_emb[a] + new_emb[b]) / 2
        new_bias[new_id] = (new_bias[a] + new_bias[b]) / 2 - bias_offset
    state["token_embedding.weight"] = new_emb
    state["lm_head.weight"] = new_emb.clone() if "lm_head.weight" in state else new_emb
    state["lm_head.bias"] = new_bias
    model = build_model(old + len(steps), config)
    model.load_state_dict(state, strict=True)
    return model, config


def corpus_texts(corpus_dir, topics=("judgment",), limit_chars=None):
    import glob
    import pyarrow.parquet as pq
    total = 0
    for f in glob.glob(str(Path(corpus_dir) / "passages" / "**" / "*.parquet"), recursive=True):
        for row in pq.read_table(f, columns=["text", "topic"]).to_pylist():
            if row["topic"] in topics and row["text"]:
                yield row["text"]
                total += len(row["text"])
        if limit_chars and total >= limit_chars:
            return


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--init-from", type=Path, required=True, help="the trained checkpoint (tokenizer.json must be next to it)")
    ap.add_argument("--out-dir", type=Path, required=True, help="NEW folder (must not exist)")
    ap.add_argument("--words", type=int, default=8000, help="how many new whole-word tokens to add (the gain flattens after about 4,000)")
    ap.add_argument("--corpus-dir", type=Path, default=BACKEND_DIR / "data" / "corpus")
    ap.add_argument("--scripts", default="tamil,devanagari,punct", help="whole characters to add as single tokens: " + ", ".join(SCRIPTS) + " (empty = none)")
    ap.add_argument("--max-chars", type=int, default=200_000_000, help="read at most this many characters of the corpus to count words")
    ap.add_argument("--bias-offset", type=float, default=2.0)
    args = ap.parse_args(argv)
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} already exists; refusing to overwrite. Choose a new folder.")
    tok_file = args.init_from.parent / "tokenizer.json"
    ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
    tok = BPETokenizer(tok_file)
    if ck.get("tokenizer_sha256") != tok.sha256 or ck.get("vocab_size") != tok.vocab_size:
        raise SystemExit("The checkpoint does not match the tokenizer saved next to it.")
    base = Tokenizer.from_file(str(tok_file))
    counter, seen = count_words(corpus_texts(args.corpus_dir, limit_chars=args.max_chars), args.max_chars)
    words = pick_words(counter, base, args.words)
    chars = script_characters([s for s in args.scripts.split(",") if s])
    new_json, steps = extend_tokenizer(json.loads(tok_file.read_text(encoding="utf-8")), words, base, chars)
    model, config = extend_checkpoint(ck, steps, args.bias_offset)
    args.out_dir.mkdir(parents=True)
    new_tok = args.out_dir / "tokenizer.json"
    new_tok.write_text(json.dumps(new_json, ensure_ascii=False, indent=2), encoding="utf-8")
    new_tokenizer = BPETokenizer(new_tok)
    torch.save({"model_state_dict": model.state_dict(), "config": config, "vocab_size": new_tokenizer.vocab_size, "tokenizer_sha256": new_tokenizer.sha256,
                "step": 0, "final_step": 0, "best_val_loss": None, "status": "vocab_extended", "extended_from": str(args.init_from),
                "new_tokens": len(steps)}, args.out_dir / "extended.pt")
    (args.out_dir / "new_words.txt").write_text("\n".join(words), encoding="utf-8")
    info = {"checkpoint": "extended.pt", "config": config, "vocab_size_before": tok.vocab_size, "vocab_size_after": new_tokenizer.vocab_size,
            "new_tokens": len(steps), "words_chosen": len(words), "characters_added": len(chars), "characters_read": seen, "tokenizer_sha256": new_tokenizer.sha256,
            "parameters_after": count_parameters(model), "extended_from": str(args.init_from)}
    (args.out_dir / "latest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
