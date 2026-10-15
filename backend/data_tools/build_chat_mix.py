"""Build the fine-tuning set: chat conversations + notes/answer examples (including the refreshed knowledge-base passages).

  chat   backend/data/chat_v1/conversations.jsonl  (collect_chat_data.py) -> windows of up to 3 exchanges in the live chat format
            <CASUAL_CONVERSATION>\nUser: ...\nAssistant: ...\nUser: ...\nAssistant: ...      (+ <BOS> / <EOS>)
  notes  backend/data/rag_v2/{train,val}.bin  (build_rag_dataset.py, reads wikipedia_*.txt AND kb_*.txt) -> a random share of the
            documents, so the model keeps answering from notes and keeps seeing the knowledge-base text.

Both parts are shuffled together at document level. Validation = the fixed 5 % of conversations + the notes validation topics (never
trained on). Besides the combined val.bin, val_chat.bin and val_notes.bin are written so each skill can be measured on its own.

Usage (project root):
    python backend/data_tools/build_chat_mix.py --out-dir backend/data/chat_mix_v1 --chat-tokens 1800000 --notes-share 0.5
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402

TAG = "<CASUAL_CONVERSATION>"
MAX_TOKENS = 512
WINDOW_EXCHANGES = 3


def windows(turns):
    """Up to WINDOW_EXCHANGES exchanges per window, stride 2 (a long conversation gives several overlapping windows)."""
    ex = [(turns[i][1], turns[i + 1][1]) for i in range(0, len(turns) - 1, 2)]
    out = []
    for start in range(0, len(ex), 2):
        chunk = ex[start:start + WINDOW_EXCHANGES]
        out.append(chunk)
        if start + WINDOW_EXCHANGES >= len(ex):
            break
    return out


def chat_text(exchanges):
    return TAG + "\n" + "\n".join(f"User: {u}\nAssistant: {a}" for u, a in exchanges)


def split_documents(arr, bos_id):
    """The notes .bin is documents back to back, each starting with <BOS>."""
    starts = np.flatnonzero(arr == bos_id)
    ends = list(starts[1:]) + [len(arr)]
    return [arr[s:e] for s, e in zip(starts, ends)]


def build(out_dir, chat_tokens, notes_share, seed, tok, conv_path, notes_dir, boost=None, cap=None):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; refusing to overwrite. Choose a new folder.")
    rng = random.Random(seed)
    docs = {"train": [], "val": []}
    by_source = Counter()
    for line in open(conv_path, encoding="utf-8"):
        c = json.loads(line)
        for w in windows(c["turns"]):
            enc = tok.encode(chat_text(w), add_bos=True, add_eos=True)
            if len(enc) <= MAX_TOKENS:
                docs[c["split"]].append((c["source"], np.asarray(enc, dtype=np.uint16)))
    # chat: keep a random share of the training windows up to the token budget (all sources stay represented)
    rng.shuffle(docs["train"])
    # ELVARN's own greetings / small talk, the everyday chats and OpenAssistant always go in first; the big persona and soda sets fill the rest
    for src, n in (cap or {}).items():                    # keep at most n windows of a big source
        keep, seen_n = [], 0
        for d in docs["train"]:
            if d[0] == src:
                seen_n += 1
                if seen_n > n:
                    continue
            keep.append(d)
        docs["train"] = keep
    for src, times in (boost or {}).items():              # show a small, important source several times
        docs["train"] += [d for d in docs["train"] if d[0] == src] * (times - 1)
    first = [d for d in docs["train"] if d[0] in ("synthetic", "everyday", "oasst2")]
    rest = [d for d in docs["train"] if d[0] not in ("synthetic", "everyday", "oasst2")]
    kept, total = [], 0
    for src, a in first + rest:
        if total >= chat_tokens:
            break
        kept.append((src, a))
        total += len(a)
        by_source[src] += 1
    chat_train = [a for _, a in kept]
    chat_val = [a for _, a in docs["val"]][:3000]

    notes = {}
    for split in ("train", "val"):
        arr = np.fromfile(Path(notes_dir) / f"{split}.bin", dtype=np.uint16)
        d = split_documents(arr, tok.bos_id)
        if split == "train":
            rng.shuffle(d)
            d = d[: int(len(d) * notes_share)]
        notes[split] = d

    def write(name, parts):
        arr = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)
        if arr.size and int(arr.max()) >= tok.vocab_size:
            raise SystemExit("Token id outside the vocabulary.")
        arr.tofile(out_dir / f"{name}.bin")
        return int(arr.size)

    out_dir.mkdir(parents=True)
    train_docs = chat_train + notes["train"]
    rng.shuffle(train_docs)
    val_docs = chat_val + notes["val"]
    rng.shuffle(val_docs)
    meta = {
        "tokenizer_sha256": tok.sha256, "vocab_size": tok.vocab_size, "format": "chat_mix_v1", "seed": seed,
        "format_description": "Documents are <BOS> + text + <EOS>. Chat documents: " + TAG + " then User/Assistant turns (up to 3 exchanges). "
                              "Notes documents: the live web-notes prompt + answer (rag_v2, includes the knowledge-base passages).",
        "train": {"documents": len(train_docs), "chat_documents": len(chat_train), "notes_documents": len(notes["train"]),
                  "chat_by_source": dict(by_source)},
        "val": {"documents": len(val_docs), "chat_documents": len(chat_val), "notes_documents": len(notes["val"])},
    }
    meta["train"]["tokens"] = write("train", train_docs)
    meta["val"]["tokens"] = write("val", val_docs)
    meta["val"]["chat_tokens"] = write("val_chat", chat_val)
    meta["val"]["notes_tokens"] = write("val_notes", notes["val"])
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chat-tokens", type=int, default=1_800_000)
    ap.add_argument("--notes-share", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boost", nargs="*", default=[], help="source=times, e.g. synthetic=4")
    ap.add_argument("--cap", nargs="*", default=[], help="source=max_windows, e.g. persona=6000")
    ap.add_argument("--conversations", type=Path, default=BACKEND_DIR / "data" / "chat_v1" / "conversations.jsonl")
    ap.add_argument("--notes-dir", type=Path, default=BACKEND_DIR / "data" / "rag_v2")
    ap.add_argument("--tokenizer-file", type=Path, default=BACKEND_DIR / "data" / "v2" / "tokenizer" / "tokenizer.json")
    a = ap.parse_args()
    tok = BPETokenizer(a.tokenizer_file)
    kv = lambda items: {k: int(v) for k, v in (i.split('=') for i in items)}
    m = build(a.out_dir, a.chat_tokens, a.notes_share, a.seed, tok, a.conversations, a.notes_dir, boost=kv(a.boost), cap=kv(a.cap))
    print(json.dumps(m, indent=2))
