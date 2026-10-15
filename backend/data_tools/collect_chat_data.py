"""Collect open chat datasets + ELVARN's own hand-written chat into one clean file of conversations.

Sources (all free, open licences, public files; no account needed):
  HuggingFaceTB/everyday-conversations-llama3.1-2k   Apache-2.0     short everyday chats and greetings
  google/Synthetic-Persona-Chat                       CC BY 4.0      casual multi-turn chat
  allenai/soda  (validation split)                    CC BY 4.0      social dialogues
  OpenAssistant/oasst2  (ready trees)                 Apache-2.0     assistant conversations
  chat_synthetic.py                                   ours           greetings, small talk, who ELVARN is, remembering a name

Cleaning: only short plain turns; English; no emails / phone numbers / links (personal data); the same wholesome-word filter as the
other converters; duplicates dropped by a hash of the text. Safe to run again: conversations already stored (by hash) are skipped, so
a daily run only adds what is new.

Output (not committed to git):  backend/data/chat_v1/conversations.jsonl   one line per conversation:
    {"id": "<sha1>", "source": "...", "split": "train"|"val", "turns": [["user","hi"],["assistant","Hello!"], ...]}

Usage (project root):
    python backend/data_tools/collect_chat_data.py [--limit-per-source 6000]
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "data_tools"))

from convert_open_datasets import acceptable, walk  # noqa: E402
import chat_synthetic  # noqa: E402

OUT_DIR = BACKEND_DIR / "data" / "chat_v1"
MAX_TURN_CHARS = 360
MAX_EXCHANGES = 6
EMOJI = "👋😊🙂😄😁😉🤔👍🎉✨😅🙌😂😢🤗😕🌙☀️"
PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|https?://|www\.|\+?\d[\d\s().-]{8,}\d|\b\d{10}\b", re.I)
SWAP = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": " - ", "–": "-", "…": "...", " ": " "}


def sanitize(text):
    """Plain text on one line, or None when the turn should not be used."""
    for a, b in SWAP.items():
        text = text.replace(a, b)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("<", "").replace(">", "")
    if not text or len(text) > MAX_TURN_CHARS:
        return None
    if any(ord(ch) > 127 and ch not in EMOJI and ch != "️" for ch in text):
        return None
    if PII.search(text) or not acceptable(text):
        return None
    return text


def clean_conversation(turns):
    """turns: list of (role, text). Returns a cleaned list that starts with the user and ends with the assistant, or None."""
    out = []
    for role, text in turns[: MAX_EXCHANGES * 2]:
        t = sanitize(text)
        if t is None:
            break
        if out and out[-1][0] == role:
            break
        out.append((role, t))
    if out and out[0][0] != "user":
        return None
    while out and out[-1][0] != "assistant":
        out.pop()
    if len(out) < 2 or any(len(t) < 2 for r, t in out if r == "assistant"):
        return None
    return out


def conv_id(turns):
    return hashlib.sha1("\n".join(f"{r}:{t.lower()}" for r, t in turns).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- readers: each yields lists of (role, text)
def read_everyday():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    for f in ("data/train_sft-00000-of-00001.parquet", "data/test_sft-00000-of-00001.parquet"):
        path = hf_hub_download("HuggingFaceTB/everyday-conversations-llama3.1-2k", f, repo_type="dataset")
        for row in pq.read_table(path).to_pylist():
            yield [("user" if m["role"] == "user" else "assistant", m["content"]) for m in row["messages"] if m["role"] in ("user", "assistant")]


def read_persona():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("google/Synthetic-Persona-Chat", "data/Synthetic-Persona-Chat_train.csv", repo_type="dataset")
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            text = row.get("Best Generated Conversation") or ""
            turns = []
            for line in text.split("\n"):
                m = re.match(r"\s*User ([12]):\s*(.*)", line)
                if m:
                    turns.append(("user" if m.group(1) == "1" else "assistant", m.group(2)))
            yield turns


def read_soda():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("allenai/soda", "valid.parquet", repo_type="dataset")
    for row in pq.read_table(path, columns=["dialogue"]).to_pylist():
        d = row["dialogue"]
        yield [("user" if i % 2 == 0 else "assistant", u) for i, u in enumerate(d)]


def read_oasst2():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("OpenAssistant/oasst2", "2023-11-05_oasst2_ready.trees.jsonl.gz", repo_type="dataset")
    with gzip.open(path, "rt", encoding="utf-8") as src:
        for line in src:
            root = json.loads(line)["prompt"]
            if root.get("lang") != "en" or root.get("deleted") or not root.get("review_result"):
                continue
            chains = []
            walk(root, [], chains)
            for chain in chains:
                if all(m.get("lang") == "en" for m in chain):
                    yield [("user" if m["role"] == "prompter" else "assistant", m["text"]) for m in chain]


def read_synthetic():
    yield from chat_synthetic.make(count=5000)


SOURCES = {"everyday": read_everyday, "persona": read_persona, "soda": read_soda, "oasst2": read_oasst2, "synthetic": read_synthetic}


def load_seen(path):
    seen = set()
    if path.is_file():
        for line in open(path, encoding="utf-8"):
            try:
                seen.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                pass
    return seen


def collect(limit_per_source=6000, out_dir=OUT_DIR, sources=None, seed=1, say=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "conversations.jsonl"
    seen = load_seen(path)
    rng = random.Random(seed)
    added = {}
    with open(path, "a", encoding="utf-8") as fh:
        for name in (sources or SOURCES):
            kept = scanned = 0
            try:
                items = list(SOURCES[name]())
            except Exception as exc:  # one source being down must not stop the others
                say(f"{name}: skipped ({type(exc).__name__}: {str(exc)[:100]})")
                continue
            if name != "synthetic":
                rng.shuffle(items)
            for turns in items:
                scanned += 1
                conv = clean_conversation(turns)
                if conv is None:
                    continue
                cid = conv_id(conv)
                if cid in seen:
                    continue
                seen.add(cid)
                split = "val" if int(cid, 16) % 20 == 0 else "train"          # a fixed 5 % of conversations is never trained on
                fh.write(json.dumps({"id": cid, "source": name, "split": split, "turns": conv}, ensure_ascii=False) + "\n")
                kept += 1
                if kept >= limit_per_source:
                    break
            added[name] = kept
            say(f"{name}: scanned {scanned}, added {kept}")
    return added


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-source", type=int, default=6000)
    ap.add_argument("--sources", nargs="*", choices=list(SOURCES))
    a = ap.parse_args()
    collect(a.limit_per_source, sources=a.sources)
