"""Convert two openly licensed datasets into ELVARN's block format.

  databricks/databricks-dolly-15k   (CC BY-SA 3.0)  -> raw_v2/dolly_qa.txt       <GENERAL_ENGLISH>  Question / Answer
  OpenAssistant/oasst1              (Apache 2.0)    -> raw_v2/oasst_chat.txt     <CASUAL_CONVERSATION>  User / Assistant

Only short, plain exchanges are kept (the model is tiny): no context passages, English only, reviewed and
not deleted. Licences and sources are listed in DATA_SOURCES.md. The raw files are not committed to git.

Usage (project root):
    python backend/data_tools/convert_open_datasets.py
"""
import argparse
import gzip
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

BACKEND_DIR = Path(__file__).resolve().parent.parent
KEEP_DOLLY = {"open_qa", "general_qa", "brainstorming", "creative_writing", "classification"}


# Keep the data wholesome and plain: no explicit/violent/hateful text, no lyrics/code/links.
BLOCK = re.compile(
    r"\b(?:sex|sexual|sexy|porn|porno|pornography|nude|naked|erotic|fuck|fucking|shit|bitch|cock|dick|pussy|rape|raped|rapist|kill|killed|killing|killer|murder|murdered|suicide|suicidal|nazi|nazis|racist|racism|slur|drug|drugs|cocaine|heroin|weed|bomb|bombs|bombing|terrorist|terrorism|weapon|weapons|gun|guns|hack|hacker|hacking|lyrics|greentext|4chan|nsfw|onlyfans|dating|boyfriend|girlfriend|chorus|verse)\b|def |import |```|https?://|www\.|\[lyrics\]|<\w+>",
    re.I)
STRONG = re.compile(r"fuck|porn|sexual|sexy|pussy|nigg|whore", re.I)      # blocked even inside longer words


def acceptable(*texts):
    return not any(BLOCK.search(t) or STRONG.search(t) for t in texts)


def clean(t):
    t = re.sub(r"[<>]", "", t)
    t = re.sub(r"\[\d+\]", "", t)                       # [1] citation marks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def dolly(out_dir):
    path = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    kept = chars = 0
    with open(out_dir / "dolly_qa.txt", "w", encoding="utf-8") as fh:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["category"] not in KEEP_DOLLY or r["context"].strip():
                continue
            q, a = clean(r["instruction"]), clean(r["response"])
            if not (8 <= len(q) <= 300 and 20 <= len(a) <= 900) or not acceptable(q, a):
                continue
            block = f"<GENERAL_ENGLISH>\nQuestion: {q}\nAnswer: {a}\n</GENERAL_ENGLISH>\n\n"
            fh.write(block); kept += 1; chars += len(block)
    print(f"dolly: {kept} Q&A blocks, about {chars / 4.2 / 1e6:.2f}M tokens (estimated)")


def walk(node, path, out):
    """Follow the best-ranked reply at every level to build one conversation per tree."""
    path = path + [node]
    replies = [r for r in node.get("replies", []) if not r.get("deleted") and r.get("review_result")]
    if not replies:
        out.append(path)
        return
    replies.sort(key=lambda r: (r.get("rank") is None, r.get("rank") if r.get("rank") is not None else 99))
    walk(replies[0], path, out)


def oasst(out_dir):
    path = hf_hub_download("OpenAssistant/oasst1", "2023-04-12_oasst_ready.trees.jsonl.gz", repo_type="dataset")
    kept = chars = 0
    with gzip.open(path, "rt", encoding="utf-8") as src, open(out_dir / "oasst_chat.txt", "w", encoding="utf-8") as fh:
        for line in src:
            tree = json.loads(line)
            root = tree["prompt"]
            if root.get("lang") != "en" or root.get("deleted") or not root.get("review_result"):
                continue
            chains = []
            walk(root, [], chains)
            for chain in chains:
                turns = []
                for m in chain[:6]:
                    if m.get("lang") != "en" or len(m["text"]) > 350 or not acceptable(m["text"]):
                        break
                    who = "User" if m["role"] == "prompter" else "Assistant"
                    turns.append(f"{who}: {clean(m['text']).replace(chr(10), ' ')}")
                while turns and not turns[-1].startswith("Assistant"):
                    turns.pop()
                if len(turns) < 2:
                    continue
                block = "<CASUAL_CONVERSATION>\n" + "\n".join(turns) + "\n</CASUAL_CONVERSATION>\n\n"
                fh.write(block); kept += 1; chars += len(block)
    print(f"oasst: {kept} conversations, about {chars / 4.2 / 1e6:.2f}M tokens (estimated)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=BACKEND_DIR / "data" / "raw_v2")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dolly(a.out_dir)
    oasst(a.out_dir)
