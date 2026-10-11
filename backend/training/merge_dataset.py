"""Merge generated part files into data/raw/domain_knowledge.txt.

Dedupes by question text, drops malformed blocks and blocks touching facts
the writers flagged as unverified, tokenizes each block with the real ELVARN
tokenizer, and fills each category up to its token budget.
Usage: python training/merge_dataset.py <parts_dir> [total_tokens]
"""
import random
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "tokenizer"))

from tokenizer import tokenizer, vocab  # noqa: E402

parts_dir = Path(sys.argv[1])
total = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
SHARE = {"DOMAIN_KNOWLEDGE": 0.60, "GENERAL_ENGLISH": 0.20,
         "CASUAL_CONVERSATION": 0.15, "HUMOUR": 0.05}

# Facts the part writers said they could not confirm.
FLAGGED = re.compile(
    r"193\(9\)|Vineet Narain|BNSS 479|Section 479|"
    r"3\.5 percent|Backward Class Muslims|50 percent (?:of )?(?:seats|women)|"
    r"MBC.{0,60}1989|1989.{0,60}MBC", re.S)

block_re = re.compile(r"<(\w+)>\n(.*?)\n</\1>", re.S)
seen, blocks = set(), {c: [] for c in SHARE}
dropped = {"dup": 0, "flagged": 0, "malformed": 0}

for f in sorted(parts_dir.glob("part_*.txt")):
    for cat, body in block_re.findall(f.read_text(encoding="utf-8")):
        if cat not in SHARE:
            dropped["malformed"] += 1
            continue
        ok = ("Question:" in body and "Answer:" in body) if cat in (
            "DOMAIN_KNOWLEDGE", "GENERAL_ENGLISH") else (
            "User:" in body and "Assistant:" in body)
        if cat == "DOMAIN_KNOWLEDGE":
            ok = ok and "Source:" in body and "Verification status:" in body
        if not ok:
            dropped["malformed"] += 1
            continue
        if FLAGGED.search(body):
            dropped["flagged"] += 1
            continue
        m = re.search(r"(?:Question|User):\s*(.+)", body)
        key = re.sub(r"\W+", " ", m.group(1).lower()).strip() + cat
        if key in seen:
            dropped["dup"] += 1
            continue
        seen.add(key)
        blocks[cat].append((f.name, body))

random.seed(7)
out, report = [], []
for cat, share in SHARE.items():
    random.shuffle(blocks[cat])
    budget, used, unk, kept = int(total * share), 0, 0, 0
    for name, body in blocks[cat]:
        toks = tokenizer(body)
        n = len(toks) + 2  # open and close markers
        if used + n > budget:
            continue
        used += n
        unk += sum(t not in vocab for t in toks)
        kept += 1
        out.append(f"<{cat}>\n{body}\n</{cat}>")
    report.append((cat, len(blocks[cat]), kept, used, budget, unk))

random.shuffle(out)
dest = BACKEND_DIR / "data" / "raw" / "domain_knowledge.txt"
dest.write_text("\n\n".join(out) + "\n", encoding="utf-8")
print("dropped:", dropped)
for cat, avail, kept, used, budget, unk in report:
    print(f"{cat:22} available={avail:5} kept={kept:5} tokens={used:7}/{budget:7} UNK%={100*unk/max(used,1):.1f}")
print("TOTAL tokens:", sum(r[3] for r in report), "->", dest)
