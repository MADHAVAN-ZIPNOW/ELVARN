"""Measure domain_knowledge.txt with the real ELVARN tokenizer.

Reports tokens and <UNK> rate per category block against the 1,000,000
token target. Usage: python training/measure_dataset.py [path]
"""
import re
import sys
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "tokenizer"))

from tokenizer import tokenizer, vocab  # noqa: E402

TARGETS = {
    "DOMAIN_KNOWLEDGE": 600_000,
    "GENERAL_ENGLISH": 200_000,
    "CASUAL_CONVERSATION": 150_000,
    "HUMOUR": 50_000,
}

path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    BACKEND_DIR / "data" / "raw" / "domain_knowledge.txt"
)
text = path.read_text(encoding="utf-8")

block_re = re.compile(r"<(\w+)>(.*?)</\1>", re.S)
totals, unks, missing = Counter(), Counter(), Counter()
blocks = Counter()

for cat, body in block_re.findall(text):
    toks = tokenizer(body)
    totals[cat] += len(toks)
    blocks[cat] += 1
    for t in toks:
        if t not in vocab:
            unks[cat] += 1
            missing[t] += 1

print(f"{'category':22}{'blocks':>8}{'tokens':>10}{'target':>10}{'left':>10}{'UNK%':>8}")
for cat, target in TARGETS.items():
    n = totals[cat]
    print(f"{cat:22}{blocks[cat]:>8}{n:>10}{target:>10}{target - n:>10}"
          f"{100 * unks[cat] / max(n, 1):>8.1f}")
print(f"{'TOTAL':22}{sum(blocks.values()):>8}{sum(totals.values()):>10}{1_000_000:>10}")
print("\nTop out-of-vocab tokens:", missing.most_common(25))
