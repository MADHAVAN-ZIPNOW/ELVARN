"""Build the fixed retrieval benchmark: 100 questions whose correct evidence is known (written once to backend/eval/retrieval_bench.json).

  * 70 "section" questions: "What does Section N of the <Act> provide?" ... with a real sentence from a judgment as the gold evidence;
  * 30 "reference" questions: "What is X?" with the defining sentence of a Wikipedia article or project note as the gold evidence.
Every question carries what a correct hit must contain: the gold sentence (strict) and the exact section + Act words / the topic words (relevant).
No language model is used; a fixed seed makes the file repeatable. The file is written once and never overwritten (the benchmark must not drift).

    python backend/eval/build_retrieval_bench.py
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
sys.path.insert(0, str(BACKEND_DIR / "data_tools"))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR.parent))

import build_graphrag_dataset as G  # noqa: E402

OUT_V1, OUT_V2 = HERE / "retrieval_bench.json", HERE / "retrieval_bench_v2.json"
SECTION_QUESTIONS, REFERENCE_QUESTIONS = 70, 30


def main():
    unique = "--unique" in sys.argv                                  # v2: only sections whose evidence is (nearly) unique in the whole knowledge base
    OUT = OUT_V2 if unique else OUT_V1
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists: the benchmark is fixed and is never overwritten.")
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR.parent / ".env")
    from knowledge.kb import KnowledgeBase
    kb = KnowledgeBase.from_env()
    rng = random.Random(7)
    cases, seen = [], set()
    mined = G.mine_judgments(kb, rng, limit=1000000, per_pair=1000 if unique else 1)
    count = {}
    for item in mined:
        count[item["sid"]] = count.get(item["sid"], 0) + 1
    for item in mined:
        if item["sid"] in seen or len(item["gold"].split()) < 12 or (unique and count[item["sid"]] != 1):
            continue
        seen.add(item["sid"])
        cases.append({"kind": "section", "question": item["question"], "gold": item["gold"], "gold_title": item["gold_title"],
                      "section": item["num"], "act": item["act"]})
        if len(cases) == SECTION_QUESTIONS:
            break
    refs = G.mine_reference(kb, rng, 400)
    rng.shuffle(refs)
    for item in refs[:REFERENCE_QUESTIONS]:
        cases.append({"kind": "reference", "question": item["question"], "gold": item["gold"], "gold_title": item["gold_title"], "topic": item["topic"]})
    kb.close()
    for i, c in enumerate(cases, 1):
        c["id"] = i
    OUT.write_text(json.dumps({"description": "ELVARN retrieval benchmark: question + the evidence a correct hit must contain.", "cases": cases},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
