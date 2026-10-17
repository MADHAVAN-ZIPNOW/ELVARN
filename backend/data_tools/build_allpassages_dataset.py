"""Build a fine-tuning set from EVERY passage of the knowledge base: one example per passage.

Each example is a keyword question taken from one sentence of the passage, plus the notes the live site would show:
  * answerable (about 75%): that sentence is one of three notes; the other two are real neighbours of the passage in Chroma (the notes a
    related search returns) that do not answer the question; the answer is the sentence, with its [n];
  * not found (about 25%): three real neighbours, none of them answering -> an honest refusal.
The neighbours come from Chroma with the passage's stored embedding (no embedding model runs), so the build is fast.
Split: 5 % of the SOURCES (a judgment or an article) are validation, never trained on.

Usage (project root, the knowledge base and Chroma running):
    python backend/data_tools/build_allpassages_dataset.py --out-dir backend/data/allpassages_v8 [--limit 20000]
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
for sub in ("", "training", "data_tools"):
    sys.path.insert(0, str(BACKEND_DIR / sub) if sub else str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR.parent))

import build_rag_dataset as R  # noqa: E402
from search.prompt import KB_HEADER  # noqa: E402
from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402

MODE_MIX = [("chat", 0.5), ("domain", 0.3), ("english", 0.2)]
NEGATIVE_SHARE = 0.25
VAL_PERCENT = 5
PAGE = 2000
FORMS = ["What is said about {k}?", "What do the notes say about {k}?", "Tell me about {k}.", "What does the text say about {k}?",
         "What is mentioned about {k}?", "Give me the key point about {k}."]
GENERIC = {"court", "appeal", "section", "petition", "held", "case", "appellant", "respondent", "learned", "counsel", "order", "judgment", "high",
           "supreme", "state", "india", "also", "which", "their", "there", "would", "shall", "where", "other", "after", "before"}


def words(text):
    return re.findall(r"[a-z][a-z0-9'-]{3,}", text.lower())


def keywords(sentence, df, rng, count=3):
    """The `count` rarest content words of the sentence, in the order they appear."""
    cands = {w for w in words(sentence) if w not in R.STOPWORDS and w not in GENERIC}
    if len(cands) < 2:
        return None
    rare = sorted(cands, key=lambda w: (df.get(w, 0), rng.random()))[:count]
    return [w for w in dict.fromkeys(words(sentence)) if w in rare]


def covers(keys, text, share=0.67):
    have = set(words(text))
    return sum(1 for k in keys if k in have) >= max(2, round(share * len(keys)))


def doc_frequency(texts):
    df = Counter()
    for t in texts:
        df.update(set(words(t)))
    return df


def neighbour_notes(hits, keys, own_source, count):
    """Up to `count` single-sentence notes from the neighbours that do not answer the question (a different source, not covering the keywords)."""
    out = []
    for text, meta in hits:
        if meta.get("source_id") == own_source:
            continue
        sents = [x for x in (R.clean_sentence(s) for s in R.split_sentences(text)) if R.valid_note(x)]
        have = set(keys)
        sents.sort(key=lambda x: -len(have & set(words(x))))
        if sents and not covers(keys, sents[0]) and not any(sents[0] in o for o in out):
            out.append(sents[0])
        if len(out) == count:
            break
    return out


def build(out_dir, tok, kb, rag, limit, seed):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; refusing to overwrite.")
    rng = random.Random(seed)
    sample = kb._read("MATCH (p:Passage) RETURN p.text AS t LIMIT 40000")
    df = doc_frequency(r["t"] for r in sample)
    print(f"document frequencies from {len(sample)} passages, {len(df)} words", flush=True)
    docs, stats, examples = {"train": [], "val": []}, Counter(), {"train": [], "val": []}
    done = 0
    for skip in range(0, 1_000_000, PAGE):
        rows = kb._read("MATCH (s:Source)<-[:FROM]-(p:Passage) RETURN p.id AS id, p.text AS t, s.id AS sid, s.doc_type AS dt "
                        "ORDER BY p.id SKIP $skip LIMIT $limit", skip=skip, limit=PAGE)
        if not rows:
            break
        got = rag.col.get(ids=[r["id"] for r in rows], include=["embeddings"])
        emb = {i: e for i, e in zip(got["ids"], got["embeddings"])}
        rows = [r for r in rows if r["id"] in emb]
        if not rows:
            continue
        near = rag.col.query(query_embeddings=[emb[r["id"]] for r in rows], n_results=10, include=["documents", "metadatas"])
        for r, texts, metas in zip(rows, near["documents"], near["metadatas"]):
            if limit and done >= limit:
                break
            done += 1
            sents = [x for x in (R.clean_sentence(s) for s in R.split_sentences(r["t"])) if R.valid_note(x)]
            if not sents:
                stats["no_valid_sentence"] += 1
                continue
            gold = rng.choice(sents)
            keys = keywords(gold, df, rng)
            if not keys:
                stats["no_keywords"] += 1
                continue
            negative = rng.random() < NEGATIVE_SHARE
            others = neighbour_notes(list(zip(texts, metas)), keys, r["sid"], 3 if negative else 2)
            if len(others) < (3 if negative else 2):
                stats["too_few_neighbours"] += 1
                continue
            question = rng.choice(FORMS).format(k=" ".join(keys))
            if negative:
                notes, answer, kind = others, rng.choice(R.REFUSALS), "negative"
            else:
                pos = rng.randrange(3)
                notes = others[:pos] + [gold] + others[pos:]
                answer, kind = R._cite(rng, [gold], [0]).replace("[1]", f"[{pos + 1}]"), "answer"
            mode = R.pick(rng, MODE_MIX)
            enc = tok.encode(R.render(mode, R.evidence_text(notes, question, KB_HEADER), answer), add_bos=True, add_eos=True)
            if len(enc) > 512:
                stats["too_long"] += 1
                continue
            split = "val" if int(hashlib.sha1(r["sid"].encode()).hexdigest(), 16) % 100 < VAL_PERCENT else "train"
            docs[split].append(np.asarray(enc, dtype=np.uint16))
            stats[f"{split}_{kind}"] += 1
            if len(examples[split]) < 300:
                examples[split].append({"type": kind, "doc_type": r["dt"], "question": question, "notes": notes, "answer": answer})
        print(f"  {done} passages processed, kept {sum(len(v) for v in docs.values())}", flush=True)
        if limit and done >= limit:
            break
    out_dir.mkdir(parents=True)
    meta = {"tokenizer_sha256": tok.sha256, "vocab_size": tok.vocab_size, "format": "allpassages_v1", "seed": seed, "counts": dict(stats)}
    for split, rows in docs.items():
        arr = np.concatenate(rows) if rows else np.zeros(0, dtype=np.uint16)
        if arr.size and int(arr.max()) >= tok.vocab_size:
            raise SystemExit("Token id outside the vocabulary.")
        arr.tofile(out_dir / f"{split}.bin")
        meta[split] = {"documents": len(rows), "tokens": int(arr.size), "unk_tokens": int((arr == tok.unk_id).sum())}
        (out_dir / f"{split}_examples.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in examples[split]), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokenizer-file", type=Path, default=BACKEND_DIR / "data" / "v2" / "tokenizer" / "tokenizer.json")
    ap.add_argument("--limit", type=int, default=0, help="only the first N passages (a quick trial)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR.parent / ".env")
    from knowledge.kb import KnowledgeBase
    from knowledge.vectorstore import RagStore
    kb, rag = KnowledgeBase.from_env(), RagStore.from_env()
    meta = build(a.out_dir, BPETokenizer(a.tokenizer_file), kb, rag, a.limit, a.seed)
    kb.close()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
