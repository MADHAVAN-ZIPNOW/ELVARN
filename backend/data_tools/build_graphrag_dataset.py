"""Build the GraphRAG fine-tuning set: questions mined from the knowledge base, notes from the REAL retrieval, answers checked against those notes.

Why: a model trained on made-up notes learns to refuse (or to copy a random sentence) on the notes the live site really sends. Here every
prompt is exactly what the live chat builds: the question goes through graphrag.build_context (Chroma search, the same ranking and snippets)
and search.prompt.build_evidence_text (the same header, at most 3 notes, cut to the model's window with the model's own tokenizer).

Questions (no language model, fixed templates and a seed):
  * judgments: "Section N of the <Act>" pairs that appear in at least two judgments -> "What does Section N of the Act provide?" ...
  * reference documents (Wikipedia, project notes): "What is X?";
  * general questions the knowledge base cannot answer (football, cooking ...).
The label comes from the notes the model will SEE, never from a sentence it cannot see:
  * a visible sentence names the section (and most words of the Act) and, for "provide / hold" questions, says something about it
    -> the answer is that sentence, with its [n];
  * no visible text names the section (or the Act is a different one) -> an honest refusal ("The notes do not say.");
  * anything in between -> the example is dropped (unclear).

Usage (project root, the knowledge base and Chroma running, GPU embedding service optional):
    python backend/data_tools/build_graphrag_dataset.py --out-dir backend/data/graphrag_v7 [--limit 300]
"""
import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
for sub in ("", "training", "data_tools"):
    sys.path.insert(0, str(BACKEND_DIR / sub) if sub else str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR.parent))

import build_rag_dataset as R  # noqa: E402
from knowledge import graphrag, groundcheck  # noqa: E402
from search.prompt import KB_HEADER, build_evidence_text  # noqa: E402
from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402

MODE_MIX = [("chat", 0.5), ("domain", 0.3), ("english", 0.2)]        # the site's default mode is chat
VAL_PERCENT = 5
INJECT_SHARE = 0.55                                                 # share of questions whose gold note is placed among the real results
DISTRACTOR_SHARE = 0.30                                             # share that get a note about ANOTHER section instead (the answer is a refusal)
SEC = re.compile(r"\b(Sections?|Articles?|Rule)\s+(\d{1,3}[A-Z]?(?:\(\w{1,3}\))*)\s+(?:of|under)\s+(?:the\s+)?"
                 r"((?:[A-Z][A-Za-z’'.&-]*\s+){1,7}?(?:Act|Code|Constitution(?:\s+of\s+India)?)(?:,\s*(?:18|19|20)\d\d)?)")
CUE = re.compile(r"\b(provides?|provided|states?|stated|requires?|required|mandates?|lays? down|held|holds?|observed|ruled|means|defines?|"
                 r"empowers?|prohibits?|enacts?|stipulates?|contemplates?)\b", re.I)
FORMS = [("What does {s} of the {a} provide?", True), ("What did the Supreme Court hold about {s} of the {a}?", True),
         ("What is said about {s} of the {a}?", False), ("What do the notes say about {s} of the {a}?", False)]
GENERAL = ["Who won the football world cup in {y}?", "How do I cook {d}?", "What is the weather like in {c} today?", "Who is the president of {n}?",
           "What is the capital of {n}?", "Tell me about the planet {p}.", "How tall is {m}?", "What is the price of {t} today?"]
FILL = dict(y=["2030", "2018", "2002", "1998"], d=["biryani", "pasta", "dosa", "a cake"], c=["Paris", "Sydney", "Oslo", "Lima"],
            n=["Brazil", "Japan", "Kenya", "Chile"], p=["Neptune", "Mars", "Saturn"], m=["Mount Everest", "the Eiffel Tower", "Burj Khalifa"],
            t=["gold", "petrol", "bitcoin"])
ACT_STOP = {"act", "code", "india", "the", "and", "for", "constitution"}


def clean_text_ok(text):
    """False for scanned text with words broken by a space ("befor e", "closur e")."""
    return re.search(r"(?<![\w.'])[B-HJ-Zb-hj-z](?![\w.'])", text) is None and not re.search(r"[A-Za-z]{3,} [a-z] [a-z]{3,}", text)


def squash(text):
    return re.sub(r"\s+", "", text.lower())


def act_words(act):
    return {w for w in re.findall(r"[a-z]{4,}", act.lower()) if w not in ACT_STOP}


def mine_judgments(kb, rng, limit=12000, per_pair=4):
    """Questions about (section, act) pairs, each with a clean sentence from a judgment that says something about it (the gold note).
    Every passage of every judgment is scanned (in pages: the whole text is too big for one query)."""
    golds, page = {}, 20000
    for skip in range(0, 400000, page):
        rows = kb._read("MATCH (s:Source {doc_type: 'judgment'})<-[:FROM]-(p:Passage) RETURN s.case_name AS case, p.text AS t "
                        "ORDER BY p.id SKIP $skip LIMIT $limit", skip=skip, limit=page)
        if not rows:
            break
        for r in rows:
            if not SEC.search(r["t"]):
                continue                                      # cheap test first: splitting every passage into sentences is slow
            for sent in R.split_sentences(r["t"]):
                sent = R.clean_sentence(sent)
                m = SEC.search(sent)
                if not m or not CUE.search(sent) or not R.valid_note(sent) or not clean_text_ok(sent):
                    continue
                kind, num, act = m.group(1), m.group(2), re.sub(r"\s+", " ", m.group(3)).strip()
                if clean_text_ok(act) and 2 <= len(act.split()) <= 9:
                    golds.setdefault((kind, num, act), []).append((r["case"] or "", sent))
        print(f"  scanned {skip + len(rows)} passages, {len(golds)} section/act pairs so far", flush=True)
    items = []
    for (kind, num, act), found in golds.items():
        rng.shuffle(found)
        for case, sent in found[:per_pair]:
            tpl, strict = rng.choice(FORMS)
            sec = f"{'Section' if kind.startswith('Section') else kind.rstrip('s')} {num}"
            items.append({"kind": "judgment", "sid": f"{kind}{num}{act}", "question": tpl.format(s=sec, a=act), "num": num, "act": act,
                          "strict": strict, "gold": sent, "gold_title": case})
    rng.shuffle(items)
    return items[:limit]


def mine_reference(kb, rng, limit):
    rows = kb._read("MATCH (s:Source {doc_type: 'reference'})<-[:FROM]-(p:Passage) WHERE p.position = 0 RETURN s.id AS sid, s.title AS title, "
                    "p.text AS t LIMIT 20000")
    rng.shuffle(rows)
    items = []
    for r in rows:
        topic = re.sub(r"\s*\((Wikipedia|ELVARN project[^)]*)\)\s*$", "", r["title"] or "").strip()
        if not topic or not 1 <= len(topic.split()) <= 6 or "?" in topic or not clean_text_ok(topic):
            continue
        key = {w for w in re.findall(r"[a-z]{4,}", topic.lower())}
        gold = next((n for n in R.passage_notes(r["t"]) if key and key <= {w for w in re.findall(r"[a-z]{4,}", n.lower())}), None)
        if gold and clean_text_ok(gold):
            items.append({"kind": "reference", "sid": r["sid"], "question": rng.choice(R.DEF_Q).format(t=topic), "topic": topic,
                          "gold": gold, "gold_title": r["title"] or topic})
        if len(items) >= limit:
            break
    return items


def mine_general(rng, count):
    out, seen = [], set()
    for _ in range(count * 20):                              # bounded: there are fewer distinct combinations than `count` may ask for
        if len(out) >= count:
            break
        q = re.sub(r"\{(\w)\}", lambda m: rng.choice(FILL[m.group(1)]), rng.choice(GENERAL))
        if q not in seen:
            seen.add(q)
            out.append({"kind": "general", "sid": "general", "question": q})
    return out


def retrieve(rag, tok, question, inject=None, rng=None):
    """The live prompt body for the question and the notes the model will see: (body, [note texts], injected index or None), or None when
    nothing is retrieved. `inject` = (sentence, title): that note is placed among the real results (rank 1 to 3), as if the search had found it."""
    ctx = graphrag.build_context(None, rag, question)
    items = [i for i in ctx.items if i["kind"] == "fact" or groundcheck.usable_note(i.get("text", ""))]
    srcs = [SimpleNamespace(snippet=i["text"], title=i["source"]["title"]) for i in items]
    if inject:
        srcs = [x for x in srcs if inject[0][:60] not in x.snippet]
        srcs.insert(rng.randrange(min(3, len(srcs) + 1)), SimpleNamespace(snippet=inject[0], title=inject[1]))
    body, used = build_evidence_text(question, srcs, lambda t: len(tok.encode(t, add_bos=False)), 512, 80, header=KB_HEADER)
    if not used:
        return None
    notes = re.findall(r"^\[\d\] (.*)$", body, re.M)
    at = next((n for n, note in enumerate(notes) if inject and inject[0][:40] in note), None) if inject else None
    return body, notes, at


def best_sentence(notes, test):
    """(note index, sentence) of the first visible sentence passing `test`, or None."""
    for n, note in enumerate(notes):
        for sent in R.split_sentences(note):
            sent = R.clean_sentence(sent)
            if test(sent, note) and R.valid_note(sent):
                return n, sent
    return None


def label(item, got, rng):
    """(answer, type) or None when the example should be dropped."""
    notes = got[1]
    if got[2] is not None and not item.get("distractor"):     # the gold note was placed among the real results and is visible
        return R._cite(rng, [item["gold"]], [0]).replace("[1]", f"[{got[2] + 1}]"), "answer"
    if item["kind"] == "general":
        return rng.choice(R.REFUSALS), "negative_general"
    if item["kind"] == "judgment":
        num, words = squash(item["num"]), act_words(item["act"])
        def names_section(text):
            flat = squash(text)
            return bool(re.search(r"(?<![\d(])" + re.escape(num) + r"(?![\d(a-z])", flat) or re.search(re.escape(num) + r"\W", flat))
        def right_act(note):
            return len(words & {w for w in re.findall(r"[a-z]{4,}", note.lower())}) >= math.ceil(0.5 * max(1, len(words)))
        hit = best_sentence(notes, lambda s, note: names_section(s) and right_act(note) and (not item["strict"] or CUE.search(s)))
        if item.get("distractor"):                            # a note about ANOTHER section was placed among the results: the right answer is a refusal
            return (rng.choice(R.REFUSALS), "negative_swapped") if not hit and not any(names_section(n) and right_act(n) for n in notes) else None
        if hit:
            return R._cite(rng, [hit[1]], [0]).replace("[1]", f"[{hit[0] + 1}]"), "answer"
        if not any(names_section(n) and right_act(n) for n in notes):
            return rng.choice(R.REFUSALS), "negative_missed"
        return None
    topic = squash(item["topic"])
    key = {w for w in re.findall(r"[a-z]{4,}", item["topic"].lower())}
    if item.get("distractor"):
        return (rng.choice(R.REFUSALS), "negative_swapped") if not any(key <= {w for w in re.findall(r"[a-z]{4,}", n.lower())} for n in notes) else None
    hit = best_sentence(notes, lambda s, note: bool(key) and key <= {w for w in re.findall(r"[a-z]{4,}", s.lower())})
    if hit:
        return R._cite(rng, [hit[1]], [0]).replace("[1]", f"[{hit[0] + 1}]"), "answer"
    if not any(topic in squash(n) for n in notes):
        return rng.choice(R.REFUSALS), "negative_missed"
    return None


def build(out_dir, tok, rag, kb, limit, seed):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; refusing to overwrite.")
    rng = random.Random(seed)
    items = mine_judgments(kb, rng) + mine_reference(kb, rng, 2500) + mine_general(rng, 150)
    rng.shuffle(items)
    if limit:
        items = items[:limit]
    print(f"{len(items)} questions mined: {dict(Counter(i['kind'] for i in items))}", flush=True)

    by_act = defaultdict(list)
    for it in items:
        by_act[it.get("act") or it["kind"]].append(it)

    def work(item):
        try:
            local = random.Random(f"{seed}-{item['question']}")                 # the same question always gets the same choice
            item = dict(item)
            roll, inject = local.random(), None
            if item.get("gold") and roll < INJECT_SHARE:
                inject = (item["gold"], item["gold_title"])
            elif item.get("gold") and roll < INJECT_SHARE + DISTRACTOR_SHARE:
                others = [o for o in by_act[item.get("act") or item["kind"]] if o["sid"] != item["sid"] and o.get("gold")]
                if others:
                    other = local.choice(others)
                    item["distractor"] = True
                    inject = (other["gold"], other["gold_title"])
            return item, retrieve(rag, tok, item["question"], inject, local)
        except Exception as exc:  # noqa: BLE001 - one failed search must not stop the build
            return item, exc
    docs = {"train": [], "val": []}
    stats = Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, (item, got) in enumerate(pool.map(work, items), 1):
            if n % 500 == 0:
                print(f"  {n}/{len(items)} retrieved, kept {sum(len(v) for v in docs.values())}", flush=True)
            if isinstance(got, Exception) or got is None:
                stats["no_notes" if got is None else "search_error"] += 1
                continue
            lab = label(item, got, rng)
            if lab is None:
                stats["dropped_unclear"] += 1
                continue
            answer, kind = lab
            mode = R.pick(rng, MODE_MIX)
            split = "val" if int(hashlib.sha1(item["sid"].encode()).hexdigest(), 16) % 100 < VAL_PERCENT else "train"
            ex = {"type": kind, "mode": mode, "topic": item["kind"], "question": item["question"], "answer": answer, "body": got[0], "notes": got[1]}
            enc = tok.encode(R.render(mode, got[0], answer), add_bos=True, add_eos=True)
            if len(enc) > 512:
                stats["too_long"] += 1
                continue
            docs[split].append((ex, enc))
            stats[f"{split}_{kind}"] += 1
    out_dir.mkdir(parents=True)
    meta = {"tokenizer_sha256": tok.sha256, "vocab_size": tok.vocab_size, "format": "graphrag_v1", "seed": seed,
            "format_description": "<BOS> + mode template around the LIVE stored-records prompt (real retrieval) + ' ' + answer + <EOS>",
            "counts": dict(stats)}
    for split, rows in docs.items():
        arr = np.asarray([t for _, enc in rows for t in enc], dtype=np.uint16)
        if arr.size and int(arr.max()) >= tok.vocab_size:
            raise SystemExit("Token id outside the vocabulary.")
        arr.tofile(out_dir / f"{split}.bin")
        meta[split] = {"documents": len(rows), "tokens": int(arr.size), "unk_tokens": int((arr == tok.unk_id).sum()),
                       "by_type": dict(Counter(e["type"] for e, _ in rows))}
        (out_dir / f"{split}_examples.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e, _ in rows), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokenizer-file", type=Path, default=BACKEND_DIR / "data" / "v2" / "tokenizer" / "tokenizer.json")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (a quick trial)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR.parent / ".env")
    from knowledge.kb import KnowledgeBase
    from knowledge.vectorstore import RagStore
    kb, rag = KnowledgeBase.from_env(), RagStore.from_env()
    meta = build(a.out_dir, BPETokenizer(a.tokenizer_file), rag, kb, a.limit, a.seed)
    kb.close()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
