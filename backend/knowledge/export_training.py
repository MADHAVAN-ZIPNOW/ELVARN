"""Export the knowledge base as training text for the model, in the block format the project's pipeline already reads.

Writes backend/data/raw_v2/kb_corpus.txt (the name matters: build_rag_dataset.py reads kb_*.txt next to wikipedia_*.txt, and
corpus.load_documents picks the file up for language-model training too). Two kinds of block, nothing else:

  <DOMAIN_KNOWLEDGE>  Topic / Text / Source      one block per stored passage of an OFFICIAL document (judgments, orders, press
                                                 releases, court records, reports ...), plus the passages of every Wikipedia article
                                                 whose topic the model has NOT seen yet (compared with the topics in the existing
                                                 training files). Articles, notes and documents the model was already trained on are
                                                 left out: repeating identical text adds nothing and invites memorising.
  <DOMAIN_KNOWLEDGE>  Topic / Question / Answer / Source    one block per person or case, written from APPROVED graph facts only,
                                                 always with the assertion level (alleged / reported / court finding) in the wording.

Why only this: the model must learn to answer FROM notes, not to store allegations about real people as facts. Passage blocks keep
their source link; fact blocks never state anything without its level. Unreadable (garbled) text is dropped. The file is rewritten
atomically and deterministically (same graph -> same file), and a small state file records how many passages were exported so the
retraining step can tell whether enough NEW material has arrived.

    python -m backend.knowledge.export_training [--out backend/data/raw_v2/kb_corpus.txt]
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.knowledge.graphrag import LEVEL_WORDS
from backend.knowledge.kb import KnowledgeBase
from backend.pipeline.process import looks_garbled

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "raw_v2" / "kb_corpus.txt"
STATE = DATA / "kb_state" / "training.json"
OFFICIAL = ("judgment", "order", "press_release", "court_record", "commission_report", "audit_report", "affidavit", "assembly", "fir", "chargesheet", "rti_response")
MIN_CHARS, MAX_CHARS = 120, 1800


def clean(text):
    return " ".join(text.replace("<", "(").replace(">", ")").split())


GROUP_CHARS = 1200


def block(topic, text, source):
    return f"<DOMAIN_KNOWLEDGE>\nTopic: {topic}\nText: {text}\nSource: {source or 'unknown'}\n</DOMAIN_KNOWLEDGE>"


def passage_blocks(kb):
    """One block per readable passage of an official document. Short press-release lines (one sentence each) are grouped by
    publisher and month into passages of up to GROUP_CHARS, because the RAG builder needs several sentences per passage."""
    seen, groups = set(), {}
    rows = kb._read("MATCH (p:Passage)-[:FROM]->(s:Source) WHERE s.doc_type IN $types "
                    "RETURN s.title AS title, s.url AS url, s.published_date AS date, s.doc_type AS doc_type, s.publisher AS publisher, "
                    "p.position AS pos, p.text AS text ORDER BY s.published_date, s.title, p.position", types=list(OFFICIAL))
    for r in rows:
        text = clean(r["text"])
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if len(text) < 20 or len(text) > MAX_CHARS or key in seen or looks_garbled(text):
            continue
        seen.add(key)
        if r["doc_type"] == "press_release" and len(text) < MIN_CHARS * 3:               # short one-line releases: group them
            gkey = (r["publisher"] or "press releases", (r["date"] or "undated")[:7])
            groups.setdefault(gkey, []).append((text, r["url"]))
            continue
        if len(text) >= MIN_CHARS:
            yield block(clean(r["title"])[:160], text, r["url"])
    for (publisher, month), items in sorted(groups.items()):
        chunk_text, urls = "", []
        for text, url in items + [("", None)]:
            if chunk_text and (not text or len(chunk_text) + len(text) + 1 > GROUP_CHARS):
                if len(chunk_text) >= MIN_CHARS:
                    yield block(f"{clean(publisher)[:100]} press releases, {month}", chunk_text.strip(), urls[0])
                chunk_text, urls = "", []
            if text:
                chunk_text, urls = f"{chunk_text} {text}", urls + [url]


def norm_topic(title):
    return re.sub(r" \(wikipedia\)$", "", title.strip().lower())


def known_topics(raw_dir):
    """Topics the existing training files already cover (Wikipedia articles and the project's notes)."""
    topics = set()
    for name in ("wikipedia_domain.txt", "wikipedia_general.txt", "domain_knowledge.txt"):
        path = Path(raw_dir) / name
        if path.is_file():
            topics.update(norm_topic(m.group(1)) for m in re.finditer(r"^Topic: (.+)$", path.read_text(encoding="utf-8", errors="replace"), re.M))
    return topics


def reference_blocks(kb, known):
    """Passages of Wikipedia articles the model has not been trained on yet (the project's own notes/documents are never re-exported)."""
    seen = set()
    rows = kb._read("MATCH (p:Passage)-[:FROM]->(s:Source {doc_type:'reference', publisher:'Wikipedia'}) "
                    "RETURN s.title AS title, s.url AS url, p.text AS text ORDER BY s.title, p.position")
    for r in rows:
        topic = norm_topic(r["title"])
        text = clean(r["text"])
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if topic in known or not (MIN_CHARS <= len(text) <= MAX_CHARS) or key in seen or looks_garbled(text):
            continue
        seen.add(key)
        yield block(clean(r["title"].rsplit(" (Wikipedia)", 1)[0])[:160], text, r["url"])


def fact_blocks(kb):
    """Question/Answer blocks from approved facts. Each answer lists what is stored with its level, never a bare claim."""
    for person in kb._read("MATCH (e:Person) RETURN e.key AS id, e.name AS name ORDER BY e.name"):
        rows = kb.cases_of_person(person["id"])
        if not rows:
            continue
        lines = []
        for r in rows[:8]:
            when = f" on {r['status_date']}" if r["status_date"] else ""
            lines.append(f"{r['case_name']}: {r['role'].replace('_', ' ')}, {(r['status'] or r['role']).replace('_', ' ')}{when} ({LEVEL_WORDS[r['level']]}, source: {r['source_title']})")
        yield (f"<DOMAIN_KNOWLEDGE>\nTopic: {clean(person['name'])}\nQuestion: Which corruption cases is {clean(person['name'])} linked to in the records?\n"
               f"Answer: According to the records: {'; '.join(clean(x) for x in lines)}. Allegations are not findings of guilt.\nSource: {rows[0]['url'] or 'unknown'}\n</DOMAIN_KNOWLEDGE>")
    for case in kb._read("MATCH (c:Case) RETURN c.key AS id, c.name AS name ORDER BY c.name"):
        events = kb.case_timeline(case["id"])
        if not events:
            continue
        lines = [f"{e['event_date'] or 'date unknown'}: {e['description']} ({e['stage'].replace('_', ' ')}; {LEVEL_WORDS[e['level']]})" for e in events[:8]]
        yield (f"<DOMAIN_KNOWLEDGE>\nTopic: {clean(case['name'])}\nQuestion: What happened in the {clean(case['name'])} case?\n"
               f"Answer: According to the records: {'; '.join(clean(x) for x in lines)}.\nSource: {events[0]['url'] or 'unknown'}\n</DOMAIN_KNOWLEDGE>")


def export(kb, out=OUT, state=STATE, known=None):
    """`known` = topics to treat as already trained on (default: read from the existing training files next to `out`)."""
    out, state = Path(out), Path(state)
    out.parent.mkdir(parents=True, exist_ok=True)
    known = known_topics(out.parent) if known is None else known
    blocks = list(passage_blocks(kb))
    refs = list(reference_blocks(kb, known))
    facts = list(fact_blocks(kb))
    body = "\n\n".join(blocks + refs + facts) + "\n"
    tmp = out.with_suffix(".part")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(out)                                         # a half-written export is never visible
    info = {"passage_blocks": len(blocks), "reference_blocks": len(refs), "fact_blocks": len(facts), "chars": len(body), "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    old = json.loads(state.read_text(encoding="utf-8")) if state.is_file() else {}
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({**old, "last_export": info}, indent=2), encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    kb = KnowledgeBase()
    try:
        print(json.dumps(export(kb, a.out), indent=2))
    finally:
        kb.close()


if __name__ == "__main__":
    main()
