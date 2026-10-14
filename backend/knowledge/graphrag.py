"""GraphRAG: the retrieval pipeline.

    question
       |  query understanding  (people / cases mentioned, follow-up resolved from the user's memory, intent, key terms)
       v
    +---------------------------+    +----------------------------------+
    | Chroma vector search      |    | Neo4j graph traversal            |      run in parallel
    | relevant text passages    |    | entities + relationships + facts |
    +---------------------------+    +----------------------------------+
       |
       v   merge and rank (verified facts and passages in ONE list, each with its citation)
    Context  ->  the model writes the final answer from the top notes; every note is cited

Graph facts carry the assertion level (alleged / reported / court_finding) and are ranked above passages when the question
names a known person or case; passages are accepted only if they are really close to the question (similarity AND shared words).
Either branch can fail without breaking the other: the answer is then built from whatever succeeded.
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from backend.knowledge import rag
from backend.knowledge.kb import CORRUPTION_WORDS

log = logging.getLogger("elvarn.graphrag")

LEVEL_WORDS = {"alleged": "alleged", "reported": "reported", "court_finding": "court finding"}
LEVEL_BONUS = {"court_finding": 0.15, "reported": 0.08, "alleged": 0.0}
FOLLOWUP = re.compile(r"\b(he|she|his|her|him|they|their|them|that case|this case|the case|it|its)\b", re.I)
INTENTS = (("timeline", re.compile(r"timeline|what happened|stages?|progress|status|verdict|judg(?:e)?ment|sentence|bail|arrest|charge", re.I)),
           ("cases", re.compile(r"\bcases?\b|corruption|accused|involved|list|scams?", re.I)))
# Thresholds measured on the real MiniLM embeddings + the real collection (see tests and the 2026-10-14 calibration):
# in-domain questions had >= 57% of their words covered by the best passage; questions the stores cannot answer topped out at 33-40%.
MIN_SIM, VERY_CLOSE_SIM = 0.30, 0.72      # a passage below MIN_SIM is never used; one at VERY_CLOSE_SIM or above needs no word overlap
MIN_COVERAGE = 0.55                       # otherwise it must contain at least this share of the question's content words (and 2+ words)
MAX_LINE = 280


@dataclass
class Understanding:
    question: str
    resolved: str
    entities: list
    intent: str
    terms: list
    followup: bool = False


@dataclass
class Context:
    items: list = field(default_factory=list)          # ranked; each: kind, text, score, source{...}, level
    understanding: Understanding = None
    notes: list = field(default_factory=list)          # which branch failed, cache hit ...

    def facts(self):
        return [i for i in self.items if i["kind"] == "fact"]

    def to_dict(self):
        u = self.understanding
        return {"items": self.items, "notes": self.notes,
                "u": {"question": u.question, "resolved": u.resolved, "entities": u.entities, "intent": u.intent, "terms": u.terms, "followup": u.followup}}

    @classmethod
    def from_dict(cls, d):
        return cls(items=d["items"], notes=d.get("notes", []) + ["cache"], understanding=Understanding(**d["u"]))


# ---- 1. query understanding ------------------------------------------------------------------------
def understand(kb, question, memory_ctx=None):
    ents = [e for e in kb.find_entities(question) if e["type"] in ("Person", "Case")]
    resolved, followup = question, False
    if not ents and memory_ctx and memory_ctx.get("entities") and FOLLOWUP.search(question):
        last = memory_ctx["entities"][0]                  # "what about his bail?" -> the person / case just discussed
        ents, followup = [last], True
        resolved = f"{question} ({last['name']})"
    intent = next((name for name, rx in INTENTS if rx.search(question)), "general")
    return Understanding(question, resolved, ents, intent, rag.terms(resolved), followup)


# ---- 2a. graph traversal -----------------------------------------------------------------------------
def _source(row):
    return {"title": row["source_title"], "url": row["url"] or "", "label": rag.LABELS.get(row["doc_type"], row["doc_type"]),
            "date": row.get("published_date"), "licence": row.get("licence"), "source_id": row["source_id"]}


def _fact(text, score, row, level):
    return {"kind": "fact", "text": text[:MAX_LINE], "score": round(score, 4), "level": level, "source": _source(row)}


def _status(row):
    word = (row.get("status") or row.get("role") or "").replace("_", " ")
    when = f" on {row['status_date']}" if row.get("status_date") else ""
    return f"{word}{when}"


def graph_items(kb, u):
    """Facts and relationships around the entities the question is about (one or two hops)."""
    out = []
    for ent in u.entities:
        if ent["type"] == "Person":
            rows = kb.cases_of_person(ent["id"])
            for r in rows:
                num = f" ({r['case_number']})" if r.get("case_number") else ""
                text = f"{ent['name']} is {r['role'].replace('_', ' ')} in {r['case_name']}{num}: {_status(r)} - {LEVEL_WORDS[r['level']]}."
                bonus = 0.1 if u.intent == "cases" else 0.0
                out.append(_fact(text, 0.80 + LEVEL_BONUS[r["level"]] + bonus, r, r["level"]))
            case_ids = list(dict.fromkeys(f"Case:{' '.join(r['case_name'].lower().split())}" for r in rows))[:2]
        else:
            case_ids = [ent["id"]]
        for cid in case_ids:                               # second hop: the case's own timeline, parties and details
            info, name = kb.case_info(cid), cid.split(":", 1)[1]
            for ev in kb.case_timeline(cid)[:5]:
                when = ev["event_date"] or "date unknown"
                text = f"{info.get('name', name)}: on {when}, {ev['description']} ({ev['stage'].replace('_', ' ')}) - {LEVEL_WORDS[ev['level']]}."
                bonus = 0.15 if u.intent == "timeline" else 0.0
                out.append(_fact(text, 0.62 + LEVEL_BONUS[ev["level"]] + bonus, ev, ev["level"]))
            for p in kb.case_parties(cid)[:6]:
                if ent["type"] == "Person" and p["person_id"] == ent["id"]:
                    continue
                text = f"In {info.get('name', name)}, {p['person']} is {p['role'].replace('_', ' ')}: {_status(p)} - {LEVEL_WORDS[p['level']]}."
                out.append(_fact(text, 0.55 + LEVEL_BONUS[p["level"]], p, p["level"]))
        for cl in kb.entity_claims(ent["id"])[:4]:
            obj = cl["object"] or cl["value"]
            out.append(_fact(f"{ent['name']} {cl['predicate'].replace('_', ' ')} {obj}.", 0.58 + LEVEL_BONUS[cl["level"]], cl, cl["level"]))
    return out


# ---- 2b. vector search -------------------------------------------------------------------------------
def vector_items(store, u, k=10):
    """Passages from Chroma that are really about the question: close in meaning AND sharing its words."""
    if not u.terms:
        return []
    need = min(2, len(u.terms))
    out = []
    for hit in store.query(u.resolved, k=k):
        sim = 1.0 - hit["distance"]
        meta = hit["meta"]
        snippet = rag.best_snippet(hit["text"], u.terms)
        shared = rag.overlap(u.terms, snippet + " " + meta.get("title", ""))
        if sim < MIN_SIM or (sim < VERY_CLOSE_SIM and (shared < need or shared / len(u.terms) < MIN_COVERAGE)):
            continue
        score = 0.9 * sim + 0.1 * min(1.0, shared / len(u.terms))
        out.append({"kind": "chunk", "text": snippet, "score": round(score, 4), "level": None,
                    "source": {"title": meta.get("title", ""), "url": meta.get("url", ""), "label": rag.LABELS.get(meta.get("doc_type"), meta.get("doc_type")),
                               "date": meta.get("published_date"), "licence": meta.get("licence"), "source_id": meta.get("source_id")}})
    return out


# ---- 3. merge and rank ---------------------------------------------------------------------------------
def merge(facts, chunks, max_items=5, per_source=2, facts_per_source=3):
    """One ranked list. The best fact and the best passage always come first (so the model sees both kinds), then by score.
    Duplicate texts are dropped and no single document may supply more than `facts_per_source` facts or `per_source` passages
    (counted separately: one judgment can give several facts and still contribute its own text)."""
    kept, seen, per = [], set(), {}
    for item in sorted(facts + chunks, key=lambda i: -i["score"]):          # drop duplicates and over-represented documents first
        key, slot = re.sub(r"\W+", " ", item["text"].lower()).strip(), (item["source"].get("source_id"), item["kind"])
        if key in seen or per.get(slot, 0) >= (facts_per_source if item["kind"] == "fact" else per_source):
            continue
        seen.add(key)
        per[slot] = per.get(slot, 0) + 1
        kept.append(item)
    first = [next((i for i in kept if i["kind"] == kind), None) for kind in ("fact", "chunk")]
    first = [i for i in first if i is not None]
    return (first + [i for i in kept if not any(i is f for f in first)])[:max_items]


# ---- domain check ----------------------------------------------------------------------------------------
OFFICIAL_LABELS = frozenset(rag.LABELS[t] for t in ("judgment", "order", "press_release", "court_record", "commission_report", "audit_report",
                                                      "assembly", "fir", "chargesheet", "affidavit", "rti_response"))
OFF_DOMAIN_MIN = 0.65      # measured on the real embeddings: off-topic questions score 0.00-0.53, in-domain ones 0.54-0.81


def in_domain(question, items):
    """Is this a Tamil Nadu corruption question? Yes if it has corruption wording, matches graph facts, or hits an official document.
    Everything else (general knowledge, news, small talk) must clear OFF_DOMAIN_MIN to use stored passages at all; otherwise the
    chat falls through to live web search or the plain model instead of being fed unrelated notes."""
    if CORRUPTION_WORDS.search(question):
        return True
    if any(i["kind"] == "fact" for i in items):
        return True
    return any(i["source"].get("label") in OFFICIAL_LABELS and i["score"] >= 0.5 for i in items)


# Bump this whenever the ranking / filtering rules above change: cached contexts built with the old rules must not be served any more.
LOGIC_VERSION = 3


def cache_key(question):
    return f"{question}|logic{LOGIC_VERSION}"


# ---- the pipeline --------------------------------------------------------------------------------------
def build_context(kb, store, question, memory=None, session=None, max_items=5):
    """Run the whole pipeline. `store` (Chroma), `memory` (Redis) and `session` are optional; the graph branch needs `kb`."""
    mem_ctx = memory.recall(session) if memory is not None and session else None
    u = understand(kb, question, mem_ctx) if kb is not None else Understanding(question, question, [], "general", rag.terms(question))
    ent_ids = [e["id"] for e in u.entities]
    if memory is not None:
        cached = memory.get_context(cache_key(u.resolved), ent_ids)
        if cached:
            return Context.from_dict(cached)
    notes = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        graph_f = pool.submit(graph_items, kb, u) if kb is not None and u.entities else None
        vec_f = pool.submit(vector_items, store, u) if store is not None else None
        facts, chunks = [], []
        for name, fut in (("graph", graph_f), ("vector", vec_f)):
            if fut is None:
                continue
            try:
                (facts if name == "graph" else chunks).extend(fut.result())
            except Exception as exc:  # noqa: BLE001 - one branch failing must not lose the other
                log.warning("%s branch failed: %s", name, exc.__class__.__name__)
                notes.append(f"{name} search unavailable")
    items = merge(facts, chunks, max_items)
    if not in_domain(u.question, items):                   # outside the corruption domain only strong hits are worth showing the model
        items = [i for i in items if i["score"] >= OFF_DOMAIN_MIN]
    ctx = Context(items=items, understanding=u, notes=notes)
    if memory is not None and not notes:                   # never cache a partial result
        memory.put_context(cache_key(u.resolved), ent_ids, ctx.to_dict())
    return ctx
