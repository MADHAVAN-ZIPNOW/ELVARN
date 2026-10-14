"""Text helpers shared by the retrieval pipeline (graphrag.py): content words, word overlap, best-sentence snippets and
document-type labels. The retrieval itself is in graphrag.py (Chroma vector search + Neo4j graph traversal)."""
import re

STOPWORDS = frozenset("""a an the and or of to in on at by for from with about as is are was were be been being do does did what who whom whose which when
where why how this that these those it its into than then there their them they he she his her him i me my we our you your tell
give list show me please say said any all some can could would should will shall may might has have had not no yes also over under
between during after before against case cases""".split())
SNIPPET_CHARS = 300
LABELS = {"judgment": "Supreme Court judgment", "order": "High Court order", "press_release": "press release", "reference": "reference (Wikipedia)",
          "court_record": "court record", "commission_report": "commission report", "audit_report": "audit report",
          "assembly": "Assembly record", "news": "news report", "fir": "FIR", "chargesheet": "chargesheet",
          "affidavit": "affidavit", "rti_response": "RTI reply"}


def terms(text):
    """Distinct content words (lower case, 3+ letters or a number), in order, without stop words."""
    seen, out = set(), []
    for w in re.findall(r"[A-Za-z0-9]+", text.lower()):
        if (len(w) >= 3 or w.isdigit()) and w not in STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def stem(word):
    return word[:5]


def overlap(words, text):
    """How many of the question's content words (by 5-letter stem) occur in `text`."""
    have = {stem(w) for w in re.findall(r"[A-Za-z0-9]+", text.lower())}
    return sum(1 for w in words if stem(w) in have)


def best_snippet(text, words, max_chars=SNIPPET_CHARS):
    """The most relevant sentence (plus the next one if it fits) of a passage."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]
    if not sentences:
        return text.strip()[:max_chars]
    best = max(range(len(sentences)), key=lambda i: (overlap(words, sentences[i]), -i))
    snippet = sentences[best]
    nxt = sentences[best + 1] if best + 1 < len(sentences) else ""
    if nxt and overlap(words, nxt) > 0 and len(snippet) + 1 + len(nxt) <= max_chars:     # a neighbour only if it is on topic too
        snippet += " " + nxt
    return snippet[:max_chars].rsplit(" ", 1)[0] if len(snippet) > max_chars else snippet
