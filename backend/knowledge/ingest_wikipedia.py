"""Load Wikipedia articles on corruption cases into the knowledge base as cited reference passages.

Wikipedia is a lead list, not proof: sources are stored as doc_type 'reference' (CC BY-SA 4.0, with the article URL),
and no facts (case parties, statuses) are created here. Those come from court / agency documents, or from reviewed extraction.

Usage (project root):
    python -m backend.knowledge.ingest_wikipedia --limit 5      # quick test (needs NEO4J_URI / NEO4J_PASSWORD)
    python -m backend.knowledge.ingest_wikipedia                # all seed categories
"""
import argparse
import re
from pathlib import Path
from urllib.parse import quote

from backend.data_tools import collect_wikipedia as wiki
from backend.knowledge.kb import KnowledgeBase

# Tamil Nadu corruption only. An article is kept only if it mentions Tamil Nadu AND corruption terms (see is_relevant).
SEED_CATEGORIES = [
    "Corruption in Tamil Nadu", "Scandals in Tamil Nadu", "Political scandals of Tamil Nadu",
    "Politicians from Tamil Nadu convicted of crimes", "Directorate of Vigilance and Anti-Corruption (Tamil Nadu)",
    "Tamil Nadu politicians", "Chief ministers of Tamil Nadu", "Tamil Nadu ministers", "Political parties in Tamil Nadu",
    "Government of Tamil Nadu", "Madras High Court", "Dravidian movement", "Law enforcement in Tamil Nadu",
]
SEED_TITLES = ["2G spectrum scam", "Sarkaria Commission", "Jayalalithaa", "Senthil Balaji", "Disproportionate assets"]
TN_TERMS = re.compile(r"Tamil Nadu|Madras State|Madras Presidency|Chennai|Madras High Court|\bDMK\b|AIADMK|\bDVAC\b", re.I)
CORRUPTION_TERMS = re.compile(r"corrupt|bribe|scam|disproportionate|embezzl|vigilance|money.?laundering|graft|convicted|acquitted|chargesheet|charge.?sheet|\bFIR\b", re.I)
CHUNK_CHARS = 900
LICENCE = "CC BY-SA 4.0"


def is_relevant(text):
    """Tamil Nadu corruption only: the article must mention both Tamil Nadu and corruption terms."""
    return bool(TN_TERMS.search(text) and CORRUPTION_TERMS.search(text))


def chunk(text, size=CHUNK_CHARS):
    """Group paragraphs into passages of about `size` characters; headings stay with the text that follows."""
    out, cur = [], ""
    for para in re.split(r"\n\s*\n|\n(?==)", text):
        para = para.strip()
        if not para or re.fullmatch(r"=+\s*(" + "|".join(wiki.CUT_SECTIONS) + r")\s*=+", para, re.I):
            continue
        if cur and len(cur) + len(para) > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n{para}".strip()
    if cur:
        out.append(cur)
    return out


def ingest_article(kb, title, text, require_relevant=True, rag=None, cache=None):
    """Store one article in both stores; returns number of passages added (0 if already ingested, too short, or - when
    require_relevant - not about Tamil Nadu corruption)."""
    if len(text) < 200 or (require_relevant and not is_relevant(text)):
        return 0
    from backend.knowledge.store import store_document
    parts = chunk(text)
    passages = [{"locator": f"chunk {i + 1}", "text": part} for i, part in enumerate(parts)]
    status = store_document(kb, rag, title=f"{title} (Wikipedia)", doc_type="reference", text=text, passages=passages,
                           url="https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")), publisher="Wikipedia",
                           licence=LICENCE, cache=cache)
    return len(parts) if status == "loaded" else 0


def run(limit=0, per_category=300, kb=None):
    """Collect Tamil Nadu corruption articles into the graph. Returns {"articles": n, "passages": n}."""
    cache = wiki.BACKEND_DIR / "data" / "cache_wikipedia" / "corruption"
    cache.mkdir(parents=True, exist_ok=True)
    own = kb is None
    kb = kb or KnowledgeBase()
    from backend.knowledge.vectorstore import RagStore
    from backend.usermemory.store import Memory
    rag, mem = RagStore.from_env(), Memory.from_env()            # the vector store and the cache, when configured
    seen, articles, passages = set(), 0, 0

    def work():
        yield "seed titles", SEED_TITLES
        for cat in SEED_CATEGORIES:
            yield cat, wiki.category_titles(cat, per_category, depth=1)

    try:
        for label, titles in work():
            for title in titles:
                if title in seen or wiki.SKIP_TITLE.search(title):
                    continue
                seen.add(title)
                rec = wiki.fetch_text(title, cache)
                if not rec:
                    continue
                n = ingest_article(kb, rec["title"], rec["text"], rag=rag)
                if n:
                    articles, passages = articles + 1, passages + n
                    print(f"{label} | {rec['title']}: {n} passages", flush=True)
                if limit and articles >= limit:
                    break
            if limit and articles >= limit:
                break
    finally:
        if own:
            kb.close()
    if mem is not None and articles:
        mem.bump_version()                                       # once at the end: cached retrievals are cleared
    print(f"done: {articles} articles, {passages} passages")
    return {"articles": articles, "passages": passages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after this many articles (0 = all)")
    ap.add_argument("--per-category", type=int, default=300)
    args = ap.parse_args()
    run(args.limit, args.per_category)


if __name__ == "__main__":
    main()
