"""Load raw files as documents and split them into train / validation.

The split is made on whole documents (never on overlapping windows), then
any training document that is a near-duplicate of a validation document is
removed, so validation loss is not inflated by leakage.

Documents:
  * files that contain <TAG> ... </TAG> blocks (domain_knowledge.txt) give one
    document per block, with the closing tag removed (EOS replaces it);
  * any other .txt/.md file is cut into paragraph groups of about
    MAX_DOC_CHARS characters.
"""
import hashlib
import re
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BACKEND_DIR / "data" / "raw"
SUPPORTED_EXTENSIONS = {".txt", ".md"}

CATEGORY_TAGS = [
    "<DOMAIN_KNOWLEDGE>",
    "<GENERAL_ENGLISH>",
    "<CASUAL_CONVERSATION>",
    "<HUMOUR>",
]

MAX_DOC_CHARS = 1200
VAL_FRACTION = 0.05          # about 1 document in 20 goes to validation
NEAR_DUP_JACCARD = 0.5
SHINGLE_WORDS = 5

_BLOCK_RE = re.compile(r"<(\w+)>\n(.*?)\n</\1>", re.S)


def _normalise(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def _bucket(text):
    """Stable 0..9999 number from the text, so the split never depends on order."""
    digest = hashlib.sha1(_normalise(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000


def _shingles(text):
    words = re.findall(r"\w+", text.lower())
    if len(words) < SHINGLE_WORDS:
        return {" ".join(words)} if words else set()
    return {
        " ".join(words[i:i + SHINGLE_WORDS])
        for i in range(len(words) - SHINGLE_WORDS + 1)
    }


def _split_plain_text(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    docs, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) > MAX_DOC_CHARS:
            docs.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        docs.append(current)
    return docs


def raw_files(raw_dir=RAW_DATA_DIR):
    return sorted(
        p for p in Path(raw_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_documents(raw_dir=RAW_DATA_DIR):
    """Return a list of dicts: {source, category, text}."""
    documents = []
    for path in raw_files(raw_dir):
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = _BLOCK_RE.findall(text)
        if blocks:
            for tag, body in blocks:
                documents.append({
                    "source": path.name,
                    "category": tag,
                    "text": f"<{tag}>\n{body.strip()}",
                })
        else:
            for body in _split_plain_text(text):
                documents.append({
                    "source": path.name,
                    "category": "PLAIN",
                    "text": body,
                })
    return documents


_TOPIC_RE = re.compile(r"^Topic: (.+)$", re.M)


def group_key(doc):
    """All passages of one Wikipedia article belong together (same side of the split).

    Only blocks that carry a Wikipedia Source line are grouped. Every other document keeps the original rule
    (split by the hash of its own text), so adding new data never moves an already-seen document from train
    to validation: a model trained on the old data keeps an honest validation set.
    """
    if "Source: https://en.wikipedia.org/" not in doc["text"]:
        return None
    m = _TOPIC_RE.search(doc["text"])
    return ("topic:" + _normalise(m.group(1))) if m else None


def split_documents(documents, group_by_topic=False):
    """Deduplicate, split by document, then remove near-duplicate leakage."""
    seen, unique, exact_dups = set(), [], 0
    for doc in documents:
        key = _normalise(doc["text"])
        if key in seen:
            exact_dups += 1
            continue
        seen.add(key)
        unique.append(doc)

    threshold = int(VAL_FRACTION * 10_000)
    train, val = [], []
    for doc in unique:
        key = (group_key(doc) if group_by_topic else None) or doc["text"]      # same topic -> same side of the split
        (val if _bucket(key) < threshold else train).append(doc)

    # Near-duplicate filter: drop training docs too similar to a validation doc.
    index = {}
    val_shingles = []
    for i, doc in enumerate(val):
        shingles = _shingles(doc["text"])
        val_shingles.append(shingles)
        for s in shingles:
            index.setdefault(s, set()).add(i)

    kept_train, near_dups = [], 0
    for doc in train:
        shingles = _shingles(doc["text"])
        candidates = set()
        for s in shingles:
            candidates |= index.get(s, set())
        leaked = False
        for i in candidates:
            union = len(shingles | val_shingles[i])
            if union and len(shingles & val_shingles[i]) / union >= NEAR_DUP_JACCARD:
                leaked = True
                break
        if leaked:
            near_dups += 1
        else:
            kept_train.append(doc)

    stats = {
        "documents_total": len(documents),
        "exact_duplicates_removed": exact_dups,
        "near_duplicates_removed_from_train": near_dups,
        "train_documents": len(kept_train),
        "val_documents": len(val),
    }
    return kept_train, val, stats
