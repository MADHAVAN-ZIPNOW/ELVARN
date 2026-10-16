"""Clean text corpus: raw HTML/text -> clean text -> quality gate -> passages (300 to 1,000 estimated tokens) -> Parquet rows.

Everything here is pure standard-library Python (3.8 compatible): it runs inside the Spark image, whose Python workers have
no bs4, lxml, langdetect, pandas or pyarrow. The Parquet files themselves are written by Spark's own writer (spark_job.py)
and updated by pyarrow in the nightly check_stores; this module only defines the columns and the rows.

Rules (docs/PIPELINE_CONTRACT.md section 4.1): a document under 1,000 characters, text that is under 70 percent letters, a
page that is mostly links or menus, excessive repeated lines, poor extraction, garbled legacy-font Tamil and exact
duplicate passages never reach the corpus. A passage id is kb.passage_id(sha256 of the clean document text, position), the
same id Neo4j and Chroma use, so every load is idempotent.
"""
import hashlib
import html as _html
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from backend.knowledge.kb import CORRUPTION_WORDS, passage_id

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
MIN_DOC_CHARS = 1000
MIN_LETTER_RATIO = 0.70
MIN_TOKENS, MAX_TOKENS = 300, 1000
OVERLAP = 0.15                                   # fraction of the previous passage repeated at the start of the next
UNIT_MAX = 400                                   # a sentence longer than this is split (at a comma or a space) before packing

# ---- columns of the Parquet corpus (contract 4.1); "type" is string | int | bool ----------------------------------------
PASSAGE_COLUMNS = [
    ("passage_id", "string"), ("doc_id", "string"), ("source", "string"), ("source_url", "string"), ("title", "string"),
    ("text", "string"), ("language", "string"), ("char_count", "int"), ("token_estimate", "int"), ("sha256", "string"),
    ("created_at", "string"), ("quality_score", "double"), ("embedded", "bool"), ("indexed_neo4j", "bool"),
    ("person_key", "string"), ("person_name", "string"), ("topic", "string"), ("corruption_related", "bool"), ("position", "int"),
]
COLUMN_NAMES = [c for c, _ in PASSAGE_COLUMNS]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_estimate(text_or_len):
    """ceil(char_count / 4): the project-wide token estimate (no tokenizer needed)."""
    n = text_or_len if isinstance(text_or_len, int) else len(text_or_len)
    return int(math.ceil(n / 4.0))


# ---- HTML -> text -----------------------------------------------------------------------------------------------------
SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe", "button", "select",
             "template", "table", "figure", "figcaption", "head", "title", "textarea", "canvas", "audio", "video", "map"}
# class tokens that mark a box that is not article prose (Wikipedia: infobox, navbox, hatnote, toc, edit links, reference lists, citation marks)
SKIP_CLASSES = {"infobox", "navbox", "vertical-navbox", "navbox-inner", "sidebar", "hatnote", "dablink", "rellink", "toc", "toccolours",
                "mw-editsection", "mw-editsection-bracket", "reflist", "references", "mw-references-wrap", "mw-ref", "reference", "noprint",
                "metadata", "ambox", "mbox-small", "catlinks", "printfooter", "breadcrumb", "breadcrumbs", "navigation", "navbar",
                "mw-empty-elt", "thumb", "thumbinner", "gallery", "portal", "authority-control", "sistersitebox", "navigation-not-searchable",
                "mw-jump-link", "sr-only", "shortdescription", "cookie", "cookies", "cookie-banner", "cookie-notice", "cookie-consent",
                "cc-window", "consent", "gdpr", "banner", "advertisement", "ad", "share", "social", "newsletter", "pagination"}
SKIP_ROLES = {"navigation", "banner", "contentinfo", "complementary", "note", "search", "dialog", "alert"}
SKIP_ID = re.compile(r"^(?:toc|catlinks|footer|mw-navigation|mw-panel|siteSub|contentSub|jump-to-nav|.*cookie.*|.*consent.*)$", re.I)
VOID = {"br", "hr", "img", "meta", "link", "input", "area", "base", "col", "embed", "source", "track", "wbr"}
BLOCK = {"p", "div", "li", "ul", "ol", "dl", "dd", "dt", "blockquote", "section", "article", "main", "pre", "tr", "br", "hr",
         "h1", "h2", "h3", "h4", "h5", "h6", "details", "summary", "body", "center"}
CUT_SECTIONS = {"references", "notes", "citations", "external links", "see also", "further reading", "bibliography", "sources",
                "footnotes", "notes and references", "references and notes", "gallery",
                "மேற்கோள்கள்", "வெளி இணைப்புகள்", "வெளியிணைப்புகள்", "இவற்றையும் பார்க்க", "உசாத்துணை", "குறிப்புகள்", "ஆதாரங்கள்",
                "அடிக்குறிப்புகள்", "மேலும் படிக்க", "சான்றுகள்"}


class _Extractor(HTMLParser):
    """Collects (kind, text) blocks of article prose; drops everything SKIP_* names. `link_chars` / `chars` measure how much
    of the kept text is link text (a page that is mostly links is a menu, not an article)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.cur, self.kind = [], [], "p"
        self.skip_tag, self.skip_depth = None, 0
        self.in_link, self.chars, self.link_chars = 0, 0, 0

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self.cur)).strip()
        if text:
            self.blocks.append((self.kind, text))
        self.cur, self.kind = [], "p"

    def _skippable(self, tag, attrs):
        a = dict(attrs)
        classes = set((a.get("class") or "").lower().split())
        return (tag in SKIP_TAGS or bool(classes & SKIP_CLASSES) or (a.get("role") or "").lower() in SKIP_ROLES
                or bool(SKIP_ID.match(a.get("id") or "")) or a.get("hidden") is not None or a.get("aria-hidden") == "true"
                or "display:none" in (a.get("style") or "").replace(" ", "").lower())

    def handle_starttag(self, tag, attrs):
        if self.skip_tag is not None:
            if tag == self.skip_tag:
                self.skip_depth += 1
            return
        if tag not in VOID and self._skippable(tag, attrs):
            self._flush()
            self.skip_tag, self.skip_depth = tag, 1
            return
        if tag in BLOCK:
            self._flush()
            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                self.kind = tag
        if tag == "a":
            self.in_link += 1

    def handle_startendtag(self, tag, attrs):
        if self.skip_tag is None and tag in BLOCK:
            self._flush()

    def handle_endtag(self, tag):
        if self.skip_tag is not None:
            if tag == self.skip_tag:
                self.skip_depth -= 1
                if self.skip_depth == 0:
                    self.skip_tag = None
            return
        if tag == "a" and self.in_link:
            self.in_link -= 1
        if tag in BLOCK:
            self._flush()

    def handle_data(self, data):
        if self.skip_tag is not None:
            return
        self.cur.append(data)
        n = len(data.strip())
        self.chars += n
        if self.in_link:
            self.link_chars += n


def _drop_cut_sections(blocks):
    """Remove 'References', 'External links', 'See also' ... and everything under them (until a heading of the same or higher level),
    then headings that have no text under them."""
    out, cut = [], None
    for kind, text in blocks:
        if kind[0] == "h" and kind[1:].isdigit():
            level = int(kind[1:])
            if cut is not None and level <= cut:
                cut = None
            if cut is None and text.strip().lower().rstrip(":") in CUT_SECTIONS:
                cut = level
                continue
        if cut is None:
            out.append((kind, text))
    final = []
    for i, (kind, text) in enumerate(out):
        is_heading = kind[0] == "h" and kind[1:].isdigit()
        if is_heading and (i + 1 == len(out) or (out[i + 1][0][0] == "h" and out[i + 1][0][1:].isdigit())):
            continue
        final.append((kind, text))
    return final


def decode_html(data):
    """Bytes -> str for HTML (utf-8, falling back to cp1252, which never fails)."""
    if isinstance(data, str):
        return data
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


LD_JSON = re.compile(r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>", re.S | re.I)
ARTICLE_TAG = re.compile(r"<article\b.*?</article>", re.S | re.I)
MIN_STORY_CHARS = 800


def _json_ld_bodies(raw):
    """The `articleBody` strings that news sites put in their JSON-LD block: the story itself, without the menus, ads and related links."""
    out = []
    for m in LD_JSON.finditer(raw):
        try:
            stack = [json.loads(m.group(1))]
        except ValueError:
            continue
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                body = x.get("articleBody")
                if isinstance(body, str) and len(body) >= 600:
                    out.append(body)
                stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)
    return out


def _extract(raw):
    p = _Extractor()
    p.feed(raw)
    p.close()
    p._flush()
    blocks = _drop_cut_sections(p.blocks)
    return "\n\n".join(t for _, t in blocks), (p.link_chars / p.chars if p.chars else 0.0)


def html_to_text_info(html):
    """-> (text, link_ratio). Main article text only, headings and paragraphs separated by a blank line.
    A news page is mostly menus, ads and related links around a short story: when the page carries its story (JSON-LD articleBody, or an <article>
    element) only that is read; every other page is read as a whole (Wikipedia and the like)."""
    raw = decode_html(html)
    bodies = _json_ld_bodies(raw)
    if bodies:
        body = max(bodies, key=len)
        body = re.sub(r"<br\s*/?>|</p>", "\n\n", body, flags=re.I)
        return _html.unescape(re.sub(r"<[^>]+>", " ", body)).strip(), 0.0
    for art in sorted(ARTICLE_TAG.findall(raw), key=len, reverse=True)[:1]:
        text, ratio = _extract(art)
        if len(text) >= MIN_STORY_CHARS and ratio < 0.5:
            return text, ratio
    return _extract(raw)


def html_to_text(html):
    """Main text of an HTML page (str or bytes). See SKIP_TAGS / SKIP_CLASSES for what is dropped."""
    return html_to_text_info(html)[0]


# ---- normalisation --------------------------------------------------------------------------------------------------
CITATION = re.compile(r"\[\s*(?:\d{1,3}|[a-z]|citation needed|edit|note \d+|nb \d+|when\?|who\?|clarification needed|dubious[^\]]{0,30}|"
                      r"better source needed|verification needed|\d+\s*-\s*\d+)\s*\]", re.I)
BOILERPLATE_LINE = re.compile(
    r"^(?:skip to (?:main )?content|jump to (?:navigation|search)|(?:we|this (?:web)?site|our site) uses? cookies.*|"
    r"(?:accept|reject|manage)(?: all)? cookies?|cookie (?:policy|settings|preferences|notice)|privacy policy|"
    r"terms (?:of use|and conditions|of service)|all rights reserved.*|copyright\b.*|©.*|share (?:this|on)\b.*|follow us\b.*|"
    r"subscribe(?: to .*)?|sign (?:in|up)|log ?in|back to top|print(?: this page)?|menu|home|search|retrieved from .*|"
    r"this page was last edited.*|text is available under.*|from wikipedia, the free encyclopedia|edit|contents|hide|"
    r"read more|click here.*|advertisement|related (?:articles|posts)|tags?:.*)$", re.I)


def normalize(text):
    """Unicode NFC, tidy whitespace, citation marks and boilerplate lines removed, lines that repeat 3 times or more dropped."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ").replace("​", "").replace("﻿", "")
    text = CITATION.sub("", text)
    lines = [re.sub(r"[ \t\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not BOILERPLATE_LINE.match(ln)]
    counts = {}
    for ln in lines:
        if ln:
            counts[ln] = counts.get(ln, 0) + 1
    lines = [ln for ln in lines if not (ln and counts[ln] >= 3 and len(ln) < 120)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def unwrap_lines(text):
    """PDF text breaks lines mid-sentence: join a line to the next when it does not end a sentence. Blank lines stay."""
    out, cur = [], ""
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            if cur:
                out.append(cur)
                cur = ""
            out.append("")
            continue
        cur = f"{cur} {ln}".strip() if cur else ln
        if re.search(r"[.!?:;।]$", ln) or len(ln) < 40:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ---- language and garbled-text detection --------------------------------------------------------------------------------
def _is_tamil(ch):
    return 0x0B80 <= ord(ch) <= 0x0BFF


def script_ratios(text):
    """-> (tamil, latin, letters): counts of Tamil-block letters (incl. vowel signs), ASCII Latin letters and all letters."""
    tamil = latin = letters = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] not in "LM":
            continue
        letters += 1
        if _is_tamil(ch):
            tamil += 1
        elif "a" <= ch.lower() <= "z":
            latin += 1
    return tamil, latin, letters


def detect_language(text):
    """'en', 'ta' or 'unknown' from Unicode script ratios (no model, no dictionary). Needs at least 20 letters."""
    tamil, latin, letters = script_ratios(text[:20000])
    if letters < 20 or looks_garbled(text):
        return "unknown"
    if tamil / letters >= 0.4:
        return "ta"
    if latin / letters >= 0.8 and tamil / letters < 0.1:
        return "en"
    return "unknown"


def looks_garbled(text):
    """Legacy-font Tamil PDFs extract as symbol soup, accented Latin-1 letters or odd capitals in the middle of words. Real
    English or Unicode Tamil is mostly letters (Tamil vowel signs count as letters), digits, spaces and punctuation."""
    sample = text[:4000]
    if not sample.strip():
        return True
    if "�" in sample or any(ord(ch) > 0xFFF0 for ch in sample):
        return True
    good = sum(1 for ch in sample if unicodedata.category(ch)[0] in "LMN" or ch in " \n\t.,;:()[]-–—/'\"‘’“”•…+=<>$%&₹")
    if good / len(sample) < 0.85:
        return True
    letters = sum(1 for ch in sample if unicodedata.category(ch)[0] == "L")
    if letters and sum(1 for ch in sample if 0x80 <= ord(ch) <= 0xFF and ch.isalpha()) / letters > 0.12:
        return True                                                    # Latin-1 accents everywhere: a legacy font read as Latin
    words = sample.split()
    tamil_words = [w for w in words if any(_is_tamil(c) for c in w)]
    if len(tamil_words) >= 10 and sum(1 for w in tamil_words if 0x0BBE <= ord(w[0]) <= 0x0BCD) / len(tamil_words) > 0.05:
        return True                                                    # a Tamil word never starts with a vowel sign or the virama
    latin_words = [w for w in words if re.fullmatch(r"[A-Za-z]{3,}", w)]
    if len(latin_words) >= 50 and sum(1 for w in latin_words if re.search(r"[a-z][A-Z]", w)) / len(latin_words) > 0.08:
        return True                                                    # lowercase-uppercase flips inside words
    return False


# ---- quality gate ---------------------------------------------------------------------------------------------------------
def quality(text, link_ratio=0.0, min_chars=MIN_DOC_CHARS):
    """-> (score 0..1, reasons). `reasons` is empty when the text may enter the corpus. Pass min_chars=0 for a passage
    (the 1,000 character rule is for whole documents); link_ratio is the share of link text from html_to_text_info()."""
    reasons = []
    text = text or ""
    if len(text) < min_chars:
        reasons.append("too_short")
    nonspace = [ch for ch in text if not ch.isspace()]
    letters = sum(1 for ch in nonspace if unicodedata.category(ch)[0] in "LM")
    letter_ratio = letters / len(nonspace) if nonspace else 0.0
    if letter_ratio < MIN_LETTER_RATIO:
        reasons.append("low_letter_ratio")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    short = [ln for ln in lines if len(ln.split()) <= 4 and not re.search(r"[.!?।]$", ln)]
    if link_ratio > 0.5 or (len(lines) >= 8 and len(short) / len(lines) >= 0.7):
        reasons.append("mostly_links_or_menus")
    unique_ratio = len(set(lines)) / len(lines) if lines else 0.0
    if len(lines) >= 6 and unique_ratio < 0.6:
        reasons.append("repeated_lines")
    words = text.split()
    singles = sum(1 for w in words if len(w) == 1 and w.isalpha() and w not in "aAI")
    if len(words) >= 50 and (sum(len(w) for w in words) / len(words) > 14 or singles / len(words) > 0.30):
        reasons.append("poor_extraction")
    elif text and text.count("�") / len(text) > 0.005:
        reasons.append("poor_extraction")
    if text.strip() and looks_garbled(text):
        reasons.append("garbled")
    sentence_ratio = min(1.0, len(re.findall(r"[.!?।]", text)) / max(1.0, len(words) / 25.0)) if words else 0.0
    score = 0.5 * letter_ratio + 0.3 * unique_ratio + 0.2 * sentence_ratio
    if reasons:
        score = min(score, 0.49)
    return round(max(0.0, min(1.0, score)), 3), reasons


# ---- passages ---------------------------------------------------------------------------------------------------------
_ABBREV = r"(?<!Mr\.)(?<!Dr\.)(?<!Ms\.)(?<!St\.)(?<!No\.)(?<!vs\.)(?<!Rs\.)(?<!Mrs\.)(?<!Tr\.)(?<!\b[A-Z]\.)"
SENTENCE_END = re.compile(_ABBREV + r"(?<=[.!?।])\s+(?=[\"'“‘(\[]?[A-Z0-9஀-௿])")


def _break_long(s):
    """Split a unit longer than UNIT_MAX at a comma/semicolon or a space (never inside a word unless there is no space at all)."""
    parts = []
    while len(s) > UNIT_MAX:
        window = s[:UNIT_MAX]
        cut = max(window.rfind(", "), window.rfind("; "), window.rfind(": "))
        cut = cut + 1 if cut >= UNIT_MAX // 2 else window.rfind(" ")
        if cut <= 0:
            cut = UNIT_MAX
        parts.append(s[:cut].strip())
        s = s[cut:].strip()
    if s:
        parts.append(s)
    return parts


def _units(text):
    """[(separator, unit)]: sentences (or pieces of long sentences); the separator is what precedes the unit in the text."""
    units = []
    for p_index, para in enumerate(re.split(r"\n\s*\n|\n", text.strip())):
        para = para.strip()
        if not para:
            continue
        first = True
        for sentence in SENTENCE_END.split(para):
            for piece in _break_long(sentence.strip()):
                sep = "" if not units else ("\n\n" if first else " ")
                units.append((sep, piece))
                first = False
    return units


def _is_heading(unit):
    return len(unit) < 80 and not re.search(r"[.!?।,;:]$", unit)


def _tail(prev, ratio=OVERLAP):
    """The last 10 to 20 percent of `prev`, starting on a sentence start if one falls in that window, otherwise on a word start."""
    n = len(prev)
    lo, hi, want = int(n * 0.10) + 1, int(n * 0.20), int(n * ratio)
    best = None
    for rule in ("sentence", "word"):
        for size in range(lo, hi + 1):
            s = n - size
            if s < 1 or not prev[s - 1].isspace() or prev[s].isspace():
                continue
            if rule == "sentence" and prev[s - 2:s - 1] not in (".", "!", "?", "।"):
                continue
            if best is None or abs(size - want) < abs(n - best - want):
                best = s
        if best is not None:
            break
    return prev[best if best is not None else n - want:].strip()


def split_passages(text, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    """Passages of 300 to 1,000 estimated tokens with a 10 to 20 percent overlap, cut on sentence/paragraph boundaries (never
    mid-word). A text that fits one passage is returned whole (it may then be shorter than min_tokens). Deterministic."""
    text = (text or "").strip()
    if not text:
        return []
    if token_estimate(text) <= max_tokens:
        return [text]
    units = _units(text)
    total = sum(len(s) + len(u) for s, u in units)
    n = int(math.ceil(total / (max_tokens * 4 * 0.65)))                 # about 2,600 new characters per passage
    target = total / float(n)
    cuts, cum, k = [0], 0, 1
    for i, (s, u) in enumerate(units):
        cum += len(s) + len(u)
        if k < n and cum >= k * target and i + 1 < len(units):
            j = i + 1
            while j > cuts[-1] + 1 and _is_heading(units[j - 1][1]):    # a heading belongs with the text after it
                j -= 1
            cuts.append(j)
            k += 1
    cuts.append(len(units))
    passages, prev = [], None
    for a, b in zip(cuts, cuts[1:]):
        body = "".join(s + u for s, u in units[a:b]).strip()
        passages.append(body if prev is None else _tail(prev, overlap) + (units[a][0] or " ") + body)
        prev = passages[-1]
    return passages


# ---- passage records ---------------------------------------------------------------------------------------------------
def build_passages(doc, text, language, doc_quality, seen=None):
    """Split a clean document into passage records (one dict per Parquet row, columns as PASSAGE_COLUMNS).
    `doc` = {doc_id, source, source_url, title, person_key, person_name, topic, created_at}. Passages that fail the passage-level
    quality gate or repeat an earlier passage (`seen`: a set of sha256 shared across documents in the run) are dropped.
    -> (records, {"duplicates": n, "rejected": n}). Positions are the index in the split, so ids never move when a duplicate is dropped."""
    seen = set() if seen is None else seen
    doc_sha = sha256_text(text)
    created = doc.get("created_at") or _now()
    records, dup, rejected = [], 0, 0
    for position, part in enumerate(split_passages(text)):
        score, reasons = quality(part, min_chars=0)
        if reasons:
            rejected += 1
            continue
        sha = sha256_text(part)
        if sha in seen:
            dup += 1
            continue
        seen.add(sha)
        records.append({
            "passage_id": passage_id(doc_sha, position), "doc_id": doc["doc_id"], "source": doc.get("source"),
            "source_url": doc.get("source_url"), "title": doc.get("title"), "text": part, "language": language,
            "char_count": len(part), "token_estimate": token_estimate(part), "sha256": sha, "created_at": created,
            "quality_score": score, "embedded": False, "indexed_neo4j": False, "person_key": doc.get("person_key"),
            "person_name": doc.get("person_name"), "topic": doc.get("topic"), "position": position,
            "corruption_related": bool(doc.get("topic") == "corruption" or CORRUPTION_WORDS.search(part)),
        })
    return records, {"duplicates": dup, "rejected": rejected, "document_sha256": doc_sha}


def row_of(record):
    """The record as a tuple in PASSAGE_COLUMNS order (what Spark's createDataFrame and pyarrow take)."""
    return tuple(record.get(name) for name in COLUMN_NAMES)


# ---- what is already written: id index + stats -------------------------------------------------------------------------------
class PassageIndex:
    """Passage ids (and text hashes) already in the Parquet corpus, kept in <corpus>/_state/passage_index.tsv (one `id<TAB>sha256`
    per line, append-only). Reprocessing a document finds its ids here and writes nothing twice; an identical text under another
    id is a duplicate. stage() records ids of the current batch; commit() appends them once the Parquet write succeeded."""

    def __init__(self, corpus_dir=CORPUS_DIR):
        self.path = Path(corpus_dir) / "_state" / "passage_index.tsv"
        self.ids, self.shas, self.pending = set(), set(), []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                pid, _, sha = line.partition("\t")
                if pid:
                    self.ids.add(pid)
                    self.shas.add(sha)

    def classify(self, record):
        """'known' (same passage id already written), 'duplicate' (same text under another id) or 'new'."""
        if record["passage_id"] in self.ids:
            return "known"
        return "duplicate" if record["sha256"] in self.shas else "new"

    def stage(self, record):
        self.ids.add(record["passage_id"])
        self.shas.add(record["sha256"])
        self.pending.append((record["passage_id"], record["sha256"]))

    def rollback(self):
        for pid, sha in self.pending:
            self.ids.discard(pid)
            self.shas.discard(sha)
        self.pending = []

    def commit(self):
        if self.pending:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.writelines(f"{pid}\t{sha}\n" for pid, sha in self.pending)
        self.pending = []


def partition_dir(corpus_dir, source, date):
    return Path(corpus_dir) / "passages" / f"source={source}" / f"date={date}"


def append_stats(corpus_dir, date, line):
    """One JSON object per batch in <corpus>/_stats/<date>.jsonl (the monitor sums them)."""
    path = Path(corpus_dir) / "_stats" / f"{date}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def stats_totals(corpus_dir=CORPUS_DIR):
    """Sum of every batch line: {processed, skipped, failed, duplicates_removed, passages, clean_tokens, batches}."""
    totals = {"processed": 0, "skipped": 0, "failed": 0, "duplicates_removed": 0, "passages": 0, "clean_tokens": 0, "batches": 0}
    for path in sorted((Path(corpus_dir) / "_stats").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            totals["batches"] += 1
            for key in totals:
                if key != "batches":
                    totals[key] += int(obj.get(key) or 0)
    return totals
