"""Readers: uploaded bytes -> a small, plain document model (Doc).

Every reader treats the file as untrusted data: nothing is executed, no URL is fetched, and the output holds
only plain text. Layout, fonts, colours and images are dropped on purpose.

Block kinds:
    ("h", level, text)   heading, level 1-6
    ("p", text)          paragraph (may contain newlines)
    ("li", text)         bullet item
    ("pre", text)        preformatted text (code, JSON)
    ("table", rows)      rows = list of lists of strings (first row is usually the header)
"""
import csv
import json
import re
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

from .errors import FileError
from .sniff import check_json, decode_text, read_zip_member, zip_guard

MAX_BLOCKS = 20000
MAX_CHARS = 2_000_000
MAX_ROWS = 10000
MAX_COLS = 100
MAX_CELL = 5000
MAX_XML_NODES = 400000

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class Doc:
    def __init__(self, kind):
        self.kind = kind
        self.blocks = []
        self.chars = 0
        self.records = None     # for csv/json sources: (header list, rows) when the data is a clean table

    def add(self, *block):
        self.blocks.append(block)
        self.chars += self._size(block)
        if len(self.blocks) > MAX_BLOCKS or self.chars > MAX_CHARS:
            raise FileError("This file has too much text to convert here (limit about 2 million characters).")

    @staticmethod
    def _size(block):
        last = block[-1]
        if isinstance(last, list):
            return sum(len(c) for row in last for c in row)
        return len(last)

    def tables(self):
        return [b[1] for b in self.blocks if b[0] == "table"]

    def has_text(self):
        for b in self.blocks:
            if b[0] == "table":
                if any(c.strip() for row in b[1] for c in row):
                    return True
            elif b[-1].strip():
                return True
        return False


def _norm(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def _cell(value):
    return str(value)[:MAX_CELL]


# ---------------- txt ----------------
def read_txt(data):
    text = _norm(decode_text(data))
    doc = Doc("txt")
    for para in re.split(r"\n\s*\n", text):
        para = "\n".join(line.rstrip() for line in para.strip("\n").split("\n"))
        if para.strip():
            doc.add("p", para)
    return doc


# ---------------- markdown ----------------
def read_md(data):
    text = _norm(decode_text(data))
    doc = Doc("md")
    para, code, in_code, items = [], [], False, []

    def flush_para():
        if para:
            doc.add("p", " ".join(s.strip() for s in para))
            para.clear()

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            if in_code:
                doc.add("pre", "\n".join(code))
                code.clear()
                in_code = False
            else:
                flush_para()
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            flush_para()
            doc.add("h", len(m.group(1)), _inline(m.group(2)))
            continue
        m = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$", line)
        if m:
            flush_para()
            doc.add("li", _inline(m.group(1)))
            continue
        if line.count("|") >= 2 and i < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i]):
            flush_para()
            rows = [_split_row(line)]
            i += 1
            while i < len(lines) and lines[i].count("|") >= 1 and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            doc.add("table", rows[:MAX_ROWS])
            continue
        if not line.strip():
            flush_para()
            continue
        if re.match(r"^\s*([-*_])\s*(\1\s*){2,}$", line):
            flush_para()
            continue
        para.append(_inline(re.sub(r"^\s*>\s?", "", line)))
    if in_code and code:
        doc.add("pre", "\n".join(code))
    flush_para()
    return doc


def _split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [_cell(_inline(c.strip())) for c in re.split(r"(?<!\\)\|", line)][:MAX_COLS]


def _inline(text):
    """Drop inline markdown marks, keep the words. Links become 'text (url)' as plain text."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)[^)]*\)", r"\1 (\2)", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?<![\w*])([*_])(?!\s)(.+?)(?<!\s)\1(?![\w*])", r"\2", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text.replace("\\|", "|")


# ---------------- csv ----------------
def read_csv(data):
    text = _norm(decode_text(data))
    first = text.split("\n", 1)[0]
    counts = {d: first.count(d) for d in (",", ";", "\t", "|")}
    delim = max(counts, key=counts.get) if max(counts.values()) > 0 else ","
    rows = []
    try:
        for row in csv.reader(text.split("\n"), delimiter=delim):
            if not row:
                continue
            rows.append([_cell(c) for c in row[:MAX_COLS]])
            if len(rows) > MAX_ROWS:
                raise FileError(f"This CSV has more than {MAX_ROWS} rows, which is more than this tool converts.")
    except csv.Error:
        raise FileError("This CSV file could not be read (a cell is too large or the quotes are broken).")
    if not rows:
        raise FileError("This file is empty.")
    doc = Doc("csv")
    doc.add("table", rows)
    doc.records = (rows[0], rows[1:])
    return doc


# ---------------- json ----------------
def _scalar(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return _cell(json.dumps(v, ensure_ascii=False, separators=(",", ":")))
    return _cell(v)


def read_json(data):
    text = decode_text(data)
    check_json(text)
    value = json.loads(text)
    doc = Doc("json")
    if isinstance(value, list) and value and len(value) <= MAX_ROWS and all(isinstance(r, dict) for r in value):
        header = []
        for r in value:
            for k in r:
                if k not in header and len(header) < MAX_COLS:
                    header.append(k)
        header = [_cell(h) for h in header]
        rows = [[_scalar(r.get(h)) for h in header] for r in value]
        doc.add("table", [header] + rows)
        doc.records = (header, rows)
    elif isinstance(value, list) and value and len(value) <= MAX_ROWS and all(isinstance(r, list) for r in value):
        rows = [[_scalar(c) for c in r[:MAX_COLS]] for r in value]
        doc.add("table", rows)
    else:
        doc.add("pre", json.dumps(value, ensure_ascii=False, indent=2))
    return doc


# ---------------- html ----------------
SKIP = {"script", "style", "iframe", "object", "embed", "noscript", "template", "svg", "head", "applet", "frame",
        "frameset", "canvas", "audio", "video", "math", "select", "textarea", "button", "datalist", "dialog"}
BLOCK = {"p", "div", "section", "article", "header", "footer", "main", "aside", "nav", "blockquote", "ul", "ol",
         "dl", "dt", "dd", "hr", "figure", "figcaption", "address", "form", "fieldset", "details", "summary", "caption"}


class _TextHTML(HTMLParser):
    """Collects visible text only. Scripts, styles, frames, forms' controls and every attribute are discarded,
    so nothing active can survive into the output."""

    def __init__(self, doc):
        super().__init__(convert_charrefs=True)
        self.doc, self.skip = doc, 0
        self.buf, self.mode, self.level = [], "p", 0
        self.tdepth, self.rows, self.row, self.cell = 0, None, None, None

    def _flush(self):
        raw = "".join(self.buf)
        self.buf = []
        if self.mode == "pre":
            text = raw.strip("\n")
        else:
            text = "\n".join(" ".join(line.split()) for line in raw.split("\n"))
            text = re.sub(r"\n{2,}", "\n", text).strip()
        if text:
            if self.mode == "h":
                self.doc.add("h", self.level, text)
            elif self.mode == "li":
                self.doc.add("li", text)
            elif self.mode == "pre":
                self.doc.add("pre", text)
            else:
                self.doc.add("p", text)
        self.mode = "p"

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
            self._flush()
            self.mode, self.level = "h", int(tag[1])
        elif tag == "li":
            self._flush()
            self.mode = "li"
        elif tag == "pre":
            self._flush()
            self.mode = "pre"
        elif tag == "br":
            (self.cell if self.cell is not None else self.buf).append("\n")
        elif tag == "table":
            if self.tdepth == 0:
                self._flush()
                self.rows = []
            self.tdepth += 1
        elif tag == "tr" and self.tdepth == 1:
            self.row = []
        elif tag in ("td", "th") and self.tdepth == 1:
            self.cell = []
        elif tag in BLOCK and self.cell is None:
            self._flush()

    def handle_startendtag(self, tag, attrs):
        if tag == "br" or tag == "hr":
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if (len(tag) == 2 and tag[0] == "h" and tag[1] in "123456") or tag in ("li", "pre"):
            self._flush()
        elif tag in ("td", "th") and self.tdepth == 1 and self.cell is not None:
            if self.row is not None and len(self.row) < MAX_COLS:
                self.row.append(_cell(" ".join("".join(self.cell).split())))
            self.cell = None
        elif tag == "tr" and self.tdepth == 1 and self.row is not None:
            if any(c for c in self.row) and len(self.rows) < MAX_ROWS:
                self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.tdepth > 0:
            self.tdepth -= 1
            if self.tdepth == 0 and self.rows:
                self.doc.add("table", self.rows)
                self.rows = None
        elif tag in BLOCK and self.cell is None:
            self._flush()

    def handle_data(self, data):
        if self.skip:
            return
        (self.cell if self.cell is not None else self.buf).append(data)


def read_html(data):
    text = _norm(decode_text(data))
    doc = Doc("html")
    parser = _TextHTML(doc)
    try:
        parser.feed(text)
        parser.close()
    except (RecursionError, AssertionError):
        raise FileError("This HTML file could not be read.")
    parser._flush()
    return doc


# ---------------- docx ----------------
def _para_text(p):
    out = []
    n = 0
    for node in p.iter():
        n += 1
        if n > MAX_XML_NODES:
            raise FileError("This Word file is too complex to convert here.")
        if node.tag == W + "t":
            out.append(node.text or "")
        elif node.tag in (W + "tab",):
            out.append("\t")
        elif node.tag in (W + "br", W + "cr"):
            out.append("\n")
    return "".join(out)


def read_docx(data):
    zf = zip_guard(data)
    try:
        xml = read_zip_member(zf, "word/document.xml")
    finally:
        zf.close()
    head = xml.lower()
    # No DTD and no entities at all: this removes entity-expansion ("billion laughs") attacks completely.
    if b"<!doctype" in head or b"<!entity" in head:
        raise FileError("This Word file contains a DTD or entity definitions and was refused for safety.")
    try:
        root = ET.fromstring(xml)
    except (ET.ParseError, RecursionError):
        raise FileError("This Word file's text could not be read.")
    body = root.find(W + "body")
    doc = Doc("docx")
    if body is None:
        raise FileError("This Word file has no body text.")
    for child in list(body):
        if child.tag == W + "p":
            text = _para_text(child).strip("\n")
            if not text.strip():
                continue
            style = ""
            ppr = child.find(W + "pPr")
            if ppr is not None:
                ps = ppr.find(W + "pStyle")
                if ps is not None:
                    style = ps.get(W + "val", "") or ""
                numbered = ppr.find(W + "numPr") is not None
            else:
                numbered = False
            m = re.match(r"(?i)^heading\s*([1-6])$", style)
            if m:
                doc.add("h", int(m.group(1)), text)
            elif style.lower() == "title":
                doc.add("h", 1, text)
            elif numbered or style.lower().startswith("list") or text.startswith("• "):
                doc.add("li", text[2:] if text.startswith("• ") else text)
            else:
                doc.add("p", text)
        elif child.tag == W + "tbl":
            rows = []
            for tr in child.findall(W + "tr")[:MAX_ROWS]:
                cells = []
                for tc in tr.findall(W + "tc")[:MAX_COLS]:
                    cells.append(_cell(" ".join(_para_text(p).strip() for p in tc.findall(W + "p")).strip()))
                rows.append(cells)
            if rows:
                doc.add("table", rows)
    return doc


READERS = {"txt": read_txt, "md": read_md, "csv": read_csv, "json": read_json, "html": read_html, "docx": read_docx}


def read(kind, data):
    doc = READERS[kind](data)
    if not doc.has_text():
        raise FileError("No readable text was found in this file.")
    return doc
