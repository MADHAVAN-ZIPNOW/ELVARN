"""The conversion matrix and the single entry point `convert`.

MATRIX (source -> targets). Layout, fonts, colours, images and comments are never kept: only the text,
headings, bullet items, code blocks and tables.
  txt  -> md, html, csv (one row per line), json, docx, pdf
  md   -> txt, html, json, docx, pdf
  csv  -> txt, md, html, json (list of objects), docx, pdf
  json -> txt, md, html, csv (only a list of records/rows), docx, pdf
  html -> txt, md, json, csv (first table only), docx, pdf
  docx -> txt, md, html, json, csv (first table only), pdf
PDF is a target only: reading PDFs safely needs a heavy library.
"""
from dataclasses import dataclass

from . import pdfwriter, readers, writers
from .errors import FileError, Unsupported
from .sniff import SOURCE_TYPES, TARGET_TYPES

MATRIX = {
    "txt": ("md", "html", "csv", "json", "docx", "pdf"),
    "md": ("txt", "html", "json", "docx", "pdf"),
    "csv": ("txt", "md", "html", "json", "docx", "pdf"),
    "json": ("txt", "md", "html", "csv", "docx", "pdf"),
    "html": ("txt", "md", "json", "csv", "docx", "pdf"),
    "docx": ("txt", "md", "html", "json", "csv", "pdf"),
}
assert set(MATRIX) == set(SOURCE_TYPES) and all(set(v) <= set(TARGET_TYPES) for v in MATRIX.values())

LABELS = {"txt": "TXT", "md": "Markdown", "html": "HTML", "csv": "CSV", "json": "JSON", "docx": "Word (.docx)", "pdf": "PDF"}
EXT = {"txt": "txt", "md": "md", "html": "html", "csv": "csv", "json": "json", "docx": "docx", "pdf": "pdf"}
MIME = {"txt": "text/plain; charset=utf-8", "md": "text/markdown; charset=utf-8", "html": "text/html; charset=utf-8",
        "csv": "text/csv; charset=utf-8", "json": "application/json",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "pdf": "application/pdf"}
WRITERS = {"txt": writers.to_txt, "md": writers.to_md, "html": writers.to_html, "csv": writers.to_csv,
           "json": writers.to_json, "docx": writers.to_docx}


PREVIEW_MAX_CHARS = 200_000


@dataclass
class Converted:
    data: bytes
    fmt: str
    lost_chars: int = 0
    preview: str = ""      # plain text of the document, only filled for docx/pdf outputs


def supported_targets(kind):
    return list(MATRIX.get(kind, ()))


def check_pair(source, target):
    if target not in TARGET_TYPES:
        raise Unsupported(f"I cannot make {target} files. I can make: {', '.join(TARGET_TYPES)}.")
    if source == target:
        raise FileError(f"That file is already a {LABELS[source]} file, so there is nothing to convert.")
    if target not in MATRIX.get(source, ()):
        raise Unsupported(f"I cannot convert {source.upper()} to {LABELS[target]}. "
                          f"From {source.upper()} I can make: {', '.join(LABELS[t] for t in MATRIX[source])}.")


def convert(source, data, target, name="", max_output=10 * 1024 * 1024):
    check_pair(source, target)
    doc = readers.read(source, data)
    lost = 0
    if target == "pdf":
        out, lost = pdfwriter.build_pdf(doc)
    else:
        out = WRITERS[target](doc, name)
    if len(out) > max_output:
        raise FileError("The converted file would be too large.")
    preview = ""
    if target in ("docx", "pdf"):
        preview = writers.to_txt(doc).decode("utf-8", "replace")[:PREVIEW_MAX_CHARS]
    return Converted(out, target, lost, preview)
