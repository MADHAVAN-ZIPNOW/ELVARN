"""Load official documents you downloaded (judgments, FIRs, press releases, audit reports ...) into the knowledge base.

Put files under a folder whose sub-folder name is the document type:

    backend/data/kb_inbox/
        judgment/       *.pdf *.txt *.html *.docx      (court judgments)
        order/          bail orders, interim orders
        fir/  chargesheet/  press_release/  audit_report/  affidavit/  rti_response/  assembly/  commission_report/  news/  reference/

Optional per-file details go in a sidecar named <file>.json, e.g. judgment/x.pdf.json:
    {"title": "State v. X, Madras HC", "url": "https://...", "published_date": "2021-03-04", "publisher": "Madras High Court"}

The same text is never loaded twice (SHA-256). Usage (project root):
    python -m backend.knowledge.ingest_folder [--inbox backend/data/kb_inbox]
"""
import argparse
import json
from pathlib import Path

from backend.files import readers
from backend.knowledge.ingest_wikipedia import chunk
from backend.knowledge.kb import KnowledgeBase

DOC_TYPES = ("fir", "chargesheet", "judgment", "order", "commission_report", "assembly", "news", "reference",
             "press_release", "audit_report", "affidavit", "rti_response", "court_record")
DEFAULT_INBOX = Path(__file__).resolve().parent.parent / "data" / "kb_inbox"
EXT_KIND = {".txt": "txt", ".md": "md", ".html": "html", ".htm": "html", ".docx": "docx"}


def doc_text(doc):
    """Plain text of a files.readers.Doc (tables become tab-separated lines)."""
    out = []
    for b in doc.blocks:
        if b[0] == "table":
            out += ["\t".join(row) for row in b[1]]
        else:
            out.append(b[-1])
    return "\n\n".join(t for t in out if t and t.strip())


def read_pdf(data):
    from io import BytesIO
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(data))
    pages = [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
    return [(n, t) for n, t in pages if t]


def file_pages(path):
    """[(locator, text)] for a file. A scanned PDF with no text layer returns [] (it needs OCR first)."""
    data = Path(path).read_bytes()
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return [(f"page {n}", t) for n, t in read_pdf(data)]
    if ext in EXT_KIND:
        return [(None, doc_text(readers.read(EXT_KIND[ext], data)))]
    raise ValueError(f"unsupported file type {ext}")


def ingest_file(kb, path, doc_type, rag=None, cache=None):
    from backend.knowledge.store import store_document
    meta_file = Path(str(path) + ".json")
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
    pages = file_pages(path)
    if not pages:
        return 0, "no text (scanned PDF? run OCR first)"
    full = "\n\n".join(t for _, t in pages)
    passages = []
    for loc, text in pages:
        for part in chunk(text):
            passages.append({"locator": loc or f"chunk {len(passages) + 1}", "text": part})
    status = store_document(kb, rag, title=meta.get("title") or Path(path).stem, doc_type=doc_type, text=full, passages=passages,
                            url=meta.get("url"), publisher=meta.get("publisher"), published_date=meta.get("published_date"),
                            licence=meta.get("licence"), cache=cache)
    return (len(passages), "ok") if status == "loaded" else (0, "already loaded")


def ingest_inbox(kb, inbox):
    summary = []
    for folder in sorted(p for p in Path(inbox).glob("*") if p.is_dir()):
        if folder.name not in DOC_TYPES:
            summary.append((str(folder), 0, f"unknown document type '{folder.name}'"))
            continue
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".pdf", *EXT_KIND):
                try:
                    n, msg = ingest_file(kb, f, folder.name)
                except Exception as exc:                  # one bad file must not stop the batch
                    n, msg = 0, f"failed: {exc}"
                summary.append((str(f), n, msg))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", default=str(DEFAULT_INBOX))
    args = ap.parse_args()
    Path(args.inbox).mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase()
    for path, n, msg in ingest_inbox(kb, args.inbox):
        print(f"{n:4d} passages  {msg:12s} {path}")


if __name__ == "__main__":
    main()
