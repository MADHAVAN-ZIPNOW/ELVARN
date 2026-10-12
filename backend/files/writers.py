"""Writers: Doc -> bytes in a target format. Output is built from plain text only, never from raw input markup."""
import csv
import html
import io
import json
import re
import zipfile

from .errors import FileError
from .readers import MAX_ROWS

_XML_BAD = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")
_NUMBER = re.compile(r"^[-+]?\d+(\.\d+)?$")


# ---------------- txt ----------------
def aligned_rows(rows, cap=40):
    cols = max((len(r) for r in rows), default=0)
    widths = [0] * cols
    flat = [[c.replace("\n", " ") for c in r] for r in rows]
    for r in flat:
        for i, c in enumerate(r):
            widths[i] = min(cap, max(widths[i], len(c)))
    return [" | ".join((r[i] if i < len(r) else "").ljust(widths[i]) for i in range(cols)).rstrip() for r in flat]


def to_txt(doc, name=""):
    out = []
    for b in doc.blocks:
        if b[0] == "h":
            out.append(b[2])
        elif b[0] == "li":
            out.append("- " + b[1])
        elif b[0] == "table":
            out.append("\n".join(aligned_rows(b[1])))
        else:
            out.append(b[1])
    return ("\n\n".join(out) + "\n").encode("utf-8")


# ---------------- markdown ----------------
def _md_escape(text):
    text = re.sub(r"([\\*_`\[\]<>])", r"\\\1", text)
    return re.sub(r"(?m)^(\s*)([#+\-]|\d+[.)])(\s)", lambda m: m.group(1) + "\\" + m.group(2) + m.group(3), text)


def _md_cell(text):
    return _md_escape(text.replace("\n", " ")).replace("|", "\\|")


def to_md(doc, name=""):
    out = []
    prev_li = False
    for b in doc.blocks:
        kind = b[0]
        if kind == "li":
            out.append(("li", "- " + _md_escape(b[1]).replace("\n", " ")))
            continue
        if kind == "h":
            piece = "#" * b[1] + " " + _md_escape(b[2]).replace("\n", " ")
        elif kind == "p":
            piece = _md_escape(b[1]).replace("\n", "  \n")
        elif kind == "pre":
            fence = "```"
            while fence in b[1]:
                fence += "`"
            piece = f"{fence}\n{b[1]}\n{fence}"
        else:
            rows = b[1]
            cols = max(len(r) for r in rows)
            norm = [[_md_cell(r[i]) if i < len(r) else "" for i in range(cols)] for r in rows]
            lines = ["| " + " | ".join(norm[0]) + " |", "| " + " | ".join(["---"] * cols) + " |"]
            lines += ["| " + " | ".join(r) + " |" for r in norm[1:]]
            piece = "\n".join(lines)
        out.append(("x", piece))
    text, prev = "", None
    for kind, piece in out:
        text += ("" if prev is None else "\n" if (kind == "li" and prev == "li") else "\n\n") + piece
        prev = kind
    return (text + "\n").encode("utf-8")


# ---------------- html ----------------
CSS = ("body{font-family:Arial,Helvetica,sans-serif;max-width:48rem;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1f2a44}"
       "table{border-collapse:collapse}td,th{border:1px solid #999;padding:.3rem .6rem;text-align:left}"
       "pre{background:#f3f4f8;padding:.8rem;overflow:auto}")


def to_html(doc, name=""):
    e = lambda s: html.escape(s, quote=True)  # noqa: E731
    body, in_list = [], False
    for b in doc.blocks:
        if b[0] != "li" and in_list:
            body.append("</ul>")
            in_list = False
        if b[0] == "h":
            body.append(f"<h{b[1]}>{e(b[2])}</h{b[1]}>")
        elif b[0] == "p":
            body.append("<p>" + e(b[1]).replace("\n", "<br>\n") + "</p>")
        elif b[0] == "li":
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append("<li>" + e(b[1]).replace("\n", "<br>") + "</li>")
        elif b[0] == "pre":
            body.append("<pre>" + e(b[1]) + "</pre>")
        else:
            rows = b[1]
            parts = ["<table>"]
            for i, r in enumerate(rows):
                tag = "th" if i == 0 else "td"
                parts.append("<tr>" + "".join(f"<{tag}>{e(c)}</{tag}>" for c in r) + "</tr>")
            parts.append("</table>")
            body.append("\n".join(parts))
    if in_list:
        body.append("</ul>")
    title = e(name) if name else "Document"
    # The CSP below blocks scripts, frames and remote loads even if this file is opened in a browser.
    head = ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n<meta name="referrer" content="no-referrer">\n'
            f"<title>{title}</title>\n<style>{CSS}</style>\n</head>\n<body>\n")
    return (head + "\n".join(body) + "\n</body>\n</html>\n").encode("utf-8")


# ---------------- csv ----------------
def csv_safe(cell):
    """Stop spreadsheet formula injection: a cell that starts with = + - @ (or a tab/CR) gets a leading apostrophe."""
    cell = str(cell)
    if cell and cell[0] in "=+-@\t\r" and not _NUMBER.match(cell):
        return "'" + cell
    return cell


def to_csv(doc, name=""):
    tables = doc.tables()
    if tables:
        rows = tables[0]
    elif doc.kind == "txt":
        lines = [ln for p in doc.blocks for ln in p[1].split("\n") if ln.strip()]
        rows = [["text"]] + [[ln] for ln in lines]
    else:
        raise FileError("There is no table in this file, so it cannot become a CSV. CSV only works for tables "
                        "(or a plain text file, which becomes one line per row).")
    if len(rows) > MAX_ROWS + 1:
        raise FileError(f"Too many rows for CSV output (limit {MAX_ROWS}).")
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\r\n")
    for r in rows:
        w.writerow([csv_safe(c) for c in r])
    return buf.getvalue().encode("utf-8-sig")


# ---------------- json ----------------
def _unique_headers(header):
    seen, out = {}, []
    for i, h in enumerate(header):
        h = h.strip() or f"column_{i + 1}"
        n = seen.get(h, 0)
        seen[h] = n + 1
        out.append(h if n == 0 else f"{h}_{n + 1}")
    return out


def to_json(doc, name=""):
    """Meaning: a CSV (or a JSON list of records) becomes a list of objects keyed by the header row;
    anything else becomes {"blocks": [{"type", "text"} ...]} in reading order."""
    if doc.records is not None and doc.kind in ("csv", "json"):
        header, rows = doc.records
        keys = _unique_headers(header)
        data = [{k: (r[i] if i < len(r) else "") for i, k in enumerate(keys)} for r in rows]
    else:
        blocks = []
        for b in doc.blocks:
            if b[0] == "h":
                blocks.append({"type": "heading", "level": b[1], "text": b[2]})
            elif b[0] == "table":
                blocks.append({"type": "table", "rows": b[1]})
            else:
                blocks.append({"type": {"p": "paragraph", "li": "list_item", "pre": "code"}[b[0]], "text": b[1]})
        data = {"blocks": blocks}
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


# ---------------- docx ----------------
NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _x(text):
    return html.escape(_XML_BAD.sub("", text), quote=False).encode("utf-8", "replace").decode("utf-8")


def _runs(text, mono=False, bold=False):
    rpr = ""
    if mono or bold:
        rpr = "<w:rPr>" + ('<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/>' if mono else "") + ("<w:b/>" if bold else "") + "</w:rPr>"
    parts = []
    for i, line in enumerate(text.split("\n")):
        if i:
            parts.append("<w:r><w:br/></w:r>")
        parts.append(f'<w:r>{rpr}<w:t xml:space="preserve">{_x(line)}</w:t></w:r>')
    return "".join(parts)


def _para(text, style=None, **kw):
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}{_runs(text, **kw)}</w:p>"


def _docx_body(doc):
    out = []
    for b in doc.blocks:
        if b[0] == "h":
            out.append(_para(b[2], f"Heading{min(b[1], 6)}"))
        elif b[0] == "p":
            out.append(_para(b[1]))
        elif b[0] == "li":
            out.append(_para("• " + b[1], "ListParagraph"))
        elif b[0] == "pre":
            out.append(_para(b[1], mono=True))
        else:
            rows = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/></w:tblPr>']
            for r in (r for r in b[1] if r):
                rows.append("<w:tr>" + "".join(f"<w:tc><w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>{_para(c)}</w:tc>" for c in r) + "</w:tr>")
            rows.append("</w:tbl>")
            out.append("".join(rows))
            out.append("<w:p/>")
    return "".join(out)


def _styles():
    heads = "".join(
        f'<w:style w:type="paragraph" w:styleId="Heading{i}"><w:name w:val="heading {i}"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>'
        f'<w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="80"/><w:outlineLvl w:val="{i - 1}"/></w:pPr>'
        f'<w:rPr><w:b/><w:sz w:val="{max(24, 40 - 4 * i)}"/></w:rPr></w:style>' for i in range(1, 7))
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles {NS}>'
            '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/><w:sz w:val="22"/></w:rPr></w:rPrDefault>'
            '<w:pPrDefault><w:pPr><w:spacing w:after="120"/></w:pPr></w:pPrDefault></w:docDefaults>'
            '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
            + heads +
            '<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/>'
            '<w:pPr><w:ind w:left="540" w:hanging="270"/><w:spacing w:after="40"/></w:pPr></w:style>'
            '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders>'
            + "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="808080"/>' for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
            + '</w:tblBorders></w:tblPr></w:style></w:styles>')


def to_docx(doc, name=""):
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {NS}><w:body>{_docx_body(doc)}'
                '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
                '</w:body></w:document>')
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                     '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                     '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '</Relationships>')
    doc_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                '</Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/styles.xml", _styles())
    return buf.getvalue()
