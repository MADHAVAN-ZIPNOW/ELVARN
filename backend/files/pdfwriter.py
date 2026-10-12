"""A small, text-only PDF writer using the standard Helvetica / Helvetica-Bold / Courier fonts (no dependencies).

Limits, on purpose:
* Only characters in the Windows-1252 set (English and Western European text) can be drawn. Other characters
  (Chinese, Arabic, Hindi, emoji, ...) become "?" and the caller is told how many were replaced.
* No images, no colours, no links; tables are drawn as plain aligned text.
"""
import unicodedata
import zlib

from .errors import FileError

PAGE_W, PAGE_H, MARGIN = 595, 842, 56
MAX_PAGES = 400
BODY = 11
CODE = 9

_HELV = {" ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667, "'": 191, "(": 333, ")": 333, "*": 389,
         "+": 584, ",": 278, "-": 333, ".": 278, "/": 278, ":": 278, ";": 278, "<": 584, "=": 584, ">": 584, "?": 556,
         "@": 1015, "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556, "`": 333, "{": 334, "|": 260, "}": 334, "~": 584}
for _c in "0123456789":
    _HELV[_c] = 556
for _c, _w in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                  (667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611,
                   722, 667, 944, 667, 667, 611)):
    _HELV[_c] = _w
for _c, _w in zip("abcdefghijklmnopqrstuvwxyz",
                  (556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278,
                   556, 500, 722, 500, 500, 500)):
    _HELV[_c] = _w


def char_width(ch, font, size):
    if font == "F3":
        return 0.6 * size
    w = _HELV.get(ch, 556 if ord(ch) < 0x2000 else 1000)
    if font == "F2":
        w *= 1.1
    return w * size / 1000.0


def text_width(text, font, size):
    return sum(char_width(c, font, size) for c in text)


def to_latin(text):
    """-> (text using only Windows-1252 characters, number of characters that had to be replaced)."""
    text = unicodedata.normalize("NFC", text.replace("\t", "    "))
    out, lost = [], 0
    for ch in text:
        if ch == "\n":
            out.append(ch)
            continue
        if ord(ch) < 32 or ord(ch) == 127:
            continue
        try:
            ch.encode("cp1252")
            out.append(ch)
        except UnicodeEncodeError:
            if unicodedata.category(ch) in ("Mn", "Cf", "Zs"):
                out.append(" " if unicodedata.category(ch) == "Zs" else "")
            else:
                out.append("?")
                lost += 1
    return "".join(out), lost


def pdf_string(text):
    """Escape for a PDF literal string: backslash and parentheses are escaped, anything outside printable ASCII is octal."""
    out = []
    for b in text.encode("cp1252", errors="replace"):
        if b in (0x5C, 0x28, 0x29):
            out.append("\\" + chr(b))
        elif 32 <= b < 127:
            out.append(chr(b))
        else:
            out.append("\\%03o" % b)
    return "".join(out)


def wrap(text, font, size, width):
    """Split one paragraph into lines no wider than `width` (points). Over-long words are cut by characters."""
    lines, cur, cur_w = [], None, 0.0
    space = char_width(" ", font, size)
    for word in text.split(" "):
        ww = text_width(word, font, size)
        if cur is not None and cur_w + space + ww <= width:
            cur += " " + word
            cur_w += space + ww
            continue
        if cur is not None:
            lines.append(cur)
        while ww > width and word:
            piece, pw = "", 0.0
            for ch in word:
                cw = char_width(ch, font, size)
                if pw + cw > width and piece:
                    break
                piece += ch
                pw += cw
            lines.append(piece)
            word = word[len(piece):]
            ww = text_width(word, font, size)
        cur, cur_w = word, ww
    lines.append(cur or "")
    return lines


def table_lines(rows, max_chars):
    cols = max((len(r) for r in rows), default=0)
    widths = [0] * cols
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = min(30, max(widths[i], len(c.replace("\n", " "))))
    out = []
    for r in rows:
        cells = [(r[i].replace("\n", " ") if i < len(r) else "")[:widths[i]].ljust(widths[i]) for i in range(cols)]
        line = " | ".join(cells).rstrip()
        out.extend(line[i:i + max_chars] for i in range(0, max(1, len(line)), max_chars))
    return out


def layout(doc):
    """Doc -> (list of items, lost characters). Item = (font, size, indent, space_before, text)."""
    items, lost = [], 0
    width = PAGE_W - 2 * MARGIN
    code_chars = int(width / (0.6 * CODE))

    def add_text(text, font, size, indent=0.0, before=0.0, bullet=False):
        nonlocal lost
        latin, n = to_latin(text)
        lost += n
        first = True
        for para in latin.split("\n"):
            if bullet and first:
                para = "\x95 " + para
            for k, line in enumerate(wrap(para, font, size, width - indent)):
                items.append((font, size, indent + (12.0 if bullet and not (first and k == 0) else 0.0), before if (first and k == 0) else 0.0, line))
            first = False

    for block in doc.blocks:
        kind = block[0]
        if kind == "h":
            size = {1: 18, 2: 15, 3: 13}.get(block[1], 12)
            add_text(block[2], "F2", size, before=size * 0.8)
        elif kind == "p":
            add_text(block[1], "F1", BODY, before=5)
        elif kind == "li":
            add_text(block[1], "F1", BODY, indent=8, before=2, bullet=True)
        elif kind == "pre":
            latin, n = to_latin(block[1])
            lost += n
            first = True
            for line in latin.split("\n"):
                for i in range(0, max(1, len(line)), code_chars):
                    items.append(("F3", CODE, 0.0, 6 if first else 0.0, line[i:i + code_chars]))
                    first = False
        elif kind == "table":
            cells = [[to_latin(c) for c in row] for row in block[1]]
            lost += sum(n for row in cells for _, n in row)
            rows = [[t for t, _ in row] for row in cells]
            first = True
            for line in table_lines(rows, code_chars):
                items.append(("F3", CODE, 0.0, 6 if first else 0.0, line))
                first = False
    return items, lost


def paginate(items):
    pages, ops, y = [], [], PAGE_H - MARGIN
    for font, size, indent, before, text in items:
        lead = size * 1.35
        y -= before
        if y - lead < MARGIN:
            pages.append(ops)
            if len(pages) >= MAX_PAGES:
                raise FileError(f"The result would be longer than {MAX_PAGES} pages, which is more than this tool makes.")
            ops, y = [], PAGE_H - MARGIN
        y -= lead
        if text.strip():
            ops.append((font, size, MARGIN + indent, y, text))
    pages.append(ops)
    return pages


def build_pdf(doc):
    """Doc -> (PDF bytes, number of characters replaced by '?')."""
    items, lost = layout(doc)
    pages = paginate(items)
    objs = {}
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    first_page = 6
    kids = " ".join(f"{first_page + 2 * i} 0 R" for i in range(len(pages)))
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()
    for num, name in ((3, "Helvetica"), (4, "Helvetica-Bold"), (5, "Courier")):
        objs[num] = f"<< /Type /Font /Subtype /Type1 /BaseFont /{name} /Encoding /WinAnsiEncoding >>".encode()
    for i, ops in enumerate(pages):
        pnum, cnum = first_page + 2 * i, first_page + 2 * i + 1
        parts = []
        for font, size, x, y, text in ops:
            parts.append(f"BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({pdf_string(text)}) Tj ET")
        stream = zlib.compress("\n".join(parts).encode("ascii"))
        objs[pnum] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] /Contents {cnum} 0 R "
                      f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> >>").encode()
        objs[cnum] = (f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n").encode() + stream + b"\nendstream"
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    xref = len(out)
    n = max(objs) + 1
    out += f"xref\n0 {n}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, n):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out), lost
