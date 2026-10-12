"""Upload checks: filename cleaning, allowlist, and content sniffing (extension AND content must agree)."""
import io
import json
import re
import unicodedata
import zipfile

from .errors import FileError, Unsupported

# extension -> our type name. This is the ALLOWLIST. Anything else is refused.
EXTENSIONS = {"txt": "txt", "text": "txt", "md": "md", "markdown": "md", "csv": "csv", "json": "json",
              "html": "html", "htm": "html", "docx": "docx"}
SOURCE_TYPES = ("txt", "md", "csv", "json", "html", "docx")
TARGET_TYPES = ("txt", "md", "html", "csv", "json", "docx", "pdf")
ACCEPT_ATTR = ".txt,.md,.markdown,.csv,.json,.html,.htm,.docx"

# files that can run code or carry macros: refused with a stronger message
DANGEROUS = {"exe", "dll", "bat", "cmd", "com", "msi", "scr", "lnk", "js", "mjs", "vbs", "vbe", "wsf", "ps1", "sh", "jar",
             "py", "php", "zip", "rar", "7z", "tar", "gz", "iso", "docm", "dotm", "xlsm", "pptm", "xlam", "svg", "hta", "apk"}

RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
MAX_NAME = 100

# ---- docx (zip) guards ----
MAX_ZIP_ENTRIES = 200
MAX_ZIP_UNCOMPRESSED = 25 * 1024 * 1024
MAX_ZIP_RATIO = 100
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 64

MAGIC = [(b"MZ", "a Windows program"), (b"\x7fELF", "a program"), (b"%PDF", "a PDF"), (b"PK\x03\x04", "a zip archive"),
         (b"\x89PNG", "an image"), (b"GIF8", "an image"), (b"\xff\xd8\xff", "an image"), (b"Rar!", "an archive"),
         (b"7z\xbc\xaf", "an archive"), (b"\x1f\x8b", "an archive"), (b"\xd0\xcf\x11\xe0", "an old Office file"),
         (b"#!", "a script")]

_BAD_NAME_CHARS = set('<>:"|?*‮‭')


def clean_filename(raw, default="file"):
    """A safe display name: no folders, no control characters, no reserved Windows names, capped length.
    The result is only ever shown or used in a download header; it is never used to build a path."""
    name = unicodedata.normalize("NFC", str(raw or ""))
    name = re.split(r"[\\/]", name)[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C" and ch not in _BAD_NAME_CHARS)
    name = name.strip(" .")
    if not name:
        return default
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = stem.strip(" .") or default
    if stem.split(".")[0].lower() in RESERVED:
        stem = "_" + stem
    ext = ext[:10]
    room = max(1, MAX_NAME - len(ext) - (1 if ext else 0))
    stem = stem[:room]
    return stem + ("." + ext if ext else "")


def extension_of(name):
    return name.rpartition(".")[2].lower() if "." in name else ""


def allowed_text():
    return "txt, md, csv, json, html, docx"


def refuse_extension(ext):
    if ext in DANGEROUS:
        return Unsupported(f".{ext} files can run code or hide macros, so they are never accepted. "
                           f"Allowed types: {allowed_text()}.")
    if ext == "pdf":
        return Unsupported("PDF can be created by this tool but not read (reading PDFs safely needs a heavy library). "
                           f"Upload one of: {allowed_text()}.")
    shown = f".{ext} files" if ext else "Files without an extension"
    return Unsupported(f"{shown} are not supported. Allowed types: {allowed_text()}. "
                       "Only a short list is allowed because every file type needs its own safe reader.")


def decode_text(data):
    """Bytes -> str, or raise FileError if it looks binary."""
    bom16 = data.startswith((b"\xff\xfe", b"\xfe\xff"))
    if b"\x00" in data and not bom16:
        raise FileError("This file looks like binary data, not text.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        if bom16:
            try:
                text = data.decode("utf-16")
            except UnicodeDecodeError:
                raise FileError("This file looks like binary data, not text.")
        else:
            text = data.decode("cp1252", errors="replace")
    sample = text[:20000]
    bad = sum(1 for c in sample if unicodedata.category(c) == "Cc" and c not in "\t\n\r\f")
    if sample and bad / len(sample) > 0.02:
        raise FileError("This file looks like binary data, not text.")
    return text


def zip_guard(data):
    """Open data as a zip and refuse bombs. Returns the ZipFile (caller closes it)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise FileError("This .docx is not a valid Word file (it is not a readable zip).")
    infos = zf.infolist()
    if len(infos) > MAX_ZIP_ENTRIES:
        zf.close()
        raise FileError("This .docx has too many parts and was refused.")
    total = 0
    for i in infos:
        total += i.file_size
        bomb = (i.file_size > MAX_XML_BYTES * 2 or total > MAX_ZIP_UNCOMPRESSED
                or (i.file_size > 1024 * 1024 and i.file_size > MAX_ZIP_RATIO * max(1, i.compress_size)))
        if bomb:
            zf.close()
            raise FileError("This .docx expands to too much data and was refused (possible zip bomb).")
        if i.flag_bits & 0x1:
            zf.close()
            raise FileError("Encrypted .docx files are not supported.")
    return zf


def read_zip_member(zf, name, limit=MAX_XML_BYTES):
    """Read one member, never more than `limit` bytes, whatever the zip header claims."""
    with zf.open(name) as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise FileError("A part of this .docx is too large and was refused.")
    return data


def check_docx(data):
    zf = zip_guard(data)
    try:
        names = set(zf.namelist())
        if "[Content_Types].xml" not in names or "word/document.xml" not in names:
            raise FileError("This does not look like a Word .docx document.")
        if any("vbaproject" in n.lower() for n in names):
            raise FileError("This Word file contains macros, so it is refused.")
        ctypes = read_zip_member(zf, "[Content_Types].xml").lower()
        if b"macroenabled" in ctypes or b"vbaproject" in ctypes:
            raise FileError("This is a macro-enabled Word file, so it is refused.")
    finally:
        zf.close()


def json_depth(text):
    depth = best = 0
    in_str = esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
            best = max(best, depth)
        elif ch in "]}":
            depth -= 1
    return best


def check_json(text):
    # depth is measured BEFORE parsing so a deeply nested file cannot hurt the parser
    if json_depth(text) > MAX_JSON_DEPTH:
        raise FileError("This JSON is nested too deeply.")
    try:
        json.loads(text)
    except (ValueError, RecursionError):
        raise FileError("This file is not valid JSON.")


def detect(name, data):
    """Return the detected type for an upload, or raise a friendly FileError. Extension and content must agree."""
    ext = extension_of(name)
    kind = EXTENSIONS.get(ext)
    if kind is None:
        raise refuse_extension(ext)
    if not data or not data.strip():
        raise FileError("This file is empty.")
    if kind == "docx":
        if not data.startswith(b"PK\x03\x04"):
            raise FileError("The file ends in .docx but its content is not a Word document.")
        check_docx(data)
        return kind
    for magic, what in MAGIC:
        if data.startswith(magic):
            raise FileError(f"The file ends in .{ext} but its content looks like {what}, so it was refused.")
    text = decode_text(data)
    if not text.strip():
        raise FileError("This file is empty.")
    if kind == "json":
        check_json(text)
    elif kind == "html" and not re.search(r"<[a-zA-Z!/]", text):
        raise FileError("The file ends in .html but contains no HTML tags.")
    return kind
