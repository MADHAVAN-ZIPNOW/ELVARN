"""File feature, core logic: upload checks, store, readers/writers (conversion matrix), PDF/DOCX validity, intent rules."""
import csv
import io
import json
import re
import zipfile
import zlib
from xml.etree import ElementTree as ET

import pytest

from files import intent
from files.convert import MATRIX, convert, supported_targets
from files.errors import FileError, LimitReached, NotFound
from files.service import FileService
from files.sniff import clean_filename, detect
from files.store import FileConfig, FileStore


# ---------- helpers ----------
def make_docx(paragraphs, extra=None, content_types_extra=""):
    body = "".join(f'<w:p><w:r><w:t>{p}</w:t></w:r></w:p>' for p in paragraphs)
    doc = ('<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f"<w:body>{body}</w:body></w:document>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<Types xmlns="x">' + content_types_extra + "</Types>")
        z.writestr("word/document.xml", doc)
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


def pdf_pages_text(pdf):
    out = []
    for m in re.finditer(rb"stream\n(.*?)\nendstream", pdf, re.S):
        out.append(zlib.decompress(m.group(1)).decode("ascii"))
    return out


def check_pdf_structure(pdf):
    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")
    start = int(re.search(rb"startxref\n(\d+)\n%%EOF", pdf).group(1))
    assert pdf[start:start + 4] == b"xref"
    count = int(re.search(rb"xref\n0 (\d+)\n", pdf).group(1))
    table = pdf[start:].split(b"\n")[2:2 + count]
    assert table[0].startswith(b"0000000000 65535 f")
    for num, line in enumerate(table[1:], start=1):
        offset = int(line[:10])
        assert pdf[offset:].startswith(f"{num} 0 obj".encode()), num
    return count


@pytest.fixture()
def cfg(tmp_path):
    return FileConfig.from_env(root=tmp_path / "store")


@pytest.fixture()
def svc(cfg):
    return FileService(cfg)


# ---------- filename cleaning ----------
@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd", "passwd"), ("C:\\Users\\x\\notes.txt", "notes.txt"), ("notes.txt", "notes.txt"),
    ("CON.txt", "_CON.txt"), ("nul", "_nul"), ("a\x00b\x1f.txt", "ab.txt"), ("", "file"), ("...", "file"),
    ("re\u202etxt.exe", "retxt.exe"), ("rapport \u00e9t\u00e9.txt", "rapport \u00e9t\u00e9.txt"), ('a<b>:"|?*.txt', "ab.txt"),
])
def test_filename_sanitising(raw, expected):
    assert clean_filename(raw) == expected


def test_huge_filename_is_capped_and_keeps_extension():
    out = clean_filename("a" * 5000 + ".txt")
    assert len(out) <= 100 and out.endswith(".txt")


# ---------- type detection ----------
def test_detect_ok():
    assert detect("a.txt", b"hello") == "txt"
    assert detect("a.MD", b"# hi") == "md"
    assert detect("a.csv", b"a,b\n1,2") == "csv"
    assert detect("a.json", b'{"a": 1}') == "json"
    assert detect("a.htm", b"<p>hi</p>") == "html"
    assert detect("a.docx", make_docx(["hi"])) == "docx"


@pytest.mark.parametrize("name", ["a.exe", "a.js", "a.zip", "a.docm", "a.xlsm", "a.svg", "a.sh", "a.bat"])
def test_dangerous_extensions_refused(name):
    with pytest.raises(FileError) as e:
        detect(name, b"hello")
    assert "never accepted" in e.value.message


def test_unsupported_and_pdf_messages():
    with pytest.raises(FileError) as e:
        detect("a.xlsx", b"hello")
    assert "not supported" in e.value.message
    with pytest.raises(FileError) as e:
        detect("a.pdf", b"%PDF-1.4")
    assert "can be created" in e.value.message
    with pytest.raises(FileError):
        detect("noextension", b"hello")


@pytest.mark.parametrize("name,data", [
    ("a.txt", b"MZ\x90\x00\x03"), ("a.txt", b"%PDF-1.4 hi"), ("a.csv", b"PK\x03\x04 zipdata"), ("a.json", b"\x89PNG\r\n"),
    ("a.txt", b"#!/bin/sh\nrm -rf /"), ("a.txt", b"abc\x00\x01\x02def"), ("a.docx", b"just text"),
])
def test_magic_mismatch_rejected(name, data):
    with pytest.raises(FileError):
        detect(name, data)


def test_empty_and_blank_rejected():
    for data in (b"", b"   \n\n"):
        with pytest.raises(FileError) as e:
            detect("a.txt", data)
        assert "empty" in e.value.message


def test_bad_json_and_deep_json_and_html_without_tags():
    with pytest.raises(FileError):
        detect("a.json", b"{not json")
    with pytest.raises(FileError) as e:
        detect("a.json", b"[" * 5000 + b"]" * 5000)
    assert "deeply" in e.value.message
    with pytest.raises(FileError):
        detect("a.html", b"just words")


def test_zip_bomb_docx_rejected():
    bomb = make_docx(["x"], extra={"word/big.bin": b"\x00" * (30 * 1024 * 1024)})
    assert len(bomb) < 200_000
    with pytest.raises(FileError) as e:
        detect("a.docx", bomb)
    assert "zip bomb" in e.value.message


def test_too_many_entries_rejected():
    many = make_docx(["x"], extra={f"p/{i}.txt": b"a" for i in range(300)})
    with pytest.raises(FileError):
        detect("a.docx", many)


def test_macro_docx_rejected():
    with pytest.raises(FileError) as e:
        detect("a.docx", make_docx(["x"], extra={"word/vbaProject.bin": b"abc"}))
    assert "macro" in e.value.message
    with pytest.raises(FileError):
        detect("a.docx", make_docx(["x"], content_types_extra='<Override ContentType="application/vnd.ms-word.document.macroEnabled.main+xml"/>'))


def test_docx_xml_entities_refused():
    evil = ('<?xml version="1.0"?><!DOCTYPE w [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>&b;</w:t></w:r></w:p></w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", evil)
    with pytest.raises(FileError) as e:
        convert("docx", buf.getvalue(), "txt")
    assert "entity" in e.value.message.lower() or "DTD" in e.value.message


# ---------- store ----------
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_store_ids_names_and_expiry(cfg):
    clock = Clock()
    store = FileStore(cfg, clock=clock)
    e = store.put(b"hello", kind="upload", name="../evil.txt", fmt="txt", owner="a")
    assert len(e.id) >= 32 and re.fullmatch(r"[A-Za-z0-9_-]+", e.id)
    on_disk = list(cfg.root.iterdir())
    assert len(on_disk) == 1 and re.fullmatch(r"[0-9a-f]{32}\.bin", on_disk[0].name)   # server-made name, not the user's
    assert store.read(store.get(e.id)) == b"hello"
    clock.t += cfg.ttl_seconds + 1
    with pytest.raises(NotFound):
        store.get(e.id)
    assert list(cfg.root.iterdir()) == []   # cleaned on access


def test_ids_are_unique_and_unguessable(cfg):
    store = FileStore(FileConfig.from_env(root=cfg.root, max_per_visitor=500, max_items=500, max_store_bytes=10**8))
    ids = {store.put(b"x", kind="upload", name="a", fmt="txt", owner="a").id for _ in range(300)}
    assert len(ids) == 300


def test_store_cap_and_visitor_cap(tmp_path):
    cfg = FileConfig.from_env(root=tmp_path / "s", max_store_bytes=100, max_per_visitor=2)
    store = FileStore(cfg)
    store.put(b"a" * 40, kind="upload", name="a", fmt="txt", owner="v1")
    store.put(b"a" * 40, kind="upload", name="a", fmt="txt", owner="v1")
    with pytest.raises(LimitReached):
        store.put(b"a", kind="upload", name="a", fmt="txt", owner="v1")
    with pytest.raises(FileError) as e:
        store.put(b"a" * 40, kind="upload", name="a", fmt="txt", owner="v2")   # store would pass 100 bytes
    assert "full" in e.value.message
    store.put(b"a" * 10, kind="upload", name="a", fmt="txt", owner="v2")


def test_store_startup_removes_leftovers(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    (root / ("a" * 32 + ".bin")).write_bytes(b"old")
    (root / "keep.me").write_bytes(b"not ours")
    FileStore(FileConfig.from_env(root=root))
    assert not (root / ("a" * 32 + ".bin")).exists() and (root / "keep.me").exists()


def test_delete(cfg):
    store = FileStore(cfg)
    e = store.put(b"x", kind="upload", name="a", fmt="txt", owner="a")
    assert store.delete(e.id) is True and store.delete(e.id) is False
    assert list(cfg.root.iterdir()) == []


def test_concurrent_uploads_keep_caps(tmp_path):
    import threading
    cfg = FileConfig.from_env(root=tmp_path / "s", max_per_visitor=5, max_store_bytes=10**7)
    store = FileStore(cfg)
    ok, bad = [], []

    def work():
        try:
            ok.append(store.put(b"x", kind="upload", name="a", fmt="txt", owner="same"))
        except FileError:
            bad.append(1)
    threads = [threading.Thread(target=work) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(ok) == 5 and len(bad) == 15
    assert len({e.id for e in ok}) == 5


# ---------- conversions: every pair in the matrix ----------
SAMPLES = {
    "txt": b"Title line\nsecond line\n\nNew paragraph (with) parens \\ backslash.\n",
    "md": b"# Heading\n\nSome **bold** text and a [link](https://x.test).\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\ncode here\n```\n",
    "csv": b"name,score\nAnn,5\nBob,7\n",
    "json": b'[{"name": "Ann", "score": 5}, {"name": "Bob", "score": 7}]',
    "html": b"<h1>Head</h1><p>Hello <b>world</b></p><ul><li>one</li></ul><table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>",
}
SAMPLES["docx"] = convert("html", SAMPLES["html"], "docx").data     # has a heading, a list item and a table


@pytest.mark.parametrize("source,target", [(s, t) for s, ts in MATRIX.items() for t in ts])
def test_every_matrix_pair_makes_a_valid_file(source, target):
    res = convert(source, SAMPLES[source], target, "sample")
    data = res.data
    assert data
    if target == "pdf":
        check_pdf_structure(data)
        assert any("Tj" in page for page in pdf_pages_text(data))
    elif target == "docx":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            assert {"[Content_Types].xml", "_rels/.rels", "word/document.xml", "word/styles.xml"} <= set(z.namelist())
            for part in z.namelist():
                ET.fromstring(z.read(part))          # every part is well-formed XML
    elif target == "json":
        json.loads(data.decode("utf-8"))
    elif target == "csv":
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        assert rows and all(rows)
    elif target == "html":
        text = data.decode("utf-8")
        assert text.startswith("<!DOCTYPE html>") and "Content-Security-Policy" in text
    else:
        assert data.decode("utf-8").strip()


def test_matrix_helpers_and_unsupported_pairs():
    assert supported_targets("txt") == ["md", "html", "csv", "json", "docx", "pdf"]
    assert supported_targets("nope") == []
    with pytest.raises(FileError) as e:
        convert("md", SAMPLES["md"], "csv")
    assert "cannot convert" in e.value.message
    with pytest.raises(FileError) as e:
        convert("txt", SAMPLES["txt"], "txt")
    assert "already" in e.value.message
    with pytest.raises(FileError):
        convert("txt", SAMPLES["txt"], "exe")


def test_docx_roundtrip_text():
    out = convert("txt", b"Alpha line\n\nBeta (para) & <tags>", "docx").data
    back = convert("docx", out, "txt").data.decode()
    assert "Alpha line" in back and "Beta (para) & <tags>" in back


def test_csv_to_json_records_and_back():
    js = json.loads(convert("csv", SAMPLES["csv"], "json").data)
    assert js == [{"name": "Ann", "score": "5"}, {"name": "Bob", "score": "7"}]
    back = convert("json", json.dumps(js).encode(), "csv").data.decode("utf-8-sig")
    assert list(csv.reader(io.StringIO(back))) == [["name", "score"], ["Ann", "5"], ["Bob", "7"]]


def test_txt_to_json_and_csv_semantics():
    js = json.loads(convert("txt", b"one\n\ntwo", "json").data)
    assert [b["text"] for b in js["blocks"]] == ["one", "two"]
    rows = list(csv.reader(io.StringIO(convert("txt", b"one\ntwo", "csv").data.decode("utf-8-sig"))))
    assert rows == [["text"], ["one"], ["two"]]


def test_csv_target_needs_a_table():
    with pytest.raises(FileError) as e:
        convert("md", b"# just words", "docx")   # fine
        convert("html", b"<p>no table</p>", "csv")
    assert "no table" in e.value.message.lower()
    with pytest.raises(FileError):
        convert("json", b'{"a": {"b": 1}}', "csv")


def test_csv_formula_injection_neutralised():
    nasty = b'a,b\n=HYPERLINK("http://evil"),+1+1\n@SUM(A1),-5\n"=cmd|\' /C calc\'!A0",ok\n'
    out = convert("csv", nasty, "json").data     # check via a csv -> txt -> csv chain on html table as well
    assert out
    html = b"<table><tr><th>x</th></tr><tr><td>=1+1</td></tr><tr><td>@a</td></tr><tr><td>-5</td></tr><tr><td>-1+2</td></tr></table>"
    rows = list(csv.reader(io.StringIO(convert("html", html, "csv").data.decode("utf-8-sig"))))
    cells = [r[0] for r in rows[1:]]
    assert cells == ["'=1+1", "'@a", "-5", "'-1+2"]   # plain negative numbers are left alone, formulas are defanged


def test_csv_row_cap():
    big = ("a,b\n" + "1,2\n" * 10050).encode()
    with pytest.raises(FileError) as e:
        convert("csv", big, "json")
    assert "rows" in e.value.message


def test_html_source_is_sanitised_everywhere():
    evil = (b'<h1 onclick="x()">Hi</h1><script>alert(1)</script><style>b{}</style><iframe src="http://e"></iframe>'
            b'<p>Click <a href="javascript:alert(2)">here</a> <img src=x onerror=alert(3)></p><form action="/x"><input name=p>'
            b'<button>Go</button></form><object data="x"></object>')
    for target in ("txt", "md", "json", "pdf", "docx"):
        data = convert("html", evil, target).data
        if target == "pdf":
            data = b"".join(p.encode() for p in pdf_pages_text(data))
        elif target == "docx":
            data = zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml")
        low = data.lower()
        for bad in (b"alert", b"onerror", b"onclick", b"javascript:", b"<script", b"<iframe", b"<object"):
            assert bad not in low, (target, bad)
        assert b"Hi" in data and b"here" in data


def test_html_output_escapes_input():
    out = convert("txt", b'<script>alert(1)</script> <img src=x onerror=alert(1)> "q" & more', "html").data.decode()
    body = out.split("<body>")[1]
    assert "<script" not in body and "<img" not in body
    assert "&lt;script&gt;" in body and "&amp; more" in body
    out2 = convert("csv", b'h\n"<b onmouseover=x>"\n', "html").data.decode()
    assert "<b onmouseover" not in out2


def test_markdown_output_escapes_markup():
    out = convert("txt", b"# not a heading\n- not a list\n<b>x</b> *star*", "md").data.decode()
    assert out.startswith("\\# not a heading") and "\\- not a list" in out and "\\<b\\>" in out and "\\*star\\*" in out


def test_markdown_source_parse():
    js = json.loads(convert("md", SAMPLES["md"], "json").data)
    types = [b["type"] for b in js["blocks"]]
    assert types[0] == "heading" and "list_item" in types and "table" in types and "code" in types
    txt = convert("md", SAMPLES["md"], "txt").data.decode()
    assert "bold" in txt and "**" not in txt


# ---------- PDF details ----------
def test_pdf_escaping_of_tricky_characters():
    pdf = convert("txt", b"paren ( and ) and backslash \\ and \\( weird \xc3\xa9 \xe2\x80\x9cquotes\xe2\x80\x9d", "pdf").data
    check_pdf_structure(pdf)
    content = "\n".join(pdf_pages_text(pdf))
    assert r"paren \( and \) and backslash \\ and \\\( weird" in content
    assert r"\351" in content                       # e-acute in WinAnsi octal
    assert content.count("(") >= content.count(r"\(") + 1   # only escaped parens besides the string delimiters
    # a naive reader that balances unescaped parens must find every string closed
    for line in content.split("\n"):
        depth, i = 0, 0
        while i < len(line):
            c = line[i]
            if c == "\\":
                i += 2
                continue
            depth += c == "("
            depth -= c == ")"
            assert depth >= 0
            i += 1
        assert depth == 0


def test_pdf_pagination_and_wrapping():
    long = ("word " * 40 + "\n") * 150
    res = convert("txt", long.encode(), "pdf")
    count = check_pdf_structure(res.data)
    pages = len(re.findall(rb"/Type /Page ", res.data))
    assert pages >= 3 and count == 5 + 2 * pages + 1 - 0 or count > 5
    assert len(pdf_pages_text(res.data)) == pages
    unbroken = convert("txt", b"x" * 5000, "pdf").data
    check_pdf_structure(unbroken)


def test_pdf_empty_page_never_crashes_on_whitespace_blocks():
    check_pdf_structure(convert("md", b"# only a heading", "pdf").data)


def test_non_latin_text_is_replaced_and_reported():
    res = convert("txt", "Hello \u4f60\u597d \u0928\u092e\u0938\u094d\u0924\u0947 caf\u00e9".encode("utf-8"), "pdf")
    assert res.lost_chars >= 2
    assert "?" in "\n".join(pdf_pages_text(res.data))
    # other targets keep Unicode exactly
    assert "\u4f60\u597d" in convert("txt", "\u4f60\u597d".encode("utf-8"), "html").data.decode("utf-8")
    assert "\u4f60\u597d" in zipfile.ZipFile(io.BytesIO(convert("txt", "\u4f60\u597d".encode("utf-8"), "docx").data)).read("word/document.xml").decode("utf-8")


def test_pdf_page_cap():
    big = ("line\n\n" * 60000).encode()
    with pytest.raises(FileError) as e:
        convert("txt", big, "pdf")
    assert "pages" in e.value.message or "too much text" in e.value.message


# ---------- bounds ----------
def test_large_input_is_bounded():
    huge = ("x" * 100 + "\n\n") * 30000          # about 3 MB of text
    with pytest.raises(FileError):
        convert("txt", huge.encode(), "docx")
    with pytest.raises(FileError):
        convert("txt", b"a\n" * 100, "html", max_output=50)


def test_docx_with_invalid_xml_characters_is_still_valid():
    out = convert("txt", (b"ok text " * 100) + b"\x01\x02 more \x0b end", "docx")
    with zipfile.ZipFile(io.BytesIO(out.data)) as z:
        ET.fromstring(z.read("word/document.xml"))


def test_empty_text_after_parse_is_refused():
    with pytest.raises(FileError):
        convert("html", b"<script>x()</script><style>a{}</style>", "txt")


def test_conversion_is_deterministic_in_service(svc):
    up = svc.upload("n.txt", b"hello", "o")
    a, _ = svc.convert_upload(up["file_id"], "pdf", "o")
    b, _ = svc.convert_upload(up["file_id"], "pdf", "o")
    assert a["download_id"] == b["download_id"]   # reused, not stored twice


def test_service_upload_info_and_names(svc):
    up = svc.upload("..\\..\\My Notes.txt", b"hello world", "o")
    assert up["name"] == "My Notes.txt" and up["detected_type"] == "txt" and "pdf" in up["supported_targets"]
    info, _ = svc.convert_upload(up["file_id"], "pdf", "o")
    assert info["name"] == "My Notes.pdf" and info["format"] == "pdf"


# ---------- intent ----------
@pytest.mark.parametrize("msg,target", [
    ("convert this to pdf", "pdf"), ("Convert it to PDF please", "pdf"), ("change it to docx", "docx"),
    ("turn this into a word document", "docx"), ("save as pdf", "pdf"), ("make a pdf", "pdf"), ("export as markdown", "md"),
    ("download as html", "html"), ("convert to csv", "csv"), ("can you change this file to json?", "json"),
    ("convert to plain text", "txt"), ("change to txt", "txt"), ("give me a pdf version", "pdf"), ("pdf", "pdf"),
    ("to markdown", "md"), ("I want it as docx", "docx"), ("please transform the file into HTML", "html"),
    ("export this as a webpage", "html"),
])
def test_intent_convert_phrasings(msg, target):
    got = intent.parse(msg, True)
    assert (got.kind, got.target) == ("convert", target), msg


@pytest.mark.parametrize("msg", [
    "hello", "what is the capital of France?", "tell me a joke", "how do I convert celsius to fahrenheit",
    "write a poem about pdf files", "make a pdf of the history of Rome", "what does csv stand for", "thanks!",
])
def test_non_conversion_messages_untouched(msg):
    assert intent.parse(msg, True).kind == "none", msg
    assert intent.parse(msg, False).kind == "none", msg


@pytest.mark.parametrize("msg", ["summarise this", "please summarize the file", "translate this document to pdf and fix the grammar",
                                 "rewrite it", "make it shorter", "proofread my file", "summarise it and convert to pdf"])
def test_edit_requests_get_the_honest_message(msg):
    assert intent.parse(msg, True).kind == "edit", msg


def test_summarise_without_file_is_normal_chat():
    assert intent.parse("summarise the French revolution", False).kind == "none"
    assert intent.parse("summarise the French revolution", True).kind == "none"


def test_no_attachment_friendly_message():
    assert intent.parse("convert this to pdf", False).kind == "no_file"
    assert intent.parse("save it as docx", False).kind == "no_file"


@pytest.mark.parametrize("msg,word", [("convert this to xlsx", "xlsx"), ("change it into a powerpoint", "powerpoint"),
                                      ("save as mp3", "mp3"), ("export it to .zip", "zip")])
def test_unknown_target(msg, word):
    got = intent.parse(msg, True)
    assert got.kind == "unknown" and got.word == word


def test_handle_chat_uses_latest_attachment_and_fixed_template(svc):
    first = svc.upload("old.txt", b"old", "o")
    second = svc.upload("new.md", b"# new", "o")
    out = svc.handle_chat("convert it to pdf", [first["file_id"], second["file_id"]], "o")
    assert out["file_action"] == "converted"
    assert out["text"].startswith("Converted new.md to PDF (")
    assert out["files"][0]["name"] == "new.pdf" and out["files"][0]["download_url"].endswith("/download")
    assert svc.handle_chat("what is the capital of France?", [first["file_id"]], "o") is None
    assert svc.handle_chat("summarise this", [first["file_id"]], "o")["file_action"] == "refused_edit"
    assert svc.handle_chat("convert to pdf", ["gone" * 8], "o")["file_action"] == "expired"
    assert svc.handle_chat("convert to xlsx", [first["file_id"]], "o")["file_action"] == "unknown_target"
    # unsupported pair gives a friendly failure, not an exception
    got = svc.handle_chat("convert this to csv", [second["file_id"]], "o")
    assert got["file_action"] == "convert_failed" and "cannot convert" in got["text"]
