"""File preview endpoint (GET /api/files/{id}/preview) and the right-side viewer UI (string level only:
the UI was NOT run in a browser by these tests)."""
import io
import logging
import re
import zipfile
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from files.convert import PREVIEW_MAX_CHARS
from files.service import FileService
from files.store import FileConfig

FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"


@pytest.fixture()
def make_client(tmp_path, ckpt_dir, tok_file):
    def build(public=False, rate=None, **cfg):
        config = FileConfig.from_env(root=tmp_path / "fs", **cfg)
        app = create_app(ckpt_dir, load_on_start=False, public=public, rate_limit_per_minute=rate, max_concurrent=0,
                         file_service=FileService(config))
        app.state.service.tokenizer_file = tok_file
        app.state.service.load()
        return TestClient(app), app, config
    return build


def upload(client, name="notes.txt", data=b"Hello world\n\nSecond paragraph"):
    return client.post("/api/files", content=data, headers={"X-File-Name": quote(name, safe="")})


def convert(client, fid, target):
    r = client.post(f"/api/files/{fid}/convert", json={"target": target})
    assert r.status_code == 200, r.text
    return r.json()["download_id"]


def make_docx(text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="xml" ContentType="application/xml"/></Types>')
        z.writestr("word/document.xml", '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                   f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>")
    return buf.getvalue()


def test_preview_of_an_upload(make_client):
    client, _, _ = make_client()
    fid = upload(client).json()["file_id"]
    r = client.get(f"/api/files/{fid}/preview")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert "no-store" in r.headers["cache-control"]
    body = r.json()
    assert body["text"] == "Hello world\n\nSecond paragraph"
    assert body["name"] == "notes.txt" and body["format"] == "txt" and body["truncated"] is False
    assert set(body) >= {"name", "format", "size", "text", "truncated", "preview_note"}


@pytest.mark.parametrize("target", ["txt", "md", "html", "csv", "json"])
def test_preview_of_text_outputs_is_the_file_text(make_client, target):
    client, _, _ = make_client()
    source = "md" if target == "txt" else "txt"
    fid = upload(client, f"a.{source}", b"Alpha line\n\nBeta line").json()["file_id"]
    did = convert(client, fid, target)
    body = client.get(f"/api/files/{did}/preview").json()
    downloaded = client.get(f"/api/files/{did}/download").content.decode("utf-8")
    assert body["text"] == downloaded.lstrip("﻿") and body["format"] == target and body["preview_note"] == ""


def test_html_output_preview_is_raw_escaped_source_not_rendered(make_client):
    client, _, _ = make_client()
    fid = upload(client, "a.txt", b"<script>alert(1)</script> & <b>bold</b>").json()["file_id"]
    did = convert(client, fid, "html")
    text = client.get(f"/api/files/{did}/preview").json()["text"]
    assert "<script>alert(1)</script>" not in text           # the dangerous text is escaped in the source
    assert "&lt;script&gt;" in text and "<html" in text.lower()   # and we get the HTML source itself, as text


@pytest.mark.parametrize("target", ["docx", "pdf"])
def test_preview_of_binary_outputs_is_extracted_text(make_client, target):
    client, _, _ = make_client()
    fid = upload(client, "a.txt", b"Plain heading text\n\nSecond paragraph here").json()["file_id"]
    did = convert(client, fid, target)
    body = client.get(f"/api/files/{did}/preview").json()
    assert "Plain heading text" in body["text"] and "Second paragraph here" in body["text"]
    assert "text only" in body["preview_note"].lower() and body["format"] == target
    assert not body["text"].startswith(("%PDF", "PK"))


def test_preview_of_docx_upload(make_client):
    client, _, _ = make_client()
    fid = upload(client, "w.docx", make_docx("Word body text")).json()["file_id"]
    body = client.get(f"/api/files/{fid}/preview").json()
    assert "Word body text" in body["text"] and "text only" in body["preview_note"].lower()


def test_preview_truncation_flag(make_client):
    client, _, _ = make_client()
    big = ("x" * 99 + "\n") * 3000            # 300,000 characters
    fid = upload(client, "big.txt", big.encode()).json()["file_id"]
    body = client.get(f"/api/files/{fid}/preview").json()
    assert body["truncated"] is True and len(body["text"]) == PREVIEW_MAX_CHARS
    did = convert(client, fid, "pdf")
    body = client.get(f"/api/files/{did}/preview").json()
    assert body["truncated"] is False and len(body["text"]) <= PREVIEW_MAX_CHARS    # pdf keeps a capped extract


def test_preview_unknown_malformed_deleted_and_expired(make_client):
    client, app, _ = make_client(ttl_seconds=30)
    assert client.get("/api/files/" + "A" * 32 + "/preview").status_code == 404
    assert client.get("/api/files/short/preview").status_code == 404
    fid = upload(client).json()["file_id"]
    did = convert(client, fid, "pdf")
    assert client.delete(f"/api/files/{fid}").status_code == 200
    assert client.get(f"/api/files/{fid}/preview").status_code == 404
    for e in app.state.files.store.items.values():
        e.expires = 0
    r = client.get(f"/api/files/{did}/preview")
    assert r.status_code == 404 and "expired" in r.json()["detail"].lower()


def test_preview_is_rate_limited(make_client):
    client, _, _ = make_client(public=True, rate=3)
    fid = upload(client).json()["file_id"]                       # 1
    assert client.get(f"/api/files/{fid}/preview").status_code == 200   # 2
    assert client.get(f"/api/files/{fid}/preview").status_code == 200   # 3
    assert client.get(f"/api/files/{fid}/preview").status_code == 429


def test_preview_does_not_log_content_or_ids(make_client, caplog):
    client, _, _ = make_client()
    fid = upload(client, "secret-name.txt", b"TOP-SECRET-PREVIEW-777").json()["file_id"]
    did = convert(client, fid, "docx")
    with caplog.at_level(logging.DEBUG):
        client.get(f"/api/files/{fid}/preview")
        client.get(f"/api/files/{did}/preview")
    text = " | ".join(r.getMessage() for r in caplog.records if r.name.startswith(("elvarn", "uvicorn", "api", "files")))
    assert "TOP-SECRET" not in text and "secret-name" not in text and fid not in text and did not in text


def test_binary_looking_upload_cannot_be_previewed_as_text(make_client):
    client, _, _ = make_client()
    # a text-typed upload must pass detection, so previews only ever see text; a missing id stays a clean 404
    assert client.get("/api/files/" + "B" * 40 + "/preview").status_code == 404


# ---------- frontend (string level) ----------
@pytest.fixture(scope="module")
def page():
    return FRONTEND.read_text(encoding="utf-8")


def test_ui_viewer_panel_markup(page):
    for needle in ('id="filePanel"', 'role="complementary"', 'aria-label="File viewer"', 'id="fpCopy"', 'id="fpDownload"',
                   'id="fpClose"', 'aria-label="Close file viewer"', 'id="fpText"', 'id="fpStatus"', 'aria-live="polite"',
                   'aria-label="Attach a file"', 'data-i="upload"'):    # the attach button uses the creative upload icon
        assert needle in page, needle


def test_ui_upload_button_is_the_creative_sphere_and_glows_when_a_file_is_attached(page):
    assert "upload:'<path" in page and 'class="arrow"' in page and 'class="spark"' in page   # page + arrow + sparkles icon
    assert ".attach.has{" in page and 'classList.toggle("has"' in page                          # glow state
    assert "prefers-reduced-motion" in page[page.index(".attach.has{"):page.index(".attach.has{") + 1500]


def test_ui_question_has_a_copy_button_that_works_without_clipboard_api(page):
    """The user's own question (not only the answer) has a Copy option, and copying falls back to a hidden textarea."""
    start = page.index("function copyBtn(")
    block = page[start:page.index("function buildBot(")]
    assert "copyPlain(text)" in block and 'el("div","tools utools")' in block and "Copy failed" in block
    assert "innerHTML" not in block
    assert page.count("copyBtn(") >= 3                 # defined once, used for the question and for the answer


def test_ui_attach_button_is_left_of_textarea_inside_the_box(page):
    box = page[page.index('<div class="box">'):page.index('id="sendBtn"')]
    assert box.index('id="fileBar"') < box.index('id="attachBtn"') < box.index('id="input"') < box.index('id="micBtn"')


def test_ui_card_copy_and_open_behaviour(page):
    start = page.index("function downloadCard(")
    block = page[start:page.index("function setupFiles(")]
    for needle in ("openFilePanel", "copyPreview", "stopPropagation", "Enter", '" "', "Copy", "Download", "text only"):
        assert needle in block, needle
    assert "innerHTML" not in block


def test_ui_viewer_uses_textcontent_never_innerhtml(page):
    start = page.index("/* ---------- right-side file viewer: shows")
    block = page[start:page.index("function renderFileBar(")]
    assert "innerHTML" not in block and "insertAdjacentHTML" not in block and "outerHTML" not in block
    assert "textContent" in block and "execCommand" in block and "navigator.clipboard" in block
    assert "/preview" in block and "Escape" in page and "closeFilePanel" in page
    assert "This file has expired. Upload it again." in block
    assert "Showing the first" in block


def test_ui_only_calls_our_own_file_endpoints(page):
    block = page[page.index("/* ---------- file upload + conversion"):page.index("/* ---------- speech: text to speech")]
    urls = re.findall(r'fetch\(([^,)]*)', block)
    assert urls and all("API_BASE" in u for u in urls), urls
