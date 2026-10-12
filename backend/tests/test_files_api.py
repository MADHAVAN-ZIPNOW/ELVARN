"""File feature, HTTP level: upload/convert/download endpoints, chat integration, streaming, limits, logs, and UI string checks."""
import io
import json
import logging
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
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


def upload(client, name="notes.txt", data=b"Hello world\n\nSecond paragraph", **headers):
    h = {"X-File-Name": quote(name, safe="")}
    h.update(headers)
    return client.post("/api/files", content=data, headers=h)


def test_upload_success_shape(make_client):
    client, _, _ = make_client()
    r = upload(client)
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"file_id", "name", "size", "detected_type", "supported_targets"}
    assert body["name"] == "notes.txt" and body["detected_type"] == "txt" and body["size"] == 29
    assert "pdf" in body["supported_targets"] and len(body["file_id"]) >= 32


def test_upload_too_large_streamed_and_content_length(make_client):
    client, _, _ = make_client(max_upload_bytes=2048)
    big = b"a" * 5000
    r = client.post("/api/files", content=big, headers={"X-File-Name": "a.txt"})
    assert r.status_code == 413 and "too large" in r.json()["detail"]
    # chunked upload with no Content-Length is stopped by the streaming limit
    r = client.post("/api/files", content=(b"a" * 1000 for _ in range(10)), headers={"X-File-Name": "a.txt"})
    assert r.status_code == 413
    assert upload(client, data=b"a" * 2048).status_code == 200


@pytest.mark.parametrize("name,data,word", [
    ("a.exe", b"MZ\x90", "never accepted"), ("a.zip", b"PK\x03\x04", "never accepted"), ("a.docm", b"PK", "never accepted"),
    ("a.xlsx", b"hello", "not supported"), ("a.txt", b"MZ\x90\x00", "content looks like"), ("a.txt", b"", "empty"),
])
def test_upload_rejections(make_client, name, data, word):
    client, _, _ = make_client()
    r = upload(client, name, data)
    assert r.status_code in (400, 415) and word in r.json()["detail"]


def test_upload_filename_is_sanitised_and_not_used_on_disk(make_client):
    client, _, cfg = make_client()
    r = upload(client, "../../../etc/CON.txt", b"hi there")
    assert r.status_code == 200 and r.json()["name"] == "_CON.txt"
    r2 = upload(client, "a" * 3000 + ".txt", b"hi there")
    assert len(r2.json()["name"]) <= 100
    for p in cfg.root.iterdir():
        assert re.fullmatch(r"[0-9a-f]{32}\.bin", p.name)
    assert not (cfg.root.parent / "etc").exists()


def test_upload_unicode_name_roundtrip(make_client):
    client, _, _ = make_client()
    r = upload(client, "résumé 你好.txt", b"hello there")
    assert r.json()["name"] == "résumé 你好.txt"


def test_convert_and_download_headers(make_client):
    client, _, _ = make_client()
    fid = upload(client, "résumé 你.txt", b"Hello world").json()["file_id"]
    conv = client.post(f"/api/files/{fid}/convert", json={"target": "pdf"})
    assert conv.status_code == 200
    info = conv.json()
    assert info["format"] == "pdf" and info["name"] == "résumé 你.pdf" and info["size"] > 100
    d = client.get(f"/api/files/{info['download_id']}/download")
    assert d.status_code == 200 and d.content.startswith(b"%PDF")
    assert d.headers["content-type"] == "application/pdf"
    assert d.headers["x-content-type-options"] == "nosniff"
    assert d.headers["cache-control"] == "no-store"
    cd = d.headers["content-disposition"]
    assert cd.startswith("attachment;") and "filename*=UTF-8''r%C3%A9sum%C3%A9%20%E4%BD%A0.pdf" in cd
    assert 'filename="r_sum_ _.pdf"' in cd or "filename=\"" in cd
    assert "\n" not in cd and "\r" not in cd


@pytest.mark.parametrize("target,mime", [
    ("html", "text/html; charset=utf-8"), ("md", "text/markdown; charset=utf-8"), ("csv", "text/csv; charset=utf-8"),
    ("json", "application/json"), ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
])
def test_download_content_type_comes_from_our_target(make_client, target, mime):
    client, _, _ = make_client()
    fid = upload(client, "x.txt", b"hello <b>there</b>").json()["file_id"]
    did = client.post(f"/api/files/{fid}/convert", json={"target": target}).json()["download_id"]
    d = client.get(f"/api/files/{did}/download")
    assert d.headers["content-type"] == mime
    assert d.headers["content-disposition"].startswith("attachment")
    assert "default-src 'none'" in d.headers["content-security-policy"]
    if target == "docx":
        assert zipfile.ZipFile(io.BytesIO(d.content)).testzip() is None


def test_unsupported_pair_unknown_target_and_bad_ids(make_client):
    client, _, _ = make_client()
    fid = upload(client, "a.md", b"# hi").json()["file_id"]
    r = client.post(f"/api/files/{fid}/convert", json={"target": "csv"})
    assert r.status_code == 415 and "cannot convert" in r.json()["detail"]
    assert client.post(f"/api/files/{fid}/convert", json={"target": "exe"}).status_code == 415
    assert client.post(f"/api/files/{fid}/convert", json={"target": "md"}).status_code == 400   # same format
    assert client.post("/api/files/" + "A" * 32 + "/convert", json={"target": "pdf"}).status_code == 404
    assert client.get("/api/files/" + "A" * 32 + "/download").status_code == 404
    assert client.get("/api/files/short/download").status_code == 404
    assert client.get("/api/files/..%2f..%2fetc%2fpasswd/download").status_code == 404
    # an UPLOAD id can never be downloaded back (uploaded HTML is never served)
    assert client.get(f"/api/files/{fid}/download").status_code == 404


def test_delete_upload(make_client):
    client, _, cfg = make_client()
    fid = upload(client).json()["file_id"]
    assert client.delete(f"/api/files/{fid}").json() == {"deleted": True}
    assert client.delete(f"/api/files/{fid}").json() == {"deleted": False}
    assert client.post(f"/api/files/{fid}/convert", json={"target": "pdf"}).status_code == 404
    assert list(cfg.root.iterdir()) == []


def test_expiry_returns_404(make_client):
    client, app, _ = make_client(ttl_seconds=30)
    fid = upload(client).json()["file_id"]
    did = client.post(f"/api/files/{fid}/convert", json={"target": "pdf"}).json()["download_id"]
    store = app.state.files.store
    for e in store.items.values():
        e.expires = 0          # as if the time had passed
    assert client.get(f"/api/files/{did}/download").status_code == 404
    assert client.post(f"/api/files/{fid}/convert", json={"target": "pdf"}).status_code == 404


def test_per_visitor_cap_and_store_cap_over_http(make_client):
    client, _, _ = make_client(max_per_visitor=2)
    assert upload(client).status_code == 200 and upload(client).status_code == 200
    r = upload(client)
    assert r.status_code == 429 and "too many files" in r.json()["detail"].lower()
    client2, _, _ = make_client(max_store_bytes=1024 * 3, max_upload_bytes=2048)
    assert upload(client2, data=b"a" * 2000).status_code == 200
    assert upload(client2, data=b"a" * 2000).status_code == 503


def test_rate_limit_applies_to_file_endpoints(make_client):
    client, _, _ = make_client(public=True, rate=3)
    codes = [upload(client).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    assert client.get("/api/health").status_code == 200       # unrelated endpoints are not limited


def test_download_is_rate_limited_too(make_client):
    client, _, _ = make_client(public=True, rate=4)
    fid = upload(client).json()["file_id"]                                         # 1
    did = client.post(f"/api/files/{fid}/convert", json={"target": "pdf"}).json()["download_id"]   # 2
    assert client.get(f"/api/files/{did}/download").status_code == 200              # 3
    assert client.get(f"/api/files/{did}/download").status_code == 200              # 4
    assert client.get(f"/api/files/{did}/download").status_code == 429


# ---------- chat integration ----------
def boom(*a, **k):
    raise AssertionError("the model must not be called for a file turn")


def test_chat_conversion_skips_the_model(make_client, monkeypatch):
    client, app, _ = make_client()
    fid = upload(client, "notes.txt", b"Hello world").json()["file_id"]
    monkeypatch.setattr(app.state.service, "generate", boom)
    monkeypatch.setattr(app.state.service, "generate_stream", boom)
    r = client.post("/api/chat", json={"text": "convert this to pdf", "file_ids": [fid]})
    assert r.status_code == 200
    body = r.json()
    assert body["text"].startswith("Converted notes.txt to PDF (") and body["file_action"] == "converted"
    assert body["model_used"] is False and len(body["files"]) == 1
    f = body["files"][0]
    assert f["name"] == "notes.pdf" and f["format"] == "pdf" and f["download_url"] == f"/api/files/{f['download_id']}/download"
    assert client.get(f["download_url"]).content.startswith(b"%PDF")


def test_chat_stream_emits_file_event_and_done(make_client, monkeypatch):
    client, app, _ = make_client()
    fid = upload(client, "a.csv", b"x,y\n1,2").json()["file_id"]
    monkeypatch.setattr(app.state.service, "generate_stream", boom)
    r = client.post("/api/chat/stream", json={"text": "export it as json", "file_ids": [fid]})
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert [e["event"] for e in events] == ["file", "done"]
    assert events[0]["files"][0]["format"] == "json"
    assert events[1]["text"].startswith("Converted a.csv to JSON") and events[1]["files"] == events[0]["files"]


def test_chat_stream_honest_refusal_and_no_file(make_client, monkeypatch):
    client, app, _ = make_client()
    fid = upload(client).json()["file_id"]
    monkeypatch.setattr(app.state.service, "generate_stream", boom)
    r = client.post("/api/chat/stream", json={"text": "summarise this file", "file_ids": [fid]})
    done = [json.loads(line) for line in r.text.splitlines()][-1]
    assert done["event"] == "done" and "only change a file's format" in done["text"] and done["files"] == []
    r = client.post("/api/chat", json={"text": "convert this to pdf", "file_ids": ["x" * 32]})
    assert r.json()["file_action"] == "expired"


def test_non_conversion_goes_through_normal_chat(make_client):
    client, app, _ = make_client()
    fid = upload(client).json()["file_id"]
    r = client.post("/api/chat", json={"text": "hello there", "file_ids": [fid], "max_new_tokens": 5, "web_search": "off"})
    assert r.status_code == 200
    body = r.json()
    assert "generated_tokens" in body and "files" not in body and "file_action" not in body
    s = client.post("/api/chat/stream", json={"text": "hello there", "file_ids": [fid], "max_new_tokens": 5, "web_search": "off"})
    kinds = [json.loads(line)["event"] for line in s.text.splitlines() if line.strip()]
    assert kinds[0] == "start" and kinds[-1] == "done" and "file" not in kinds


def test_old_requests_without_file_ids_are_unchanged(make_client):
    client, _, _ = make_client()
    r = client.post("/api/chat", json={"text": "convert this to pdf", "max_new_tokens": 5, "web_search": "off"})
    assert r.status_code == 200 and "generated_tokens" in r.json() and "file_action" not in r.json()
    assert client.post("/api/chat", json={"text": "hi", "file_ids": ["a"] * 6}).status_code == 422


def test_chat_with_file_ids_and_no_attachment_wording_is_friendly(make_client, monkeypatch):
    client, app, _ = make_client()
    fid = upload(client).json()["file_id"]
    monkeypatch.setattr(app.state.service, "generate", boom)
    r = client.post("/api/chat", json={"text": "convert this to xlsx", "file_ids": [fid]})
    assert r.json()["file_action"] == "unknown_target" and "cannot make .xlsx" in r.json()["text"]


# ---------- logs ----------
def test_no_file_contents_or_names_in_logs(make_client, caplog):
    client, _, _ = make_client()
    secret = b"TOP-SECRET-CONTENT-12345"
    with caplog.at_level(logging.DEBUG):
        fid = upload(client, "my-secret-name.txt", secret).json()["file_id"]
        did = client.post(f"/api/files/{fid}/convert", json={"target": "html"}).json()["download_id"]
        client.get(f"/api/files/{did}/download")
        client.post("/api/files/" + "A" * 32 + "/convert", json={"target": "pdf"})
    text = " | ".join(r.getMessage() for r in caplog.records if r.name.startswith(("elvarn", "uvicorn", "api", "files")))
    assert "TOP-SECRET" not in text and "my-secret-name" not in text and fid not in text and did not in text
    assert "upload ok" in text and "convert ok" in text


# ---------- frontend (string level; the UI was NOT run in a browser by these tests) ----------
@pytest.fixture(scope="module")
def page():
    return FRONTEND.read_text(encoding="utf-8")


def test_ui_has_attach_controls_and_labels(page):
    for needle in ('id="attachBtn"', 'aria-label="Attach a file"', 'id="fileInput"', 'accept=".txt,.md,.markdown,.csv,.json,.html,.htm,.docx"',
                   'id="fileBar"', "function downloadCard", "function uploadFile", "function setupFiles", "dragover", "Download",
                   "/api/files", "file_ids", "aria-label", "srStatus"):
        assert needle in page, needle
    assert "Remove attached file" in page and "Cancel upload" in page and "has expired" in page


def test_ui_file_code_never_uses_innerhtml_for_names(page):
    start = page.index("/* ---------- file upload + conversion ---------- */")
    end = page.index("/* ---------- speech: text to speech ---------- */")
    block = page[start:end]
    assert "innerHTML" not in block and "insertAdjacentHTML" not in block and "outerHTML" not in block
    assert "textContent" in block and 'el("div","dlname"' in block     # el() sets textContent


def test_ui_inline_script_syntax_and_file_code_present(page, tmp_path):
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    main = max(scripts, key=len)
    assert "uploadFile" in main
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = tmp_path / "inline.js"
    js.write_text(main, encoding="utf-8")
    out = subprocess.run([node, "--check", str(js)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
