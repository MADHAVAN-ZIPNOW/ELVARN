"""Download a chat as PDF / Word / text (POST /api/export)."""
import io
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from files.routes import install_file_routes
from files.service import FileService


@pytest.fixture()
def client():
    app = FastAPI()
    install_file_routes(app, FileService(), lambda request: "v1")
    return TestClient(app)


BODY = {"title": "K.B. Sampath appeal", "messages": [
    {"role": "user", "text": "What was K.B. Sampath's appeal?"},
    {"role": "bot", "text": "It was an appeal against the **Inspector of Police, CBI**.\n- point one\n- point two",
     "sources": [{"title": "CRL A/811/2013", "url": "https://example.org/a", "domain": "example.org"}]}]}


def test_pdf_is_a_real_pdf_with_a_download_name(client):
    r = client.post("/api/export", json={**BODY, "format": "pdf"})
    assert r.status_code == 200 and r.content.startswith(b"%PDF") and r.headers["content-type"] == "application/pdf"
    assert ".pdf" in r.headers["content-disposition"] and "attachment" in r.headers["content-disposition"]


def test_word_file_contains_the_chat_and_sources(client):
    r = client.post("/api/export", json={**BODY, "format": "docx"})
    assert r.status_code == 200
    xml = zipfile.ZipFile(io.BytesIO(r.content)).read("word/document.xml").decode("utf-8")
    assert "Inspector of Police" in xml and "https://example.org/a" in xml and "You" in xml


def test_text_file_is_plain_readable_text(client):
    r = client.post("/api/export", json={**BODY, "format": "txt"})
    assert r.status_code == 200 and "Inspector of Police" in r.text and "CRL A/811/2013" in r.text and "**" not in r.text


@pytest.mark.parametrize("bad", [{"format": "exe"}, {"format": "pdf", "messages": []}])
def test_bad_requests_are_refused(client, bad):
    assert client.post("/api/export", json={**BODY, **bad}).status_code == 422


def test_an_unsafe_title_cannot_break_the_download_name(client):
    r = client.post("/api/export", json={**BODY, "title": 'a"b\r\nX-Evil: 1/../c', "format": "txt"})
    assert r.status_code == 200 and "X-Evil" not in r.headers and "\n" not in r.headers["content-disposition"]
