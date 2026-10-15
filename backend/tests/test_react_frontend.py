"""The React app (frontend-react/) is built into dist/, committed (Render has no Node step) and served by the Python server."""
import re
from pathlib import Path

from fastapi.testclient import TestClient

from api.server import REACT_DIST, create_app

ROOT = Path(__file__).resolve().parent.parent.parent


def test_the_built_app_is_present_and_points_at_files_that_exist():
    html = (REACT_DIST / "index.html").read_text(encoding="utf-8")
    assert '<div id="root">' in html
    refs = re.findall(r'(?:src|href)="/static/(assets/[^"]+)"', html)
    assert refs and all((REACT_DIST / r).is_file() for r in refs)


def test_the_server_serves_the_react_app_and_its_assets(ckpt_dir):
    client = TestClient(create_app(ckpt_dir, load_on_start=False, max_concurrent=0))
    page = client.get("/")
    assert page.status_code == 200 and '<div id="root">' in page.text
    asset = re.search(r'src="(/static/assets/[^"]+\.js)"', page.text).group(1)
    js = client.get(asset)
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert client.get("/static/logo.png").status_code == 200
    assert client.get("/favicon.png").status_code == 200


def test_the_source_is_split_into_many_files_not_one():
    src = ROOT / "frontend-react" / "src"
    files = [p for p in src.rglob("*") if p.suffix in (".jsx", ".js", ".css")]
    assert len(files) >= 25 and max(len(p.read_text(encoding="utf-8").splitlines()) for p in files) < 560


def test_node_modules_are_not_committed():
    assert "frontend-react/node_modules/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
