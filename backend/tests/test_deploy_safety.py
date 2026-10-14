"""The Render deploy installs only requirements-render.txt (no neo4j / chromadb / redis / pyarrow / kafka / pypdf) and configures no stores.
The server must still import and answer: the stores are optional extras, never a hard dependency. (A module-level `import neo4j`
would have taken the live site down on the next push.)"""
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

SCRIPT = textwrap.dedent("""
    import os, sys
    for m in ("neo4j", "chromadb", "redis", "pyarrow", "confluent_kafka", "pypdf", "fakeredis"):
        sys.modules[m] = None                       # importing any of them raises ImportError, as on a machine without the package
    for v in ("NEO4J_URI", "CHROMA_HOST", "REDIS_URL"):
        os.environ.pop(v, None)
    sys.path.insert(0, %r)
    import backend.api.server as server
    app = server.create_app(%r, load_on_start=False, max_concurrent=0)
    assert app.state.kb is None and app.state.rag is None and app.state.memory is None
    print("OK")
""")


def test_server_starts_without_the_store_packages_or_any_store_configured(ckpt_dir):
    code = SCRIPT % (str(ROOT), str(ckpt_dir))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=180)
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-1500:]


def test_render_requirements_do_not_need_the_store_packages():
    text = (ROOT / "requirements-render.txt").read_text(encoding="utf-8").lower()
    for pkg in ("neo4j", "chromadb", "redis", "pyarrow", "confluent", "pyspark"):
        assert pkg not in text            # keeping the deploy image small is intentional; stores are opt-in
