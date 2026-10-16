"""The owner-only data collection dashboard is part of the React source and of the committed build."""
from pathlib import Path

from api.server import REACT_DIST

SRC = Path(__file__).resolve().parent.parent.parent / "frontend-react" / "src"


def test_the_dashboard_is_built_into_the_bundle():
    js = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in (REACT_DIST / "assets").glob("*.js"))
    css = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in (REACT_DIST / "assets").glob("*.css"))
    assert "/api/collect" in js and "X-Owner-Key" in js and "pipelinePage" in js
    assert ".pl-flow" in css and "prefers-reduced-motion" in css


def test_the_header_button_only_shows_in_owner_mode():
    chrome = (SRC / "components" / "Chrome.jsx").read_text(encoding="utf-8")
    app = (SRC / "App.jsx").read_text(encoding="utf-8")
    assert "{owner && (" in chrome and 'id="collectBtn"' in chrome
    assert "const owner = !!getOwnerKey();" in app and "cv && owner" in app
    assert "collectCall(getOwnerKey(), \"/status\")" in app and "owner={owner && canCollect}" in app          # hidden where the server cannot collect (the public site)


def test_no_mock_data_ships_in_the_source():
    text = "".join(p.read_text(encoding="utf-8") for p in (SRC / "pipeline").glob("*.js"))
    assert "mock" not in text.lower() and "fixture" not in text.lower()


def test_the_one_person_panel_is_wired_to_the_targeted_route():
    panel = (SRC / "components" / "TargetedPanel.jsx").read_text(encoding="utf-8")
    hook = (SRC / "hooks" / "usePipelineStatus.js").read_text(encoding="utf-8")
    assert "needs_choice" in panel and 'aria-live="polite"' in panel and "collect C. Joseph Vijay ED cases" in panel
    assert '"/targeted"' in hook
    js = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in (REACT_DIST / "assets").glob("*.js"))
    assert "/targeted" in js and "from Wikidata" in js
