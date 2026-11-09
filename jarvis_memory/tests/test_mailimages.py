"""The team-update e-mail's pictures when the server sends it itself: hosted links instead of attachments, guarded by an unguessable token.
No database: a tiny stand-in plays mail_outbox."""
from contextlib import contextmanager
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import mailimages
from app.config import Settings

KEY = "k" * 32
SETTINGS = Settings(owner_key=KEY, public_url="https://jarvis.example.com/")
PNG = b"\x89PNG\r\n\x1a\nfake"


class FakeDb:
    def __init__(self):
        self.rows = {}

    @contextmanager
    def db(self):
        outer = self

        class Conn:
            last = None

            def execute(self, sql, params=()):
                if sql.startswith("INSERT"):
                    rid = len(outer.rows) + 1
                    outer.rows[rid] = {"to_addr": params[1], "chart_png": params[4]}
                    self.last = {"id": rid}
                else:
                    row = outer.rows.get(params[0])
                    self.last = row if row and row["to_addr"] == params[1] else None
                return self

            def fetchone(self):
                return self.last

        yield Conn()

    owner_id = "owner"


def client(svc):
    app = FastAPI()
    app.state.settings, app.state.service = SETTINGS, svc
    app.include_router(mailimages.make_router())
    return TestClient(app)


def test_the_chart_is_stored_and_served_only_to_its_exact_link():
    svc = FakeDb()
    url = mailimages.store_chart(svc, SETTINGS, PNG)
    assert url.startswith("https://jarvis.example.com/mail-img/chart/1-") and url.endswith(".png")      # no double slash from the trailing /
    c = client(svc)
    path = url.replace("https://jarvis.example.com", "")
    ok = c.get(path)
    assert ok.status_code == 200 and ok.content == PNG and ok.headers["content-type"] == "image/png"
    assert c.get("/mail-img/chart/1-" + "0" * 24 + ".png").status_code == 404           # wrong token
    assert c.get("/mail-img/chart/2-" + path.split("-")[1]).status_code == 404          # the token belongs to chart 1 only
    assert c.get("/mail-img/chart/nonsense.png").status_code == 404


def test_the_rocket_is_public():
    r = client(FakeDb()).get("/mail-img/rocket.gif")
    assert r.status_code == 200 and r.content[:3] == b"GIF" and r.headers["content-type"] == "image/gif"


def test_the_page_links_swap_in_and_nothing_is_stored_without_an_address():
    page = '<img src="cid:rocket"><img src="cid:chart3d">'
    out = mailimages.link_pictures(page, SETTINGS, "https://jarvis.example.com/mail-img/chart/1-abc.png")
    assert out == '<img src="https://jarvis.example.com/mail-img/rocket.gif"><img src="https://jarvis.example.com/mail-img/chart/1-abc.png">'
    nowhere = SimpleNamespace(public_url="", owner_key=KEY)
    assert mailimages.store_chart(FakeDb(), nowhere, PNG) is None and mailimages.link_pictures(page, nowhere, None) == page
