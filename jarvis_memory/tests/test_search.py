from datetime import datetime, timedelta, timezone

import pytest

S = "/api/jarvis/memory/search"
D = lambda day, hour=12: datetime(2026, 10, day, hour, 0, tzinfo=timezone.utc)


@pytest.fixture
def corpus(add):
    """A small, known library of memories with controlled timestamps."""
    return {
        "elvarn": add("Fixed the ELVARN reranker bug and redeployed to Render.", title="ELVARN reranker fix", source="manual",
                      summary="Madhavan fixed a reranker bug.", tags=["elvarn", "debugging"], projects=["ELVARN"], people=["Ravi"],
                      next_steps=["Write regression tests"], tasks=[{"task": "Write regression tests", "status": "pending", "due_at": None}],
                      sentiment="positive", created_at=D(10)),
        "voice": add("Talked through the Jarvis memory design while walking.", title="Jarvis design", source="voice",
                     raw_transcript="talked through the jarvis memory design while walking", tags=["jarvis", "design"],
                     projects=["Jarvis"], people=["Madhavan"], sentiment="neutral", created_at=D(12)),
        "email": add("Replied by email: finished the deployment checklist.", title="Checklist done", source="email",
                     tags=["deployment"], projects=["ELVARN"], tasks=[{"task": "Deployment checklist", "status": "done", "due_at": None}],
                     sentiment="positive", created_at=D(14)),
        "review": add("Daily review: research on transformers, stuck on training instability.", title="Daily review Oct 15",
                      source="daily_review", tags=["research", "transformers"], projects=["ELVARN"],
                      tasks=[{"task": "Fix training instability", "status": "blocked", "due_at": None}], sentiment="negative", created_at=D(15)),
        "plain": add("Went to the market, bought vegetables.", title="Errands", source="manual", created_at=D(16)),
    }


def titles(client, **body):
    r = client.post(S, json=body)
    assert r.status_code == 200, r.text
    return [h["memory"]["title"] for h in r.json()["items"]]


# ------------------------------------------------------------------ text search
def test_search_finds_text_in_content(client, corpus):
    assert titles(client, q="vegetables") == ["Errands"]


def test_search_uses_stemming(client, corpus):
    assert "ELVARN reranker fix" in titles(client, q="redeploying")     # the text says "redeployed"
    assert "Checklist done" in titles(client, q="deploy")               # the text says "deployment"


@pytest.mark.parametrize("q,expected", [
    ("Ravi", "ELVARN reranker fix"),                  # people
    ("debugging", "ELVARN reranker fix"),             # tags
    ("regression", "ELVARN reranker fix"),            # next steps + task text
    ("instability", "Daily review Oct 15"),           # task text / content
    ("transformers", "Daily review Oct 15"),
    ("walking", "Jarvis design"),                     # raw transcript + content
    ("Madhavan", "ELVARN reranker fix"),              # summary
    ("checklist", "Checklist done"),                  # title / content / task
])
def test_every_field_is_searchable(client, corpus, q, expected):
    assert expected in titles(client, q=q)


def test_search_syntax_phrases_exclusion_or(client, corpus):
    assert titles(client, q='"reranker bug"') == ["ELVARN reranker fix"]
    assert "ELVARN reranker fix" not in titles(client, q="ELVARN -reranker")
    both = titles(client, q="vegetables or walking")
    assert set(both) == {"Errands", "Jarvis design"}


def test_no_match_is_an_empty_list_not_an_error(client, corpus):
    r = client.post(S, json={"q": "xyzzy-no-such-word"}).json()
    assert r["items"] == [] and r["total"] == 0 and r["has_more"] is False


def test_punctuation_only_query_does_not_crash(client, corpus):
    assert client.post(S, json={"q": "!!! ???"}).status_code == 200


# ------------------------------------------------------------------ filters
def test_filter_by_source(client, corpus):
    assert titles(client, sources=["voice"]) == ["Jarvis design"]
    assert set(titles(client, sources=["email", "daily_review"])) == {"Checklist done", "Daily review Oct 15"}


def test_filter_by_date_range(client, corpus):
    got = titles(client, date_from=D(11).isoformat(), date_to=D(15, 23).isoformat(), sort="oldest")
    assert got == ["Jarvis design", "Checklist done", "Daily review Oct 15"]
    assert titles(client, date_to=D(10, 13).isoformat()) == ["ELVARN reranker fix"]


def test_filter_by_time_of_day(client, add):
    add("morning note", title="am", created_at=D(20, 9))
    add("evening note", title="pm", created_at=D(20, 21))
    assert titles(client, date_from=D(20, 18).isoformat(), date_to=D(20, 23).isoformat()) == ["pm"]


def test_filter_by_project_person_tag_case_insensitively(client, corpus):
    assert len(titles(client, project="elvarn")) == 3
    assert titles(client, person="RAVI") == ["ELVARN reranker fix"]
    assert titles(client, tag="Transformers") == ["Daily review Oct 15"]


@pytest.mark.parametrize("status,expected", [("pending", "ELVARN reranker fix"), ("done", "Checklist done"), ("blocked", "Daily review Oct 15")])
def test_filter_by_task_status(client, corpus, status, expected):
    assert titles(client, task_status=status) == [expected]


def test_filter_by_sentiment(client, corpus):
    assert set(titles(client, sentiment="positive")) == {"ELVARN reranker fix", "Checklist done"}
    assert titles(client, sentiment="negative") == ["Daily review Oct 15"]


def test_filter_has_transcript_and_attachment(client, corpus):
    assert titles(client, has_transcript=True) == ["Jarvis design"]
    assert len(titles(client, has_transcript=False)) == 4
    assert titles(client, has_attachment=True) == []        # no media exists until Stage 4


def test_filters_combine_with_text(client, corpus):
    assert titles(client, q="deployment", project="ELVARN", sources=["email"]) == ["Checklist done"]
    assert titles(client, q="deployment", sources=["voice"]) == []


# ------------------------------------------------------------------ sorting, paging, snippets
def test_sorting(client, corpus):
    assert titles(client, sort="newest")[0] == "Errands" and titles(client, sort="oldest")[0] == "ELVARN reranker fix"


def test_relevance_puts_the_best_match_first(client, add):
    add("elvarn elvarn elvarn is all I did today with elvarn", title="mostly elvarn", created_at=D(1))
    add("I touched elvarn briefly then did other unrelated things all day long", title="barely", created_at=D(2))
    assert titles(client, q="elvarn", sort="relevance")[0] == "mostly elvarn"
    assert titles(client, q="elvarn", sort="newest")[0] == "barely"


def test_pagination(client, add):
    for i in range(25):
        add(f"entry number {i}", title=f"e{i:02d}", created_at=D(1) + timedelta(minutes=i))
    p1 = client.post(S, json={"page": 1, "page_size": 10}).json()
    p3 = client.post(S, json={"page": 3, "page_size": 10}).json()
    p9 = client.post(S, json={"page": 9, "page_size": 10}).json()
    assert p1["total"] == 25 and len(p1["items"]) == 10 and p1["has_more"] is True
    assert len(p3["items"]) == 5 and p3["has_more"] is False
    assert p9["items"] == [] and p9["total"] == 25            # beyond the end: empty, but the real total is reported
    seen = {h["memory"]["id"] for p in (1, 2, 3) for h in client.post(S, json={"page": p, "page_size": 10}).json()["items"]}
    assert len(seen) == 25                                      # no duplicates or gaps across pages


def test_snippets_highlight_matches_with_safe_markers(client, corpus):
    hit = client.post(S, json={"q": "reranker"}).json()["items"][0]
    assert "\u0002" in hit["snippet"] and "\u0003" in hit["snippet"]
    assert "<mark>" not in hit["snippet"]                       # the UI renders markers itself; no HTML is ever produced


def test_html_in_memories_is_stored_verbatim_and_snippets_are_tag_free(client, add):
    mid = add("reranker <script>alert(1)</script> done", title="xss")
    hit = client.post(S, json={"q": "reranker"}).json()["items"][0]
    assert "<script>" not in hit["snippet"]                               # Postgres strips markup when it builds a snippet
    assert hit["memory"]["content"] == "reranker <script>alert(1)</script> done"   # the original is kept exactly...
    # ...so the UI must always render memory text as text (React does this by default), never as HTML.


def test_without_a_query_the_snippet_is_the_start_of_the_text(client, corpus):
    hit = client.post(S, json={"sort": "oldest"}).json()["items"][0]
    assert hit["snippet"].startswith("Fixed the ELVARN") and "\u0002" not in hit["snippet"]


def test_results_carry_what_the_ui_needs(client, corpus):
    m = client.post(S, json={"q": "reranker"}).json()["items"][0]["memory"]
    for key in ("id", "title", "created_at", "source", "tags", "projects", "people", "attachments", "summary"):
        assert key in m


def test_deleted_memories_never_appear(client, add):
    add("forget me reranker", title="gone", deleted=True)
    assert titles(client, q="reranker") == []


@pytest.mark.parametrize("body", [{"page": 0}, {"page_size": 0}, {"page_size": 500}, {"sort": "random"}, {"sources": ["sms"]}, {"q": "x" * 301}, {"nope": 1}])
def test_bad_search_requests_are_rejected(client, body):
    assert client.post(S, json=body).status_code == 422
