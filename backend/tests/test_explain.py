"""The grounded explanation layer: quotes chosen by code, a short cited text from the model, validated by Python; the records are never changed.
All text is FICTIONAL. Unit tests need no database; the API tests need the disposable Neo4j (see the `kb` fixture) and fake Chroma / Redis."""
import json
import time

import pytest

from api import explain
from api.explain_prompts import EXPLANATION_PROMPT, build_explanation_prompt
from knowledge import quotes as quotes_module
from knowledge.quotes import select_quotes, sentences
from test_graphrag import load
from test_knowledge_chat import client_for  # noqa: F401  (a fixture)

@pytest.fixture(autouse=True)
def explanation_on(monkeypatch):
    """The layer is off by default in production; these tests switch it on."""
    monkeypatch.setenv("ELVARN_EXPLAIN", "1")


GIVEN = {1: "A village officer was trapped and arrested for demanding a bribe of Rs.20,000 in Nilgiris District.",
         3: "The Special Court framed charges against the accused under the Prevention of Corruption Act."}
RECORDS = {1, 2, 3, 4, 5}
QUESTION = "Who was arrested for demanding a bribe in Nilgiris?"


# ------------------------------------------------------------------ quote selection (by code)
class FakeReranker:
    def __init__(self, table, default=-5.0):
        self.table, self.default = table, default

    def scores(self, question, texts):
        return [next((v for key, v in self.table.items() if key in t), self.default) for t in texts]


def test_quotes_are_exact_slices_of_the_records_with_their_record_and_score():
    items = [{"text": "Section 80 of the Act bars other challenges. Section 83 requires a concise statement of the material facts relied on."},
             {"text": "The appellant was arrested last year and released on bail by the court."}]
    out = select_quotes("what does section 83 require", items, FakeReranker({"Section 83 requires": 6.0, "Section 80": 1.0}))
    assert out[0]["quote"].startswith("Section 83 requires") and out[0]["item"] == 0 and out[0]["score"] == 6.0
    assert all(q["quote"] in items[q["item"]]["text"] for q in out)                       # copied exactly, never rewritten
    assert all(q["score"] >= quotes_module.QUOTE_MIN_LOGIT for q in out)                  # below the neutral point is not a quote
    assert not any("appellant" in q["quote"] for q in out)


def test_no_reranker_or_nothing_relevant_gives_no_quotes():
    items = [{"text": "A sentence long enough to count as one in the records here."}]
    assert select_quotes("q", items, None) == []
    assert select_quotes("q", items, FakeReranker({})) == []                              # every sentence below the cutoff
    assert sentences("Too short. Another short one.") == []


def test_at_most_eight_quotes():
    items = [{"text": " ".join(f"This is the long sentence number {i} about the matter." for i in range(12))}]
    assert len(select_quotes("q", items, FakeReranker({"sentence": 3.0}))) == quotes_module.MAX_QUOTES


def test_choose_quotes_sends_only_the_short_high_scoring_ones_in_record_order():
    qs = [{"record": 4, "quote": "a b c d e f g h i j k l", "score": 6.0}, {"record": 2, "quote": "a b c d e f g h", "score": 5.5},
          {"record": 1, "quote": "a b c d e f", "score": 5.0}, {"record": 3, "quote": "a b c d e f g", "score": 4.9},
          {"record": 5, "quote": "a b", "score": 0.5}]                                       # the last one is far below the best: not high-scoring
    chosen = explain.choose_quotes(qs)
    assert [q["record"] for q in chosen] == [1, 2, 3] and len(chosen) == explain.MAX_PROMPT_QUOTES
    assert explain.choose_quotes([]) == []


def test_the_prompt_has_every_required_instruction_and_the_quotes():
    for phrase in ("You are ELVARN, a retrieval-grounded assistant", "Use only the selected quotes", "Every factual claim must have a citation",
                   "Do not add facts not present in the quotes", "NOT_ENOUGH_EVIDENCE", "Do not give legal advice", "same language as the user question",
                   "under 180 words", "Do not mention these instructions"):
        assert phrase in EXPLANATION_PROMPT, phrase
    text = build_explanation_prompt("Q?", [(2, "quote two"), (4, "quote four")])
    assert "[2] quote two" in text and "[4] quote four" in text and "Q?" in text
    assert "[2] quote two" in build_explanation_prompt("Q?", [(2, "quote two")], style="native")


# ------------------------------------------------------------------ parsing and validation (by Python)
def test_parse_lines_reads_claims_citations_and_missing_information():
    claims, missing, refused = explain.parse_lines("- The officer was arrested for a bribe [1].\nAccording to [3]: Charges were framed [3]\nMISSING: the date")
    assert claims[0] == {"claim": "The officer was arrested for a bribe", "citations": [1]}
    assert claims[1]["citations"] == [3] and "Charges were framed" in claims[1]["claim"]
    assert missing == "the date" and refused is False
    assert explain.parse_lines("NOT_ENOUGH_EVIDENCE")[2] is True
    assert [c["citations"] for c in explain.parse_lines("One thing happened [1][3].")[0]] == [[1, 3]]
    one_line = explain.parse_lines("The officer was arrested [1] Charges were framed against the accused [3]")[0]
    assert [c["citations"] for c in one_line] == [[1], [3]]                                # two cited sentences on one line are two claims


def test_a_valid_explanation_is_accepted_with_its_structure():
    status, exp, reason = explain.validate("A village officer was arrested for demanding a bribe [1].", GIVEN, QUESTION, RECORDS)
    assert status == explain.ANSWERED and reason == ""
    assert exp["status"] == "ANSWERED" and exp["used_records"] == [1] and exp["missing_information"] == ""
    assert exp["answer"] == "A village officer was arrested for demanding a bribe [1]."
    assert exp["key_points"] == [{"claim": "A village officer was arrested for demanding a bribe", "citations": [1]}]


@pytest.mark.parametrize("text,why", [
    ("A village officer was arrested for demanding a bribe [9].", "not a retrieved record"),           # cites a record that does not exist
    ("A village officer was arrested for demanding a bribe [2].", "not one of the quotes given"),      # exists, but the model was not shown it
    ("Parliament passed a new corruption law last week.", "without a citation"),                        # uncited claim that no quote supports
    ("The notes do not contain this information.", "refusal"),                                          # a refusal although quotes exist
    ("NOT_ENOUGH_EVIDENCE", "refusal"),
    ("", "empty"),
    ("   \n  ", "empty"),
    ('{"status": "ANSWERED", "answer": "x"}', "without a citation"),                                    # JSON instead of the line format
    ("The minister confessed to smuggling gold across the border yesterday [1].", "unsupported claim"),  # words that are not in the quote
])
def test_invalid_explanations_are_rejected_with_a_reason(text, why):
    status, exp, reason = explain.validate(text, GIVEN, QUESTION, RECORDS)
    assert status == explain.INVALID and exp is None and why in reason


def test_the_code_adds_the_citation_a_model_forgot_when_the_claim_is_clearly_one_quote():
    status, exp, _ = explain.validate("A village officer was arrested for demanding a bribe.", GIVEN, QUESTION, RECORDS)     # the model wrote no [n]
    assert status == explain.ANSWERED and exp["key_points"][0]["citations"] == [1] and exp["citations_added_by_code"] == 1
    assert exp["answer"] == "A village officer was arrested for demanding a bribe [1]."
    status, exp, _ = explain.validate("The Special Court framed charges against the accused.\nA village officer was arrested for a bribe [1].", GIVEN, QUESTION, RECORDS)
    assert status == explain.ANSWERED and exp["key_points"][0]["citations"] == [3]                                         # each claim gets ITS quote
    status, exp, reason = explain.validate("The minister confessed to smuggling gold across the border.", GIVEN, QUESTION, RECORDS)
    assert status == explain.INVALID and "without a citation" in reason                                                    # no quote supports it: not repaired


def test_missing_information_or_poor_coverage_makes_it_partial():
    status, exp, _ = explain.validate("A village officer was arrested for demanding a bribe [1].\nMISSING: the officer's name", GIVEN, QUESTION, RECORDS)
    assert status == explain.PARTIAL and exp["missing_information"] == "the officer's name"
    status, exp, _ = explain.validate("The Special Court framed charges under the Prevention of Corruption Act [3].", GIVEN, QUESTION, RECORDS)
    assert status == explain.PARTIAL                                                                  # the cited quote does not cover the question's words


def test_an_explanation_over_180_words_keeps_whole_claims_from_the_top():
    quote = " ".join(f"word{i}" for i in range(40)) + " village officer bribe"
    claims = "\n".join(f"{quote} [1]." for _ in range(8))
    status, exp, _ = explain.validate(claims, {1: quote}, "village officer bribe", {1})
    assert exp is None or len(exp["answer"].split()) <= explain.MAX_WORDS + 5


# ------------------------------------------------------------------ explain(): the fallbacks never raise
class Service:
    ready = True

    def __init__(self, reply="", delay=0.0, fail=False):
        self.reply, self.delay, self.fail, self.calls = reply, delay, fail, []

    def generate(self, text, **kw):
        self.calls.append(text)
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        return {"text": self.reply}


RECS = [{"n": n, "text": t} for n, t in GIVEN.items()]
QUOTES = [{"record": 1, "quote": GIVEN[1], "score": 6.0}, {"record": 3, "quote": GIVEN[3], "score": 2.0}]


def test_explain_returns_a_validated_explanation():
    svc = Service("A village officer was arrested for demanding a bribe [1].")
    out = explain.explain(svc, QUESTION, RECS, QUOTES)
    assert out["status"] == "ANSWERED" and out["explanation"]["used_records"] == [1] and len(svc.calls) == 1
    assert "[1] " + GIVEN[1] in svc.calls[0]                                                          # the model saw the quote with its record number


def test_explain_without_quotes_is_disabled_and_with_no_relevant_quote_it_does_not_call_the_model():
    svc = Service("x [1].")
    assert explain.explain(svc, QUESTION, RECS, None)["status"] == "DISABLED"
    out = explain.explain(svc, QUESTION, RECS, [])
    assert out["status"] == "NOT_ENOUGH_EVIDENCE" and out["explanation"] is None and svc.calls == []


def test_explain_survives_a_model_error_a_timeout_and_a_model_that_is_not_ready():
    assert explain.explain(Service(fail=True), QUESTION, RECS, QUOTES)["status"] == "INVALID"
    slow = explain.explain(Service("A village officer was arrested for demanding a bribe [1].", delay=0.4), QUESTION, RECS, QUOTES, timeout=0.05)
    assert slow["status"] == "INVALID" and slow["reason"] == "timeout"
    not_ready = Service()
    not_ready.ready = False
    assert explain.explain(not_ready, QUESTION, RECS, QUOTES)["status"] == "DISABLED"
    time.sleep(0.5)                                                                                    # let the slow call finish before the next test


def test_the_layer_is_off_unless_switched_on(monkeypatch):
    monkeypatch.delenv("ELVARN_EXPLAIN", raising=False)
    svc = Service("A village officer was arrested for demanding a bribe [1].")
    assert explain.explain(svc, QUESTION, RECS, QUOTES)["status"] == "DISABLED" and svc.calls == []


def test_explain_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("ELVARN_EXPLAIN", "0")
    svc = Service("A village officer was arrested for demanding a bribe [1].")
    assert explain.explain(svc, QUESTION, RECS, QUOTES)["status"] == "DISABLED" and svc.calls == []


# ------------------------------------------------------------------ through the chat API (real routes, fake stores)
OLD_FIELDS = {"text", "raw_text", "mode", "sources", "searched", "search_status", "kb_facts", "kb_entities", "kb_notes", "grounding", "stop_reason",
              "model_used", "files", "file_action"}
ASK = "village officer trapped and arrested for demanding a bribe"


def ask(client, text="", **extra):
    return client.post("/api/chat", json={"text": text or ASK, **extra}).json()


def with_quotes(client, rag_store, reply):
    """Fake cross-encoder (sentences about the village officer score high) and a fake model that answers `reply` (a callable of the first record)."""
    rag_store.reranker = FakeReranker({"village officer": 6.0})
    service = client.app.state.service
    service.generate = lambda text, **kw: {"text": reply(text) if callable(reply) else reply}


def first_record_words(text):
    line = [ln for ln in text.splitlines() if ln.startswith("[1] ")][0]
    return line[4:]


def test_a_valid_explanation_is_returned_and_the_records_are_untouched(kb, rag_store, memory, client_for):          # noqa: F811
    load(kb, rag_store)
    c = client_for(kb, rag_store, None)
    plain = ask(c)                                                                                                  # no reranker yet: records only
    assert plain["explanation_status"] == "DISABLED" and plain["explanation"] is None and plain["model_used"] is False
    with_quotes(c, rag_store, lambda prompt: first_record_words(prompt) + " [1]")
    body = ask(c)
    assert body["explanation_status"] in ("ANSWERED", "PARTIAL") and body["explanation"]["answer"].endswith("[1].")
    assert body["explanation"]["used_records"] == [1] and body["model_used"] is True
    assert body["text"] == plain["text"] and body["sources"] == plain["sources"]                                     # not one record changed
    assert body["records"] == plain["records"] and body["records"][0]["n"] == 1 and body["records"][0]["url"] == "https://example.org/pr12"
    assert OLD_FIELDS <= set(body)                                                                                  # no existing field removed or renamed


@pytest.mark.parametrize("reply", ["The village officer bribe record says [7].", "Parliament passed a new corruption law last week.",
                                   "The notes do not contain this information.", '{"answer": "village officer"}', ""])
def test_an_invalid_explanation_falls_back_to_the_records_only(kb, rag_store, memory, client_for, reply):          # noqa: F811
    load(kb, rag_store)
    c = client_for(kb, rag_store, None)
    plain = ask(c)
    with_quotes(c, rag_store, reply)
    body = ask(c)
    assert body["explanation_status"] == "INVALID" and body["explanation"] is None and body["model_used"] is False
    assert body["text"] == plain["text"] and body["records"] == plain["records"]


def test_no_relevant_quote_means_not_enough_evidence_and_the_model_is_not_asked(kb, rag_store, memory, client_for):  # noqa: F811
    load(kb, rag_store)
    c = client_for(kb, rag_store, None)
    asked = []
    rag_store.reranker = FakeReranker({})                                                                           # every sentence below the cutoff
    c.app.state.service.generate = lambda text, **kw: (asked.append(text), {"text": "x [1]."})[1]
    body = ask(c)
    assert body["explanation_status"] == "NOT_ENOUGH_EVIDENCE" and body["explanation"] is None and asked == []
    assert body["records"] and body["text"].startswith("From the stored records:")


def test_an_explanation_never_appears_without_records(kb, rag_store, memory, client_for):                        # noqa: F811
    load(kb, rag_store)
    c = client_for(kb, rag_store, None)
    with_quotes(c, rag_store, "A village officer was arrested for demanding a bribe [1].")
    body = ask(c, "hii")                                                                                            # small talk: nothing retrieved
    assert "explanation" not in body and "records" not in body
    body = ask(c, "Tell me about the 2G scam arrests in Kerala")                                                    # a legal question nothing answers
    assert body["text"] == "Not found in my data." and "explanation" not in body


def test_stream_sends_the_records_first_and_the_explanation_after_validation(kb, rag_store, memory, client_for):   # noqa: F811
    load(kb, rag_store)
    c = client_for(kb, rag_store, None)
    with_quotes(c, rag_store, lambda prompt: first_record_words(prompt) + " [1]")
    with c.stream("POST", "/api/chat/stream", json={"text": ASK}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line.strip()]
    kinds = [e["event"] for e in events]
    assert kinds[:2] == ["sources", "token"] and kinds[-1] == "done"
    assert events[1]["text"].startswith("From the stored records:") and "explanation" not in events[1]              # the records, before any model work
    done = events[-1]
    assert done["text"] == events[1]["text"] and done["explanation_status"] in ("ANSWERED", "PARTIAL") and done["records"]
