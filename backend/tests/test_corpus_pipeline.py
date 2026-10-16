"""Processing side of the knowledge pipeline: HTML cleaning, language, quality, passages, entities, Wikidata relations, the Spark batch logic
(without Spark) and check_stores. Every name here is FICTIONAL. Tests with the `kb` fixture need the disposable Neo4j (NEO4J_TEST_URI)."""
import json
import re

import pytest

from backend.pipeline import corpus, entities, process
from backend.pipeline import spark_job


def prose(n, start=0):
    """Distinct English sentences (no line repeats, no duplicate passages), about 150 characters each."""
    return " ".join(f"In {1950 + (i % 60)} the fictional Council of Thenpuram reviewed budget item {i} and recorded decision number {i * 7} "
                    f"about the water scheme of ward {i % 9}." for i in range(start, start + n))


def paragraphs(n_sentences, start=0, per=5):
    sentences = [prose(1, i) for i in range(start, start + n_sentences)]
    return "\n\n".join(" ".join(sentences[i:i + per]) for i in range(0, len(sentences), per))


WIKI_HTML = """<!DOCTYPE html><html><head><title>Aravind Kesavan</title><style>.mw-parser-output{color:red}</style>
<script>var tracking = "cookie";</script></head><body>
<header><a href="/">Home</a></header>
<div class="cookie-banner">We use cookies to improve your experience. Accept cookies</div>
<nav><ul><li><a href="/a">Main page</a></li><li><a href="/b">Contents</a></li></ul></nav>
<section data-mw-section-id="0">
<div role="note" class="hatnote">For the footballer of the same name, see Aravind Kesavan (footballer).</div>
<table class="infobox"><tr><th>Born</th><td>3 June 1924</td></tr><tr><th>Party</th><td>Thenpuram Progressive Front</td></tr></table>
<p><b>Aravind Kesavan</b> (1924 to 2001) was a fictional politician of Thenpuram.<sup class="mw-ref reference"><a href="#cite_note-1">[1]</a></sup>
 He led the Thenpuram Progressive Front for twenty years.<span class="mw-editsection"><a>edit</a></span></p>
<div class="toc" id="toc"><ul><li>1 Early life</li><li>2 Career</li></ul></div>
</section>
<section data-mw-section-id="1"><h2 id="Early_life">Early life<span class="mw-editsection"><a href="/edit">edit</a></span></h2>
<p>PARA1</p><p>PARA2</p></section>
<section data-mw-section-id="2"><h2 id="Career">Career</h2><p>PARA3</p>
<div class="navbox"><ul><li><a>Thenpuram politicians</a></li></ul></div></section>
<section data-mw-section-id="3"><h2 id="References">References</h2><ol class="references"><li>Fictional Gazette, 1960.</li></ol></section>
<section data-mw-section-id="4"><h2 id="External_links">External links</h2><ul><li><a href="http://example.org">Official site</a></li></ul></section>
<footer>Text is available under the Creative Commons licence. Privacy policy</footer>
</body></html>""".replace("PARA1", prose(6)).replace("PARA2", prose(6, 10)).replace("PARA3", prose(6, 20))


# ---- HTML, normalisation ---------------------------------------------------------------------------------------
def test_html_to_text_keeps_the_article_and_drops_the_page_furniture():
    text = corpus.html_to_text(WIKI_HTML)
    assert "Aravind Kesavan (1924 to 2001) was a fictional politician" in text
    assert "Early life" in text and "Career" in text and "budget item 3 " in text
    for gone in ("tracking", "mw-parser-output", "cookies", "Main page", "footballer", "Born", "Progressive Front for twenty years.edit", "[1]",
                 "1 Early life", "Thenpuram politicians", "Fictional Gazette", "Official site", "Privacy policy", "edit"):
        assert gone not in text, gone
    assert "References" not in text and "External links" not in text
    assert text.index("Early life") < text.index("Career")          # headings stay in order with their paragraphs
    link_text, ratio = corpus.html_to_text_info("<p>" + "<a href='x'>menu item</a> " * 30 + "</p>")
    assert ratio > 0.9


def test_normalize_cleans_whitespace_marks_and_repeated_boilerplate():
    raw = "Café   in  Thenpuram[1] was founded.[citation needed]\r\n\r\n\r\n\r\nSkip to content\nShare this page\nSubscribe now\n" + "Related\n" * 3 + "Last line. "
    out = corpus.normalize(raw)
    assert "Café in Thenpuram was founded." in out                # NFC and single spaces
    assert "[1]" not in out and "citation needed" not in out
    assert "Skip to content" not in out and "Share this page" not in out and "Related" not in out
    assert "\n\n\n" not in out and out.endswith("Last line.")


# ---- language and garbled text -------------------------------------------------------------------------------------
TAMIL = " ".join(f"இது தேர்தல் நடைமுறை பற்றிய செய்தி எண் {i}. மக்கள் அனைவரும் வாக்களிக்க வேண்டும்." for i in range(40))


def test_language_detection_english_tamil_and_unknown():
    assert corpus.detect_language(prose(5)) == "en"
    assert corpus.detect_language(TAMIL) == "ta"
    assert corpus.detect_language("1234 5678 !!!") == "unknown" and corpus.detect_language("") == "unknown"
    assert corpus.detect_language("Привет мир, это длинный русский текст для проверки языка.") == "unknown"


def test_garbled_legacy_tamil_is_rejected_but_real_tamil_is_not():
    legacy = ("Ì¸¢ÕõÀ¢ø §¾º¢Âý ¿¡ðÎ º¢ÕÀ¡ðÎ ¸Ä¡º¡Ãõ «È¢×õ ¦¸¡ñÎ «Ãº¢Âø ¯Ä¸õ ") * 30
    assert corpus.looks_garbled(legacy) and corpus.detect_language(legacy) == "unknown"
    assert not corpus.looks_garbled(TAMIL) and not corpus.looks_garbled(prose(5))
    assert process.looks_garbled("u�aOiQ�6Q�lGlsuja^ �^iMi^g-j�i�siMQj") and process.looks_garbled("   ")
    score, reasons = corpus.quality(legacy)
    assert "garbled" in reasons


# ---- quality gate ---------------------------------------------------------------------------------------------------
def test_quality_accepts_a_good_document_and_applies_every_rejection_rule():
    good = paragraphs(20)
    score, reasons = corpus.quality(good)
    assert reasons == [] and 0.5 < score <= 1.0
    assert "too_short" in corpus.quality("A short page. " * 10)[1]
    assert corpus.quality("A short page. " * 10, min_chars=0)[1] == []                       # the length rule is for documents only
    assert "low_letter_ratio" in corpus.quality(("1234 5678 9012 3456 7890 $%&# " * 80))[1]
    menu = "\n".join(f"Menu item {chr(65 + i % 26)}" for i in range(60)) + "\n" + "x" * 1000
    assert "mostly_links_or_menus" in corpus.quality(menu)[1]
    assert "mostly_links_or_menus" in corpus.quality(good, link_ratio=0.8)[1]
    repeated = "\n".join(["The same line again and again for the reader to see."] * 5 + [prose(1, i) for i in range(40, 44)]) * 2
    assert "repeated_lines" in corpus.quality(repeated)[1]
    spaced = " ".join(" ".join(w) for w in prose(12).split())                                 # "T h e  c o u n c i l ..."
    assert "poor_extraction" in corpus.quality(spaced)[1]
    assert corpus.quality("")[0] <= 0.49 and corpus.quality("")[1]


# ---- passages --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sentences", [40, 60, 110, 240])
def test_passages_are_300_to_1000_tokens_overlap_and_never_cut_words(sentences):
    text = paragraphs(sentences)
    parts = corpus.split_passages(text)
    assert len(parts) >= 2
    words = set(text.split())
    for i, p in enumerate(parts):
        assert 300 <= corpus.token_estimate(p) <= 1000, (i, len(p))
        assert set(p.split()) <= words                                        # no word was cut
    for prev, nxt in zip(parts, parts[1:]):
        shared = next(n for n in range(len(prev), 0, -1) if nxt.startswith(prev[-n:]) and prev[-n - 1:-n].isspace() or n == 1)
        ratio = shared / len(prev)
        assert 0.10 <= ratio <= 0.20, ratio
    assert corpus.split_passages(text) == parts                               # deterministic


def test_a_short_document_is_one_passage_and_empty_text_is_none():
    text = prose(3)
    assert corpus.split_passages(text) == [text] and corpus.split_passages("  ") == []
    assert corpus.token_estimate("abcde") == 2 and corpus.token_estimate(400) == 100


def test_passage_records_have_stable_ids_hashes_and_drop_exact_duplicates():
    from backend.knowledge.kb import passage_id
    meta = {"doc_id": "doc_a", "source": "web", "source_url": "https://u/a", "title": "A", "person_key": "wd:Q1", "person_name": "Aravind Kesavan",
            "topic": "person", "created_at": "2026-10-16T09:00:00+00:00"}
    text = paragraphs(60)
    recs, info = corpus.build_passages(meta, text, "en", 0.9)
    again, _ = corpus.build_passages(meta, text, "en", 0.9)
    assert [r["passage_id"] for r in recs] == [r["passage_id"] for r in again]
    assert [r["passage_id"] for r in recs] == [passage_id(corpus.sha256_text(text), i) for i in range(len(recs))]
    assert info["document_sha256"] == corpus.sha256_text(text) and all(set(corpus.COLUMN_NAMES) <= set(r) for r in recs)
    assert all(r["token_estimate"] == corpus.token_estimate(r["char_count"]) and not r["embedded"] and not r["indexed_neo4j"] for r in recs)
    seen = set()
    first, _ = corpus.build_passages(meta, text, "en", 0.9, seen)
    second, info2 = corpus.build_passages({**meta, "doc_id": "doc_b"}, text + "\n\n" + prose(1, 999), "en", 0.9, seen)   # same text in another document
    assert len(first) == len(recs) and info2["duplicates"] >= len(recs) - 1
    assert len({r["sha256"] for r in first + second}) == len(first + second)


def test_passage_index_remembers_ids_across_runs(tmp_path):
    idx = corpus.PassageIndex(tmp_path)
    rec = {"passage_id": "p1", "sha256": "h1"}
    assert idx.classify(rec) == "new"
    idx.stage(rec)
    idx.rollback()
    assert idx.classify(rec) == "new"                                         # a failed batch leaves no trace
    idx.stage(rec)
    idx.commit()
    later = corpus.PassageIndex(tmp_path)
    assert later.classify(rec) == "known" and later.classify({"passage_id": "p2", "sha256": "h1"}) == "duplicate"


# ---- entities ---------------------------------------------------------------------------------------------------------
def test_entity_extraction_from_a_passage():
    text = ("Thiru Murugan Selvam of the Thenpuram Progressive Front met Aravind Kesavan in Madurai on 12 October 2019. The DVAC and the CBI "
            "registered the disproportionate assets case and the Harbour Land scam; the Madras High Court heard it in 2021. See https://example.org/report.")
    roster = [{"name": "Aravind Kesavan", "aliases": ["A. Kesavan"], "person_key": "wd:Q9000001", "wikidata_id": "Q9000001"}]
    found = {(e["type"], e["name"]): e for e in entities.extract_entities(text, roster)}
    assert found[("Person", "Aravind Kesavan")]["person_key"] == "wd:Q9000001"
    assert ("Person", "Murugan Selvam") in found
    assert ("InvestigatingAgency", "Directorate of Vigilance and Anti-Corruption") in found and ("InvestigatingAgency", "Central Bureau of Investigation") in found
    assert ("Court", "Madras High Court") in found and ("Place", "Madurai") in found
    assert ("Date", "2019-10-12") in found and ("Date", "2021") in found and ("Reference", "https://example.org/report") in found
    assert ("Case", "Disproportionate assets case") in found and ("Case", "Harbour Land scam") in found
    assert not any(e["type"] == "Person" and e["name"] in ("The", "Madras") for e in found.values())
    assert entities.extract_entities("the ed and the cbi said nothing about it", []) == []       # acronyms are case sensitive


def test_cooccurrence_is_never_an_accusation():
    ents = [{"type": "Person", "name": "Aravind Kesavan"}, {"type": "Case", "name": "Harbour Land scam"}, {"type": "Place", "name": "Madurai"}]
    rels = entities.cooccurrence_relations(ents)
    assert len(rels) == 1
    r = rels[0]
    assert (r["type"], r["method"], r["confidence"]) == ("MENTIONED_WITH", "cooccurrence", "low")
    for doc_type in ("reference", "news", "judgment", "order", "press_release"):
        assert entities.assertion_level(doc_type, "cooccurrence") in ("alleged", "reported")


def test_assertion_level_rule():
    assert entities.assertion_level("reference", "wikidata") == "reported"
    assert entities.assertion_level("news", "wikidata") == "alleged"
    assert entities.assertion_level("judgment", "wikidata") == "court_finding" and entities.assertion_level("order", "rule") == "court_finding"
    assert entities.assertion_level("press_release", "rule") == "reported"


WIKIDATA = {"entities": {"Q9000001": {
    "id": "Q9000001", "labels": {"en": {"language": "en", "value": "Aravind Kesavan"}, "ta": {"language": "ta", "value": "அரவிந்த் கேசவன்"}},
    "descriptions": {"en": {"value": "fictional Indian politician"}}, "aliases": {"en": [{"value": "A. Kesavan"}]},
    "claims": {
        "P21": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q6581097"}}}, "rank": "normal"}],
        "P569": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"time": "+1924-06-03T00:00:00Z", "precision": 11}}}, "rank": "normal"}],
        "P570": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"time": "+2001-02-00T00:00:00Z", "precision": 10}}}, "rank": "normal"}],
        "P19": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q9000010"}}}, "rank": "normal"}],
        "P102": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q9000020"}}}, "rank": "normal", "qualifiers": {
            "P580": [{"snaktype": "value", "datavalue": {"value": {"time": "+1949-01-01T00:00:00Z", "precision": 9}}}],
            "P582": [{"snaktype": "value", "datavalue": {"value": {"time": "+2001-01-01T00:00:00Z", "precision": 9}}}]}},
                 {"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q9000021"}}}, "rank": "deprecated"}],
        "P39": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q9000030"}}}, "rank": "preferred", "qualifiers": {
            "P580": [{"snaktype": "value", "datavalue": {"value": {"time": "+1969-02-10T00:00:00Z", "precision": 11}}}],
            "P582": [{"snaktype": "value", "datavalue": {"value": {"time": "+1971-01-31T00:00:00Z", "precision": 11}}}],
            "P768": [{"snaktype": "value", "datavalue": {"value": {"id": "Q9000040"}}}]}}],
        "P1038": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q9000050"}}}, "rank": "normal", "qualifiers": {
            "P1039": [{"snaktype": "value", "datavalue": {"value": {"id": "Q9000051"}}}]}}],
        "P69": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": "Q9000060"}}}, "rank": "normal"}]}}},
    "labels": {"Q9000010": "Thenpuram", "Q9000020": "Thenpuram Progressive Front", "Q9000030": "Chief Minister of Thenpuram",
               "Q9000040": "Thenpuram North", "Q9000050": "Karthik Kesavan", "Q9000051": "father", "Q9000060": "Fictional University of Thenpuram"}}


def test_relations_from_wikidata():
    ents, rels = entities.relations_from_wikidata(WIKIDATA, "wd:Q9000001")
    by = {r["type"]: r for r in rels}
    assert set(by) == {"MEMBER_OF_PARTY", "HELD_POSITION", "CONSTITUENCY", "RELATIVE_OF", "BORN_IN", "EDUCATED_AT"}
    assert all(r["method"] == "wikidata" and r["confidence"] == "high" for r in rels)
    party = by["MEMBER_OF_PARTY"]
    assert party["to"] == {"type": "PoliticalParty", "name": "Thenpuram Progressive Front"} and (party["start"], party["end"]) == ("1949", "2001")
    assert len([r for r in rels if r["type"] == "MEMBER_OF_PARTY"]) == 1                    # the deprecated claim is ignored
    held = by["HELD_POSITION"]
    assert held["to"]["type"] == "Position" and (held["start"], held["end"]) == ("1969-02-10", "1971-01-31")
    assert by["CONSTITUENCY"]["to"] == {"type": "Place", "name": "Thenpuram North"}
    assert by["RELATIVE_OF"]["kind"] == "father" and by["RELATIVE_OF"]["to"]["name"] == "Karthik Kesavan"
    assert by["BORN_IN"]["to"]["name"] == "Thenpuram" and by["EDUCATED_AT"]["to"]["type"] == "Organisation"
    person = next(e for e in ents if e["type"] == "Person" and e["name"] == "Aravind Kesavan")
    assert person["person_key"] == "wd:Q9000001" and person["wikidata_id"] == "Q9000001" and person["birth_date"] == "1924-06-03" and person["death_date"] == "2001-02"
    assert {e["type"] for e in ents} >= {"Person", "PoliticalParty", "Position", "Place", "Organisation"}


def test_wikidata_summary_reads_like_english():
    text = entities.wikidata_summary(WIKIDATA)
    assert text.startswith("Aravind Kesavan (born 3 June 1924 in Thenpuram; died February 2001)")
    assert "He was a member of the party Thenpuram Progressive Front from 1949 to 2001." in text
    assert "He held the position of Chief Minister of Thenpuram from 10 February 1969 to 31 January 1971, representing Thenpuram North." in text
    assert "His father was Karthik Kesavan." in text and "Thenpuram Progressive Front" in text
    assert entities.wikidata_summary({"entities": {"Q1": {"claims": {}}}}) == "" and entities.relations_from_wikidata({"entities": {}}) == ([], [])


# ---- process_message on archived files -----------------------------------------------------------------------------------
def message(doc_id, raw_path, **extra):
    base = {"event_type": "document_ingested", "doc_id": doc_id, "source": "web", "source_url": f"https://en.wikipedia.org/wiki/{doc_id}",
            "raw_path": raw_path, "mime_type": "text/html", "language": "en", "collected_at": "2026-10-16T09:30:00+00:00", "status": "NEW",
            "title": "Aravind Kesavan", "doc_type": "reference", "publisher": "Wikipedia", "licence": "CC BY-SA 4.0", "site": "politician",
            "person_key": "wd:Q9000001", "person_name": "Aravind Kesavan", "topic": "person", "wikidata_id": "Q9000001"}
    base.update(extra)
    return base


def archive(tmp_path, rel, content):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return "kb_raw/" + rel


def test_process_message_html_with_and_without_the_kb_raw_prefix(tmp_path):
    raw = archive(tmp_path, "web/2026-10-16/doc_aaa.html", WIKI_HTML)
    r = process.process_message(message("doc_aaa", raw), tmp_path)
    r2 = process.process_message(message("doc_aaa", raw[len("kb_raw/"):]), tmp_path)
    assert not r["skipped"] and r["date"] == "2026-10-16" and r["language"] == "en" and r["records"]
    assert [x["passage_id"] for x in r["records"]] == [x["passage_id"] for x in r2["records"]]
    assert all("cookies" not in x["text"] and "Born" not in x["text"] for x in r["records"])
    assert r["records"][0]["entities"] and any(e["type"] == "Person" and e["person_key"] == "wd:Q9000001" for x in r["records"] for e in x["entities"])
    assert r["passages"][0]["locator"] == "passage 1" and r["doc"]["sha256"] == corpus.sha256_text(r["text"])


def test_process_message_rejects_short_menu_and_tamil_garbage(tmp_path):
    short = archive(tmp_path, "web/2026-10-16/doc_s.html", "<html><body><p>Only a few words here, but long enough for the language check to pass.</p></body></html>")
    assert process.process_message(message("doc_s", short), tmp_path)["reasons"] == ["too_short"]
    menu = archive(tmp_path, "web/2026-10-16/doc_m.html", "<html><body><div>" + "".join(f"<p><a href='/{i}'>Link number {i}</a></p>" for i in range(80)) + "</div></body></html>")
    assert "mostly_links_or_menus" in process.process_message(message("doc_m", menu), tmp_path)["reasons"]
    legacy = archive(tmp_path, "pdf/2026-10-16/doc_g.txt", ("Ì¸¢ÕõÀ¢ø §¾º¢Âý ¿¡ðÎ º¢ÕÀ¡ðÎ ¸Ä¡º¡Ãõ «È¢×õ ¦¸¡ñÎ ") * 60)
    r = process.process_message(message("doc_g", legacy, source="pdf", mime_type="text/plain"), tmp_path)
    assert r["skipped"] and r["reasons"] == ["language_unknown"]


def test_process_message_tamil_text_and_wikidata_json(tmp_path):
    ta = archive(tmp_path, "web/2026-10-16/doc_ta.txt", TAMIL)
    r = process.process_message(message("doc_ta", ta, mime_type="text/plain", language="ta"), tmp_path)
    assert not r["skipped"] and r["language"] == "ta" and r["records"][0]["language"] == "ta"
    js = archive(tmp_path, "dataset/2026-10-16/doc_wd.json", json.dumps(WIKIDATA))
    w = process.process_message(message("doc_wd", js, source="dataset", mime_type="application/json", topic="structured", site="wikidata"), tmp_path)
    assert not w["skipped"] and len(w["records"]) == 1 and "member of the party" in w["text"]
    assert {r["type"] for r in w["relations"]} >= {"MEMBER_OF_PARTY", "HELD_POSITION", "BORN_IN"} and all(r["passage_id"] == w["records"][0]["passage_id"] for r in w["relations"])
    assert process.process_message(message("doc_x", archive(tmp_path, "dataset/2026-10-16/doc_x.json", '{"a": 1}'), source="dataset", mime_type="application/json"), tmp_path)["skipped"]


def test_old_messages_without_event_type_still_work_and_keep_their_candidates():
    from backend.pipeline.common import make_message
    msg = make_message(site="dvac", doc_type="press_release", title="t", url="https://u", publisher="DVAC", licence="l", published_date="2026-10-12",
                       text="DVAC press release dated 2026-10-12: Tr.A.Test was arrested on 12.10.2026 for demanding a bribe of Rs.20,000/-.")
    r = process.process_message(msg)
    assert not r["skipped"] and "records" not in r and any(c["stage"] == "arrest" for p in r["passages"] for c in p["candidates"])


# ---- Spark batch logic without Spark ---------------------------------------------------------------------------------
class FakeWriter:
    def __init__(self):
        self.calls = []

    def __call__(self, groups):
        self.calls.append({k: list(v) for k, v in groups.items()})

    @property
    def rows(self):
        return [row for call in self.calls for rows in call.values() for row in rows]


def graph_counts(kb):
    return {k: kb._read(q)[0]["n"] for k, q in {
        "docs": "MATCH (s:Source:Document) RETURN count(s) AS n", "passages": "MATCH (p:Passage) RETURN count(p) AS n",
        "has": "MATCH (:Source)-[r:HAS_PASSAGE]->(:Passage) RETURN count(r) AS n", "from": "MATCH (:Passage)-[r:FROM]->(:Source) RETURN count(r) AS n",
        "entities": "MATCH (e:Entity) RETURN count(e) AS n", "mentions": "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS n",
        "relations": "MATCH ()-[r:RELATION]->() RETURN count(r) AS n"}.items()}


def test_batch_is_idempotent_in_parquet_rows_and_graph(kb, tmp_path):
    raw_dir = tmp_path / "raw"
    page = archive(raw_dir, "web/2026-10-16/doc_aaa.html", WIKI_HTML)
    wd = archive(raw_dir, "dataset/2026-10-16/doc_wd.json", json.dumps(WIKIDATA))
    msgs = [message("doc_aaa", page), message("doc_wd", wd, source="dataset", mime_type="application/json", topic="structured"),
            message("doc_bad", archive(raw_dir, "web/2026-10-16/doc_bad.html", "<p>tiny</p>"))]
    out_dir = tmp_path / "corpus"
    writer = FakeWriter()
    line = spark_job.handle_results([process.process_message(m, raw_dir) for m in msgs], kb, corpus.PassageIndex(out_dir), writer, out_dir,
                                    batch_id=7, offsets={"0": 12, "1": 3})
    assert (line["processed"], line["skipped"], line["failed"]) == (2, 1, 0) and line["passages"] == len(writer.rows) > 1
    assert line["offsets"] == {"0": 12, "1": 3} and sorted(line["doc_ids"]) == ["doc_aaa", "doc_wd"] and line["finished_at"]
    assert line["clean_tokens"] == sum(r[corpus.COLUMN_NAMES.index("token_estimate")] for r in writer.rows)
    assert set(writer.calls[0]) == {("web", "2026-10-16"), ("dataset", "2026-10-16")}
    flags = corpus.COLUMN_NAMES.index("embedded"), corpus.COLUMN_NAMES.index("indexed_neo4j")
    assert all(r[flags[0]] is False and r[flags[1]] is True for r in writer.rows)
    stats = [json.loads(x) for x in next((out_dir / "_stats").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert len(stats) == 1 and stats[0]["passages"] == line["passages"]
    before = graph_counts(kb)
    assert before["docs"] == 2 and before["passages"] == before["has"] == before["from"] == line["passages"] and before["relations"] >= 6 and before["mentions"] > 0
    assert kb._read("MATCH ()-[r:RELATION {method:'wikidata'}]->() WHERE r.assertion_level <> 'reported' RETURN count(r) AS n")[0]["n"] == 0
    assert kb.audit() == []

    again = FakeWriter()                                                                  # the same messages arrive again (a retried batch)
    line2 = spark_job.handle_results([process.process_message(m, raw_dir) for m in msgs], kb, corpus.PassageIndex(out_dir), again, out_dir)
    assert again.rows == [] and line2["passages"] == 0 and line2["processed"] == 2
    assert graph_counts(kb) == before


def test_a_failed_parquet_write_leaves_no_index_entries(kb, tmp_path):
    raw_dir = tmp_path / "raw"
    m = message("doc_aaa", archive(raw_dir, "web/2026-10-16/doc_aaa.html", WIKI_HTML))

    def boom(groups):
        raise OSError("disk full")
    idx = corpus.PassageIndex(tmp_path / "corpus")
    with pytest.raises(OSError):
        spark_job.handle_results([process.process_message(m, raw_dir)], kb, idx, boom, tmp_path / "corpus")
    assert not idx.pending and not idx.ids and not (tmp_path / "corpus" / "_state" / "passage_index.tsv").exists()
    ok = FakeWriter()
    spark_job.handle_results([process.process_message(m, raw_dir)], kb, corpus.PassageIndex(tmp_path / "corpus"), ok, tmp_path / "corpus")
    assert ok.rows                                                                         # the retry writes the rows after all


def test_court_finding_needs_a_judgment_and_never_comes_from_cooccurrence(kb):
    doc = {"doc_id": "doc_j", "doc_type": "reference", "source": "web", "source_url": "u"}
    kb.add_document_graph({**doc, "title": "t", "created_at": "2026-10-16"}, [])
    rel = {"type": "HELD_POSITION", "from": {"type": "Person", "name": "Aravind Kesavan"}, "to": {"type": "Position", "name": "Minister"},
           "method": "wikidata", "confidence": "high", "assertion_level": "court_finding"}
    with pytest.raises(ValueError):
        kb.add_relations([rel], doc)
    judgment = {**doc, "doc_type": "judgment"}
    kb.add_relations([{**rel, "assertion_level": None}], judgment)
    assert kb._read("MATCH ()-[r:RELATION]->() RETURN r.assertion_level AS l")[0]["l"] == "court_finding"
    with pytest.raises(ValueError):
        kb.add_relations([{**rel, "type": "MENTIONED_WITH", "method": "cooccurrence", "assertion_level": "court_finding"}], judgment)


# ---- check_stores ---------------------------------------------------------------------------------------------------------
def test_check_stores_embeds_indexes_updates_flags_and_reports(kb, rag_store, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from backend.knowledge import check_stores
    raw_dir = tmp_path / "raw"
    recs = process.process_message(message("doc_aaa", archive(raw_dir, "web/2026-10-16/doc_aaa.html", WIKI_HTML)), raw_dir)["records"]
    types = {"string": pa.string(), "int": pa.int32(), "bool": pa.bool_(), "double": pa.float64()}
    schema = pa.schema([(n, types[t]) for n, t in corpus.PASSAGE_COLUMNS])
    folder = corpus.partition_dir(tmp_path / "corpus", "web", "2026-10-16")
    folder.mkdir(parents=True)
    rows = [{c: r.get(c) for c in corpus.COLUMN_NAMES} for r in recs]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), folder / "part-0.parquet")
    pq.write_table(pa.Table.from_pylist(rows[:1], schema=schema), folder / "part-1.parquet")           # a retried batch wrote one row twice
    corpus.append_stats(tmp_path / "corpus", "2026-10-16", {"processed": 1, "skipped": 2, "failed": 1, "duplicates_removed": 3, "passages": len(rows), "clean_tokens": 1})
    report = check_stores.run(tmp_path / "corpus", raw_dir, kb=kb, rag=rag_store)
    n = len(rows)
    assert report["passages"] == n and report["extracted_documents"] == 1 and report["embedded_passages"] == n and report["neo4j_indexed_passages"] == n
    assert report["clean_tokens"] == sum(r["token_estimate"] for r in rows) and report["duplicates_removed"] == 3 + 1
    assert report["failed_extractions"] == 3 and report["raw_files"] == 1 and "unavailable" not in report
    assert rag_store.count() == n and graph_counts(kb)["passages"] == n and kb.audit() == []
    table = pq.read_table(folder / "part-0.parquet")
    assert table.schema == schema and all(table["embedded"].to_pylist()) and all(table["indexed_neo4j"].to_pylist())
    assert pq.read_table(folder / "part-1.parquet").num_rows == 0 and not list(folder.glob("*.tmp"))
    assert json.loads((tmp_path / "corpus" / "_stats" / "report.json").read_text(encoding="utf-8")) == report
    assert check_stores.run(tmp_path / "corpus", raw_dir, kb=kb, rag=rag_store) == report         # a second night changes nothing
    assert graph_counts(kb)["passages"] == n


def test_check_stores_without_the_stores_still_reports_the_corpus(tmp_path):
    pytest.importorskip("pyarrow")
    from backend.knowledge import check_stores
    report = check_stores.run(tmp_path / "corpus", tmp_path / "raw", kb=None, rag=None)
    assert report["passages"] == 0 and set(report["unavailable"]) == {"neo4j", "chroma"}


def test_the_same_page_in_another_form_is_not_stored_twice_and_persons_are_linked_to_their_documents(kb):
    older = {"doc_id": "doc_old", "doc_type": "reference", "source": "web", "source_url": "https://en.wikipedia.org/wiki/Fictional_case",
             "title": "Fictional case (Wikipedia)", "sha256": "hash-old", "person_key": "wd:Q9001"}
    rec = {"passage_id": "p-old-0", "position": 0, "text": "A fictional case in Tamil Nadu.", "token_estimate": 8, "sha256": "ph0"}
    did, created = kb.add_document_graph(older, [rec])
    assert created
    newer = {**older, "doc_id": "doc_new", "sha256": "hash-new"}                           # cleaned another way: different text hash, same page
    again, created2 = kb.add_document_graph(newer, [{**rec, "passage_id": "p-new-0", "text": "A fictional case in Tamil Nadu (cleaned differently)."}])
    assert again == did and created2 is False
    assert kb._read("MATCH (s:Source) RETURN count(s) AS n")[0]["n"] == 1
    assert kb._read("MATCH (p:Passage) RETURN count(p) AS n")[0]["n"] == 1
    kb.entity("Person", "Arulmozhi Vendhan")
    kb._run("MATCH (e:Entity {key:'Person:arulmozhi vendhan'}) SET e.person_key='wd:Q9001'")
    pairs = [("wd:Q9001", "doc_new", older["source_url"]), ("wd:Q9001", "doc_missing", None), ("wd:Q404", "doc_old", None)]
    assert kb.link_person_documents(pairs) == [("wd:Q9001", "doc_new")]                       # found through the url; the others have no person or no document
    assert kb.link_person_documents(pairs) == [("wd:Q9001", "doc_new")]                       # and again: nothing is added twice
    assert kb._read("MATCH (:Entity {person_key:'wd:Q9001'})-[r:HAS_DOCUMENT]->(:Source) RETURN count(r) AS n")[0]["n"] == 1
