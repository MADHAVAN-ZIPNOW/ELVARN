"""The roster of Tamil Nadu politicians the collector works through, 1947 to 2026, built from Wikidata (CC0).

Who is on it: every person holding a position of Tamil Nadu (or of the Madras State it was before 1969) in Wikidata:
chief ministers, ministers and speakers, and members of the Madras / Tamil Nadu legislative assemblies. Each person gets the
English and Tamil Wikipedia titles (when they exist), the party, and the first and last year they held a position.

Order of work (`priority`): chief ministers (0), then ministers and speakers (1), then members of the assembly (2); inside a group by the
first year they held office. People with no dated term are kept (Wikidata often leaves the dates out) and sort last in their group.

    python -m backend.pipeline.roster            # build or refresh the roster in the collector's database
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from backend.pipeline.registry import Registry

SPARQL = "https://query.wikidata.org/sparql"
UA = "ELVARN-learning-project/1.0 (personal research project; madhavan.ai.research@gmail.com)"      # the project's own contact, as in collect_wikipedia
TAMIL_NADU = "Q1445"
CHIEF_MINISTER = "Q24040821"
MLA = "Q24558764"
MIN_YEAR, MAX_YEAR = 1947, 2026
POSITION_LABEL = re.compile(r"^(Member of the .*(Legislative (Assembly|Council))|.*\bMinister\b.*|.*\bspeaker\b.*|Chief Minister.*)", re.I)
NOT_A_POSITION = re.compile(r"\b(Department|Act|Order|Commission|by-elections?|division)\b", re.I)


def _sparql(query, tries=4, pause=4):
    url = SPARQL + "?format=json&query=" + urllib.parse.quote(query)
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)["results"]["bindings"]
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"Wikidata did not answer: {last}")


def _val(row, name):
    return row.get(name, {}).get("value")


def _qid(uri):
    return uri.rsplit("/", 1)[-1] if uri else None


def _year(iso):
    m = re.match(r"-?(\d{4})", iso or "")
    return int(m.group(1)) if m else None


def priority_of(label, qid):
    if qid == CHIEF_MINISTER or re.search(r"chief minister", label or "", re.I):
        return 0
    if re.search(r"minister|speaker", label or "", re.I):
        return 1
    return 2


def positions():
    """[(qid, label, priority)] of Tamil Nadu positions that people hold (not departments, acts or by-elections)."""
    rows = _sparql('SELECT ?pos ?posLabel WHERE { ?pos wdt:P1001 wd:%s . SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }' % TAMIL_NADU)
    found = {CHIEF_MINISTER: "Chief Minister of Tamil Nadu", MLA: "Member of the Tamil Nadu Legislative Assembly"}
    for r in rows:
        qid, label = _qid(_val(r, "pos")), _val(r, "posLabel") or ""
        if re.fullmatch(r"Q\d+", label) or NOT_A_POSITION.search(label) or not POSITION_LABEL.match(label):
            continue
        found.setdefault(qid, label)
    return [(qid, label, priority_of(label, qid)) for qid, label in found.items()]


def holders(qid):
    """Rows (person, label, enwiki, tawiki, party, start, end) of everyone who held position `qid`."""
    query = ("SELECT ?p ?pLabel ?enwiki ?tawiki ?partyLabel ?start ?end WHERE { ?p p:P39 ?st . ?st ps:P39 wd:%s . "
             "OPTIONAL { ?st pq:P580 ?start } OPTIONAL { ?st pq:P582 ?end } OPTIONAL { ?p wdt:P102 ?party } "
             "OPTIONAL { ?enwiki schema:about ?p ; schema:isPartOf <https://en.wikipedia.org/> } "
             "OPTIONAL { ?tawiki schema:about ?p ; schema:isPartOf <https://ta.wikipedia.org/> } "
             'SERVICE wikibase:label { bd:serviceParam wikibase:language "en,ta". } }' % qid)
    return _sparql(query)


def _title(url):
    return urllib.parse.unquote(url.rsplit("/wiki/", 1)[-1]).replace("_", " ") if url else None


def merge(people, rows, pos_label, priority):
    """Fold the SPARQL rows of one position into `people` (key -> entry). A person may hold several positions and have several parties."""
    for r in rows:
        qid = _qid(_val(r, "p"))
        name = _val(r, "pLabel")
        en, ta = _title(_val(r, "enwiki")), _title(_val(r, "tawiki"))
        if not qid or (not name or re.fullmatch(r"Q\d+", name)):
            name = en or ta
        if not name:
            continue                                            # nobody to search for
        start, end = _year(_val(r, "start")), _year(_val(r, "end"))
        if (end is not None and end < MIN_YEAR) or (start is not None and start > MAX_YEAR):
            continue                                            # outside 1947 to 2026
        p = people.setdefault("wd:" + qid, {"key": "wd:" + qid, "wikidata_id": qid, "name": name, "party": None, "first_year": None,
                                            "last_year": None, "positions": [], "en_title": en, "ta_title": ta, "priority": priority})
        p["en_title"], p["ta_title"] = p["en_title"] or en, p["ta_title"] or ta
        p["priority"] = min(p["priority"], priority)
        if _val(r, "partyLabel") and not re.fullmatch(r"Q\d+", _val(r, "partyLabel")):
            p["party"] = p["party"] or _val(r, "partyLabel")
        if pos_label not in p["positions"]:
            p["positions"].append(pos_label)
        if start:
            p["first_year"] = start if p["first_year"] is None else min(p["first_year"], start)
        last = end or (start and MAX_YEAR)
        if last:
            p["last_year"] = last if p["last_year"] is None else max(p["last_year"], last)
    return people


def build(registry=None, progress=print):
    """Fetch every position's holders and store the roster. Returns the number of people. Existing progress (done / in progress) is kept."""
    registry = registry or Registry()
    people = {}
    for qid, label, priority in positions():
        try:
            rows = holders(qid)
        except RuntimeError as exc:
            progress(f"skipped {label}: {exc}")
            continue
        merge(people, rows, label, priority)
        progress(f"{label}: {len(rows)} rows, {len(people)} people so far")
    for p in people.values():
        p["positions"] = p["positions"][:12]
        registry.upsert_person(p)
    if people:
        snapshot = registry.path.with_name("roster.json")
        snapshot.write_text(json.dumps(sorted(people.values(), key=lambda p: (p["priority"], p["first_year"] or 9999, p["name"])), ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return len(people)


def main():
    print(f"roster: {build()} people")


if __name__ == "__main__":
    main()
