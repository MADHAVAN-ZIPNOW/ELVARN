"""A typed request for one person and some sources, for example  "collect C. Joseph Vijay ED cases"  ->  that person, the ED source only.

    parse(text)                  -> (name words, source groups)      "collect C. Joseph Vijay ED cases" -> ("C. Joseph Vijay", ["ed"])
    resolve(registry, name)      -> the roster people the name could mean (one entry: go; several: the owner picks; none: Wikidata is asked)

Source groups (what the owner can ask for) and the collector stages each one runs:

    wikidata  -> wikidata                      courts -> court_records (Supreme Court, Madras High Court)
    news      -> news_records (headlines and links of news reports)
    wikipedia -> wikipedia_en, wikipedia_ta    ed     -> ed_records     dvac -> dvac_records     cbi -> cbi_records
    articles  -> corruption_articles (Wikipedia pages about cases and scandals)
    all       -> every stage (the default when the request names no source)
"""
import json
import re
import urllib.parse
import urllib.request

from backend.pipeline.official_index import HONORIFICS, norm
from backend.pipeline.roster import UA

GROUPS = {
    "wikidata": ("wikidata",),
    "wikipedia": ("wikipedia_en", "wikipedia_ta"),
    "articles": ("corruption_articles",),
    "courts": ("court_records",),
    "ed": ("ed_records",),
    "dvac": ("dvac_records",),
    "cbi": ("cbi_records",),
    "news": ("news_records",),
}
ALL_STAGES = ("wikidata", "wikipedia_en", "wikipedia_ta", "corruption_articles", "court_records", "ed_records", "dvac_records", "cbi_records", "news_records")

# phrase -> group. A group is asked for when ANY of its phrases is in the request; the words are removed from the name.
KEYWORDS = (
    ("ed", r"\bed\b|\benforcement directorate\b|\bdirectorate of enforcement\b|\bpmla\b"),
    ("cbi", r"\bcbi\b|\bcentral bureau of investigation\b"),
    ("dvac", r"\bdvac\b|\bvigilance\b|\banti[- ]corruption\b"),
    ("courts", r"\bcourts?\b|\bjudge?ments?\b|\bhigh court\b|\bsupreme court\b|\bhc\b|\bsc\b|\bverdicts?\b"),
    ("wikipedia", r"\bwikipedia\b|\bwiki\b"),
    ("wikidata", r"\bwikidata\b|\brelationships?\b"),
    ("articles", r"\barticles?\b|\bscandals?\b|\bscams?\b"),
    ("news", r"\bnews\b|\bheadlines?\b|\bmedia\b|\bnewspapers?\b"),
)
FILLER = re.compile(r"\b(collect|collection|get|fetch|find|download|gather|please|pls|all|the|of|for|his|her|their|about|on|from|only|data|records?|"
                    r"cases?|corruption|documents?|docs?|material|news|release|releases|press|and|with)\b", re.I)


def parse(text):
    """-> (name, groups). `groups` is [] when the request names no source (meaning all)."""
    low = (text or "").lower()
    groups = [g for g, rx in KEYWORDS if re.search(rx, low)]
    rest = low
    for _, rx in KEYWORDS:
        rest = re.sub(rx, " ", rest)
    rest = FILLER.sub(" ", rest)
    name = re.sub(r"\s+", " ", re.sub(r"[^\w.\s-]", " ", rest)).strip()
    return name, groups


def stages_for(groups):
    """The collector stages (in their normal order) that the requested groups cover; every stage when none were named."""
    if not groups or "all" in groups:
        return list(ALL_STAGES)
    wanted = {s for g in groups for s in GROUPS[g]}
    return [s for s in ALL_STAGES if s in wanted]


def _hay(p):
    return norm(f"{p['name']} {p.get('en_title') or ''}")


def resolve(registry, name, limit=8):
    """Roster people the name could mean, best first. Every word of the request must appear in the person's name (spelling variants folded)."""
    words = [w for w in norm(name).split() if w not in HONORIFICS and len(w) >= 2]
    if not words:
        return []
    total, people = registry.people(limit=100000)
    hits = [p for p in people if all(w in _hay(p) for w in words)]
    hits.sort(key=lambda p: (len(_hay(p)) - sum(len(w) for w in words), p["priority"], p["first_year"] or 9999))       # the closest name first
    return hits[:limit]


def wikidata_candidates(name, limit=5):
    """People Wikidata knows under this name (for someone who is not on the roster yet): [{qid, name, description}]."""
    url = ("https://www.wikidata.org/w/api.php?action=wbsearchentities&language=en&type=item&format=json&limit=%d&search=%s"
           % (limit, urllib.parse.quote(name)))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        found = json.load(resp).get("search", [])
    return [{"qid": f["id"], "name": f.get("label") or f["id"], "description": f.get("description") or ""} for f in found]


def add_from_wikidata(registry, qid):
    """Put a person who is not on the roster (found by name) into it, with the English and Tamil Wikipedia titles. Returns the person row."""
    url = ("https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=labels|sitelinks&languages=en|ta&sitefilter=enwiki|tawiki&ids=" + qid)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        ent = json.load(resp)["entities"][qid]
    links = ent.get("sitelinks") or {}
    name = ((ent.get("labels") or {}).get("en") or {}).get("value") or (links.get("enwiki") or {}).get("title") or qid
    registry.upsert_person({"key": "wd:" + qid, "wikidata_id": qid, "name": name, "party": None, "first_year": None, "last_year": None, "positions": [],
                            "en_title": (links.get("enwiki") or {}).get("title"), "ta_title": (links.get("tawiki") or {}).get("title"), "priority": 3})
    return registry.person("wd:" + qid)


def _card(p, **extra):
    return {"key": p["key"], "name": p["name"], "party": p.get("party"), "first_year": p.get("first_year"), "last_year": p.get("last_year"), **extra}


def plan(registry, text=None, person=None, sources=None, lookup=wikidata_candidates, add=add_from_wikidata):
    """Turn a request into what to run. `text` is the typed sentence; `person` (a roster key such as wd:Q123) and `sources` (group names) are what the
    owner picked from a list of candidates, and override what the sentence said.
    -> {"ok": True, "person": card, "sources": [...], "stages": [...]}                                 run it
       {"ok": False, "needs_choice": True, "candidates": [card...], "sources": [...], "message"}      ask the owner which person is meant
       {"ok": False, "message"}                                                                        nothing to run"""
    name, said = parse(text) if text else ("", [])
    groups = [g for g in (sources or said) if g in GROUPS or g == "all"] or ["all"]
    chosen = None
    if person:
        chosen = registry.person(person)
        if chosen is None and person.startswith("wd:Q"):
            chosen = add(registry, person[3:])
    else:
        if not name:
            return {"ok": False, "message": "Type a name, for example: collect C. Joseph Vijay ED cases"}
        hits = resolve(registry, name)
        if len(hits) == 1:
            chosen = hits[0]
        elif hits:
            return {"ok": False, "needs_choice": True, "sources": groups, "candidates": [_card(p) for p in hits],
                    "message": f"{len(hits)} politicians match \"{name}\". Pick one."}
        else:
            found = lookup(name)
            if len(found) == 1:
                chosen = add(registry, found[0]["qid"])
            elif found:
                return {"ok": False, "needs_choice": True, "sources": groups, "message": f"\"{name}\" is not on the roster. Pick the right person from Wikidata.",
                        "candidates": [{"key": "wd:" + f["qid"], "name": f["name"], "description": f["description"], "new": True} for f in found]}
            else:
                return {"ok": False, "message": f"No politician found for \"{name}\"."}
    if chosen is None:
        return {"ok": False, "message": "That person is not known."}
    return {"ok": True, "person": _card(chosen), "sources": groups, "stages": stages_for(groups)}
