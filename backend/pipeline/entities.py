"""Rule-based entities and relations for the corpus passages (standard library only, Python 3.8 compatible).

No model is involved: people come from the roster (names and aliases the collector already knows) plus a conservative
pattern (an honorific followed by capitalised words), organisations / parties / agencies / courts from a dictionary, places
from the Tamil Nadu district list, dates and years by pattern, cases by "<Name> case|scam|scandal", references by URL.

Assertion levels (the project rule): a fact from a reference work is "reported" at most; "court_finding" only when the
document is a judgment or an order. A person and a case that merely appear in one passage are linked with the relation
MENTIONED_WITH (method "cooccurrence", confidence "low"): that is never an accusation, so its level never goes above "reported".
Wikidata records (Special:EntityData JSON) become structured relations with method "wikidata".
"""
import re

from backend.knowledge import review
from backend.knowledge.kb import assertion_level  # noqa: F401  (the rule lives next to the graph writes that enforce it)

# ---- dictionaries ---------------------------------------------------------------------------------------------------------
# (regex, type, canonical name, aliases, case_sensitive). Acronyms are case sensitive so "ed" or "cbi" in a word never match.
ORG_TERMS = [
    (r"\bDMK\b|Dravida Munnetra Kazhagam", "PoliticalParty", "Dravida Munnetra Kazhagam", ["DMK"], True),
    (r"\bAIADMK\b|All India Anna Dravida Munnetra Kazhagam", "PoliticalParty", "All India Anna Dravida Munnetra Kazhagam", ["AIADMK"], True),
    (r"\bPMK\b|Pattali Makkal Katchi", "PoliticalParty", "Pattali Makkal Katchi", ["PMK"], True),
    (r"\bMDMK\b", "PoliticalParty", "Marumalarchi Dravida Munnetra Kazhagam", ["MDMK"], True),
    (r"\bDMDK\b", "PoliticalParty", "Desiya Murpokku Dravida Kazhagam", ["DMDK"], True),
    (r"\bVCK\b", "PoliticalParty", "Viduthalai Chiruthaigal Katchi", ["VCK"], True),
    (r"\bBJP\b|Bharatiya Janata Party", "PoliticalParty", "Bharatiya Janata Party", ["BJP"], True),
    (r"Indian National Congress|Tamil Nadu Congress Committee", "PoliticalParty", "Indian National Congress", [], False),
    (r"Communist Party of India \(Marxist\)|\bCPI\(M\)", "PoliticalParty", "Communist Party of India (Marxist)", ["CPI(M)"], True),
    (r"Directorate of Vigilance and Anti-Corruption|\bDVAC\b", "InvestigatingAgency", "Directorate of Vigilance and Anti-Corruption", ["DVAC"], True),
    (r"Central Bureau of Investigation|\bCBI\b", "InvestigatingAgency", "Central Bureau of Investigation", ["CBI"], True),
    (r"Enforcement Directorate|\bED\b", "InvestigatingAgency", "Enforcement Directorate", ["ED"], True),
    (r"Income Tax Department", "InvestigatingAgency", "Income Tax Department", [], False),
    (r"Central Vigilance Commission", "InvestigatingAgency", "Central Vigilance Commission", [], False),
    (r"\bLokayukta\b", "InvestigatingAgency", "Lokayukta", [], True),
    (r"Comptroller and Auditor General|\bCAG\b", "InvestigatingAgency", "Comptroller and Auditor General", ["CAG"], True),
    (r"Madras High Court|High Court of Madras|Madurai Bench", "Court", "Madras High Court", [], False),
    (r"Supreme Court(?: of India)?", "Court", "Supreme Court of India", ["Supreme Court"], False),
    (r"Special Court", "Court", "Special Court", [], False),
    (r"Sessions Court", "Court", "Sessions Court", [], False),
    (r"Tamil Nadu Legislative Assembly|Madras Legislative Assembly", "Organisation", "Tamil Nadu Legislative Assembly", [], False),
    (r"Government of Tamil Nadu", "Organisation", "Government of Tamil Nadu", [], False),
    (r"\bTANGEDCO\b", "Organisation", "TANGEDCO", [], True),
    (r"\bTNPSC\b", "Organisation", "Tamil Nadu Public Service Commission", ["TNPSC"], True),
    (r"\bTASMAC\b", "Organisation", "TASMAC", [], True),
]
_ORG_RES = [(re.compile(rx, 0 if cs else re.I), t, name, aliases) for rx, t, name, aliases, cs in ORG_TERMS]

PLACES = ["Tamil Nadu", "Chennai", "Madras", "Madurai", "Coimbatore", "Tiruchirappalli", "Tiruchi", "Trichy", "Salem", "Tirunelveli",
          "Tiruppur", "Erode", "Vellore", "Thanjavur", "Nilgiris", "Kancheepuram", "Chengalpattu", "Chengalpet", "Tiruvallur",
          "Kanniyakumari", "Kanyakumari", "Dindigul", "Namakkal", "Virudhunagar", "Sivaganga", "Ramanathapuram", "Thoothukudi",
          "Tuticorin", "Cuddalore", "Villupuram", "Kallakurichi", "Karur", "Ariyalur", "Perambalur", "Pudukkottai", "Theni",
          "Krishnagiri", "Dharmapuri", "Tiruvannamalai", "Tenkasi", "Ranipet", "Tirupathur", "Tiruvarur", "Nagapattinam",
          "Mayiladuthurai", "Puducherry"]
_PLACE_RE = re.compile(r"(?<!\w)(" + "|".join(sorted(PLACES, key=len, reverse=True)) + r")(?!\w)")

HONORIFIC = (r"(?:Thiru|Tmt|Dr|Mr|Mrs|Ms|Shri|Sri|Selvi|Thirumathi|Justice|Hon'ble|Tr|Chief Minister|Former Chief Minister|"
             r"Minister|Deputy Chief Minister|Leader of Opposition)")
_PERSON_RE = re.compile(r"\b" + HONORIFIC + r"\.?\s+((?:[A-Z][A-Za-z.'’-]*)(?:\s+[A-Z][A-Za-z.'’-]*){0,2})")
NOT_NAME = {"The", "A", "An", "In", "On", "At", "He", "She", "It", "His", "Her", "They", "This", "That", "Government", "Court", "High",
            "Supreme", "Party", "Police", "Minister", "Chief", "Special", "Case", "Act", "Section", "Sessions", "Bench", "Assembly",
            "Directorate", "Bureau", "Commission", "Department", "And", "Of", "For", "To", "Was", "Is", "Said", "Who", "Under"} | set(PLACES)

_CASE_FIXED = re.compile(r"\b(disproportionate assets|2G spectrum|money[- ]laundering|bribery|tax evasion|land grab|illegal mining|"
                         r"transport corruption|egg scam|coal)\s+(case|scam|scandal)\b", re.I)
_CASE_NAMED = re.compile(r"((?:(?:[A-Z][\w'’.-]*|\d+G)\s+){1,4})(case|scam|scandal)\b")
CASE_STOP = {"The", "A", "An", "In", "On", "At", "This", "That", "His", "Her", "Their", "Of", "For", "After", "Before", "During", "Since",
             "When", "While", "Criminal", "Civil", "Another", "Same", "Said", "Such", "Court", "High", "Supreme", "No", "Any", "Each"}
_URL_RE = re.compile(r"https?://[^\s<>\")\]]+")
_YEAR_RE = re.compile(r"(?<![\w-])(19[4-9]\d|20[0-2]\d)(?![\w-])")
MAX_DATES = 8                                               # per passage: years and dates are many, and each is a node


# ---- text extraction ------------------------------------------------------------------------------------------------------
def _roster_list(roster):
    out = []
    for r in roster or ():
        if isinstance(r, str):
            r = {"name": r}
        if r.get("name"):
            out.append(r)
    return out


def _ent(type_, name, aliases=(), **extra):
    e = {"type": type_, "name": name, "aliases": sorted({a for a in aliases if a and a != name})}
    e.update({k: v for k, v in extra.items() if v})
    return e


def find_dates(text):
    """ISO dates found in the text (2018-03-10, 12.10.2026, 5 January 2021, January 5, 2021), in order, without repeats."""
    found = []
    for m in re.finditer(r"\b((?:1[89]|20)\d\d)-(\d\d)-(\d\d)\b", text):
        found.append((m.start(), f"{m.group(1)}-{m.group(2)}-{m.group(3)}"))
    for rx, parts in review.DATE_RES:
        for m in rx.finditer(text):
            y, mo, d = parts(m)
            try:
                found.append((m.start(), "%04d-%02d-%02d" % (int(y), int(mo), int(d))))
            except ValueError:
                pass
    out = []
    for _, d in sorted(found):
        if d not in out:
            out.append(d)
    return out


def extract_entities(text, roster=()):
    """Entities mentioned in one passage: [{type, name, aliases, (person_key, wikidata_id)}], without repeats.
    `roster` = known people: names or {name, aliases, person_key, wikidata_id}."""
    found = {}

    def add(e):
        found.setdefault((e["type"], " ".join(e["name"].lower().split())), e)

    for r in _roster_list(roster):
        names = [r["name"]] + list(r.get("aliases") or [])
        for n in names:
            if len(n) >= 4 and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text, re.I):
                add(_ent("Person", r["name"], names, person_key=r.get("person_key"), wikidata_id=r.get("wikidata_id")))
                break
    for m in _PERSON_RE.finditer(text):
        words = []
        for w in m.group(1).split():
            if w.strip(".") in NOT_NAME:
                break
            words.append(w.rstrip(".,"))
        if words and len(" ".join(words)) >= 3:
            add(_ent("Person", " ".join(words)))
    for rx, type_, name, aliases in _ORG_RES:
        if rx.search(text):
            add(_ent(type_, name, aliases))
    for m in _PLACE_RE.finditer(text):
        add(_ent("Place", m.group(1)))
    for m in _CASE_FIXED.finditer(text):
        add(_ent("Case", f"{m.group(1)[0].upper()}{m.group(1)[1:]} {m.group(2).lower()}"))
    for m in _CASE_NAMED.finditer(text):
        words = m.group(1).split()
        while words and words[0] in CASE_STOP:
            words.pop(0)
        if words:
            add(_ent("Case", f"{' '.join(words)} {m.group(2)}"))
    dates = find_dates(text)
    for d in dates:
        add(_ent("Date", d))
    for y in _YEAR_RE.findall(text):
        if not any(d.startswith(y) for d in dates):
            add(_ent("Date", y))
    for url in _URL_RE.findall(text):
        add(_ent("Reference", url.rstrip(".,;")))
    out = list(found.values())
    dates_seen = 0
    kept = []
    for e in out:                                           # years and dates would otherwise swamp the graph
        if e["type"] == "Date":
            dates_seen += 1
            if dates_seen > MAX_DATES:
                continue
        kept.append(e)
    return kept


def _ref(e):
    return {"type": e["type"], "name": e["name"]}


def cooccurrence_relations(entities):
    """Person and Case in the same passage -> MENTIONED_WITH (method cooccurrence, confidence low). Nothing else is inferred."""
    people = [e for e in entities if e["type"] == "Person"]
    cases = [e for e in entities if e["type"] == "Case"]
    return [{"type": "MENTIONED_WITH", "from": _ref(p), "to": _ref(c), "method": "cooccurrence", "confidence": "low",
             "start": None, "end": None, "kind": None} for p in people for c in cases]


# ---- Wikidata (Special:EntityData JSON) -----------------------------------------------------------------------------------
MALE, FEMALE = "Q6581097", "Q6581072"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _label_of(obj, langs=("en", "ta")):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for lang in langs:
            v = obj.get(lang)
            if isinstance(v, dict) and v.get("value"):
                return v["value"]
            if isinstance(v, str) and v:
                return v
        if obj.get("value"):
            return obj["value"]
    return None


def wikidata_labels(entity_json):
    """{QID: label} for every item the JSON can name: the entities' own labels, and an optional top-level "labels" map
    ({"Q10225": "DMK"}) that the collector can add after resolving the linked items (Special:EntityData holds ids only)."""
    labels = {}
    for qid, ent in (entity_json.get("entities") or {}).items():
        label = _label_of(ent.get("labels"))
        if label:
            labels[qid] = label
    for qid, v in (entity_json.get("labels") or {}).items():
        label = _label_of(v)
        if label:
            labels[qid] = label
    return labels


def _item_id(snak):
    v = ((snak or {}).get("datavalue") or {}).get("value")
    return v.get("id") if isinstance(v, dict) else None


def _time(snak):
    """ISO date (or year-month, or year) from a Wikidata time snak, honouring its precision; None when unknown."""
    v = ((snak or {}).get("datavalue") or {}).get("value")
    if not isinstance(v, dict) or not v.get("time"):
        return None
    m = re.match(r"([+-])(\d+)-(\d\d)-(\d\d)", v["time"])
    if not m or m.group(1) == "-":
        return None
    year, month, day, prec = m.group(2).lstrip("0").rjust(4, "0"), m.group(3), m.group(4), v.get("precision", 11)
    if prec >= 11 and month != "00" and day != "00":
        return f"{year}-{month}-{day}"
    if prec >= 10 and month != "00":
        return f"{year}-{month}"
    return year


def _human(date):
    if not date:
        return None
    p = date.split("-")
    if len(p) == 3:
        return f"{int(p[2])} {MONTHS[int(p[1]) - 1]} {p[0]}"
    if len(p) == 2:
        return f"{MONTHS[int(p[1]) - 1]} {p[0]}"
    return date


def _claims(entity, pid):
    return [c for c in (entity.get("claims") or {}).get(pid, []) if c.get("rank") != "deprecated"]


def _qual(claim, pid):
    snaks = (claim.get("qualifiers") or {}).get(pid) or []
    return snaks[0] if snaks else None


def parse_wikidata(entity_json):
    """The facts of the main item as plain data (labels resolved; items with no label are counted in 'unresolved')."""
    labels = wikidata_labels(entity_json)
    ents = entity_json.get("entities") or {}
    qid = next(iter(ents), None)
    ent = ents.get(qid) or {}
    unresolved = []

    def label(q):
        if q in labels:
            return labels[q]
        if q:
            unresolved.append(q)
        return None

    facts = {"qid": qid, "name": labels.get(qid), "description": _label_of(ent.get("descriptions"), ("en",)),
             "aliases": [a.get("value") for a in ((ent.get("aliases") or {}).get("en") or []) if a.get("value")],
             "gender": next((_item_id(c.get("mainsnak")) for c in _claims(ent, "P21")), None),
             "born": next((_time(c.get("mainsnak")) for c in _claims(ent, "P569") if _time(c.get("mainsnak"))), None),
             "died": next((_time(c.get("mainsnak")) for c in _claims(ent, "P570") if _time(c.get("mainsnak"))), None),
             "birthplace": next((label(_item_id(c.get("mainsnak"))) for c in _claims(ent, "P19") if _item_id(c.get("mainsnak"))), None),
             "parties": [], "positions": [], "relatives": [], "education": []}
    for c in _claims(ent, "P102"):
        name = label(_item_id(c.get("mainsnak")))
        if name:
            facts["parties"].append({"name": name, "start": _time(_qual(c, "P580")), "end": _time(_qual(c, "P582"))})
    for c in _claims(ent, "P39"):
        name = label(_item_id(c.get("mainsnak")))
        if name:
            facts["positions"].append({"name": name, "start": _time(_qual(c, "P580")), "end": _time(_qual(c, "P582")),
                                       "district": label(_item_id(_qual(c, "P768"))) if _qual(c, "P768") else None})
    for c in _claims(ent, "P1038"):
        name = label(_item_id(c.get("mainsnak")))
        if name:
            kind = label(_item_id(_qual(c, "P1039"))) if _qual(c, "P1039") else None
            facts["relatives"].append({"name": name, "kind": kind.lower() if kind else None})
    for c in _claims(ent, "P69"):
        name = label(_item_id(c.get("mainsnak")))
        if name:
            facts["education"].append({"name": name, "start": _time(_qual(c, "P580")), "end": _time(_qual(c, "P582"))})
    facts["unresolved"] = sorted(set(unresolved))
    return facts


def relations_from_wikidata(entity_json, person_key=None):
    """-> (entities, relations) from a raw Wikidata entity JSON. Relations carry method "wikidata", confidence "high" and the
    dates of the statement: MEMBER_OF_PARTY (P102), HELD_POSITION (P39), CONSTITUENCY (P768 qualifier of P39), RELATIVE_OF (P1038 with
    kind P1039), BORN_IN (P19), EDUCATED_AT (P69). Items whose label is unknown are left out (see parse_wikidata 'unresolved')."""
    f = parse_wikidata(entity_json)
    if not f["name"]:
        return [], []
    person = _ent("Person", f["name"], f["aliases"], person_key=person_key, wikidata_id=f["qid"])
    if f["born"]:
        person["birth_date"] = f["born"]
    if f["died"]:
        person["death_date"] = f["died"]
    ents, rels = {("Person", f["name"].lower()): person}, []

    def rel(type_, to_type, to_name, start=None, end=None, kind=None):
        ents.setdefault((to_type, to_name.lower()), _ent(to_type, to_name))
        rels.append({"type": type_, "from": _ref(person), "to": {"type": to_type, "name": to_name}, "method": "wikidata",
                     "confidence": "high", "start": start, "end": end, "kind": kind})

    for p in f["parties"]:
        rel("MEMBER_OF_PARTY", "PoliticalParty", p["name"], p["start"], p["end"])
    for p in f["positions"]:
        rel("HELD_POSITION", "Position", p["name"], p["start"], p["end"])
        if p["district"]:
            rel("CONSTITUENCY", "Place", p["district"], p["start"], p["end"])
    for r in f["relatives"]:
        rel("RELATIVE_OF", "Person", r["name"], kind=r["kind"])
    if f["birthplace"]:
        rel("BORN_IN", "Place", f["birthplace"], f["born"])
    for e in f["education"]:
        rel("EDUCATED_AT", "Organisation", e["name"], e["start"], e["end"])
    return list(ents.values()), rels


def wikidata_summary(entity_json):
    """A short readable English paragraph of the record, so the structured data also enters the text corpus. '' if nothing is known."""
    f = parse_wikidata(entity_json)
    if not f["name"]:
        return ""
    he, his = {MALE: ("He", "His"), FEMALE: ("She", "Her")}.get(f["gender"], (f["name"], f["name"] + "'s"))
    life = []
    if f["born"]:
        life.append("born " + _human(f["born"]) + (f" in {f['birthplace']}" if f["birthplace"] else ""))
    if f["died"]:
        life.append("died " + _human(f["died"]))
    head = f["name"] + (f" ({'; '.join(life)})" if life else "")
    out = [f"{head} is described in Wikidata as {f['description']}." if f["description"] else f"{head} is a person recorded in Wikidata."]
    for p in f["parties"]:
        out.append(f"{he} was a member of the party {p['name']}" + _span(p) + ".")
    for p in f["positions"]:
        out.append(f"{he} held the position of {p['name']}" + _span(p) + (f", representing {p['district']}" if p["district"] else "") + ".")
    for r in f["relatives"]:
        out.append(f"{his} {r['kind']} was {r['name']}." if r["kind"] else f"{he} is related to {r['name']}.")
    for e in f["education"]:
        out.append(f"{he} was educated at {e['name']}" + _span(e) + ".")
    return " ".join(out)


def _span(item):
    a, b = _human(item.get("start")), _human(item.get("end"))
    return f" from {a} to {b}" if a and b else (f" from {a}" if a else (f" until {b}" if b else ""))
