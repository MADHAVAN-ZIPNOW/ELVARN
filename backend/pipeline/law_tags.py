"""Case names, subjects and statutes of court judgments. Standard library only: the Spark container (Python 3.8, no extra packages) imports this too.

    case_of_hc / case_of_sc   the case name, type and number from the dataset's title / party columns
    full_title                "Case name, No., Court, date": the title of the document in every store
    subjects_of / statutes_of what the text is about (election, MLA/MP, criminal ...) and which Acts it cites
"""
import re
import unicodedata

SUBJECT_NAMES = ("election", "constitutional", "criminal", "corruption", "tax", "service", "property", "family", "company", "labour", "motor", "environment")


_HC_TITLE = re.compile(r"^\s*(?P<type>[^/]{1,40}?)\s*/\s*(?P<num>\d{1,7})\s*/\s*(?P<year>\d{4})\s+of\s+(?P<parties>.+?)\s*$", re.I | re.S)
_VS = re.compile(r"\s+(?:vs?\.?|versus|v/s)\s+", re.I)
_ON_DATE = re.compile(r"\s+on\s+\d{1,2}\s+[A-Za-z]+,?\s+\d{4}\s*$")


def clean(text):
    return re.sub(r"\s+", " ", str(text or "").replace("\xa0", " ")).strip(" ,.-")


def case_of_hc(title):
    """'CR. MISC./5208/2019 of Amit Kumar Vs The State Of Bihar' -> name 'Amit Kumar vs The State Of Bihar', type 'CR. MISC.', number '5208/2019'."""
    t = clean(title)
    m = _HC_TITLE.match(t)
    if m:
        return {"case_name": _VS.sub(" vs ", clean(m.group("parties"))), "case_type": m.group("type").strip(), "case_number": f"{int(m.group('num'))}/{m.group('year')}"}
    return {"case_name": _VS.sub(" vs ", t), "case_type": None, "case_number": None}


def case_of_sc(row):
    title = _ON_DATE.sub("", clean(row.get("title")))
    name = title or " vs ".join(x for x in (clean(row.get("petitioner")), clean(row.get("respondent"))) if x)
    return {"case_name": _VS.sub(" vs ", name), "case_type": None, "case_number": clean(row.get("case_id")) or None}


def full_title(case, court, date):
    """The title every store shows: the case name first, then the number, the court and the decision date."""
    bits = [case["case_name"][:200] or "Judgment"]
    if case.get("case_number"):
        bits.append(f"{case['case_type']} {case['case_number']}" if case.get("case_type") else f"No. {case['case_number']}")
    bits.append(court)
    if date:
        bits.append(date)
    return ", ".join(bits)[:300]


# ---- subjects and statutes found in the first pages ---------------------------------------------------------------
SUBJECTS = {
    "election": r"Representation of (?:the )?People|election petition|returned candidate|Election Commission|Lok Sabha|Rajya Sabha|Vidhan Sabha|Legislative (?:Assembly|Council)|"
                r"Member of (?:the )?(?:Legislative Assembly|Parliament|Legislative Council)|(?-i:\bMLAs?\b|\bMPs?\b)|disqualification|nomination paper",
    "constitutional": r"Article\s+\d+[A-Z]?\s+of\s+the\s+Constitution|Constitution of India|fundamental right|writ petition|public interest litigation|(?-i:\bPIL\b)",
    "criminal": r"Indian Penal Code|(?-i:\bIPC\b)|Code of Criminal Procedure|Cr\.?\s?P\.?\s?C|Bharatiya Nyaya Sanhita|\bbail\b|anticipatory bail|(?-i:\bFIR\b)|NDPS|POCSO",
    "corruption": r"Prevention of Corruption Act|disproportionate assets|illegal gratification|\bbribe|Prevention of Money.?Laundering|(?-i:\bPMLA\b)|Lokayukta",
    "tax": r"Income.?Tax Act|Central Excise|Customs Act|(?-i:\bGST\b)|Goods and Services Tax|assessment year|Finance Act",
    "service": r"service matters?|Administrative Tribunal|pension|seniority|promotion|disciplinary (?:proceedings|inquiry)|Fundamental Rules",
    "property": r"Land Acquisition|Transfer of Property Act|Registration Act|specific performance|tenant|eviction|Rent Control|title suit",
    "family": r"Hindu Marriage Act|Hindu Succession Act|Domestic Violence|maintenance|custody|divorce|Guardians and Wards",
    "company": r"Companies Act|Arbitration and Conciliation|Insolvency and Bankruptcy|Negotiable Instruments|Section\s+138|SARFAESI|(?-i:\bNCLT\b)",
    "labour": r"Industrial Disputes Act|workmen|Employees.? Provident Fund|Employees.? State Insurance|Minimum Wages|Payment of Gratuity",
    "motor": r"Motor Vehicles Act|Motor Accident Claims|compensation.*accident",
    "environment": r"Environment \(Protection\) Act|National Green Tribunal|Forest (?:Conservation )?Act|pollution",
}
_SUBJECT_RX = {name: re.compile(rx, re.I) for name, rx in SUBJECTS.items()}
_WORD = r"(?:[A-Z][\w&()'\-]*|of|and|for|the)"
_ACT = re.compile(rf"\b({_WORD}(?:\s+{_WORD}){{0,8}}\s+Act)\s*,?\s*(?:of\s+)?((?:18|19|20)\d{{2}})")
_LEAD = {"interplay", "between", "in", "the", "under", "of", "by", "see", "and", "for", "to", "as", "per", "that", "this", "with", "section", "sections", "act"}


def subjects_of(text):
    return [name for name, rx in _SUBJECT_RX.items() if rx.search(text)]


def statutes_of(text, limit=12):
    """The Acts named in the text, as 'Representation of the People Act, 1951' (each once, in order of appearance)."""
    out = []
    for m in _ACT.finditer(re.sub(r"\s+", " ", re.sub(r"List of Acts", " ", text))):
        words = m.group(1).split()
        while words and words[0].lower() in _LEAD:
            words.pop(0)
        if len(words) < 2:
            continue
        name = f"{' '.join(words)}, {m.group(2)}"
        if name not in out:
            out.append(name)
        if len(out) >= limit:
            break
    return out


def slug(text, n=70):
    s = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n] or "case"


# ---- where a case is in its life, and what kind of petition it is ---------------------------------------------------------------------------
LIFECYCLE = ("FIR", "Investigation", "Chargesheet", "Trial court", "High Court", "Supreme Court", "Final judgment")
_PETITIONS = {
    "writ petition": r"writ petition|\bW\.?\s?P\.?\s?\(|\bW\.?\s?P\.?\s?No",
    "election petition": r"election petition",
    "bail petition": r"bail (?:petition|application)|anticipatory bail|Crl\.?\s?M\.?\s?P",
    "quashing petition": r"quash(?:ing)? (?:of )?(?:the )?(?:FIR|proceedings|charge)|Crl\.?\s?O\.?\s?P|Section 482",
    "special leave petition": r"special leave petition|(?-i:\bSLP\b)",
    "review petition": r"review petition",
    "revision petition": r"criminal revision|Crl\.?\s?R\.?\s?C",
    "appeal": r"\bappeal\b",
}
_PETITION_RX = {name: re.compile(rx, re.I) for name, rx in _PETITIONS.items()}
_FIR = re.compile(r"\bFIR\b|registered a case|case (?:was )?registered|booked", re.I)
_CHARGE = re.compile(r"charge ?sheet|filed (?:a )?charge", re.I)
_FINAL = re.compile(r"allowed|dismissed|disposed|convicted|acquitted|sentenced", re.I)


def petitions_of(text):
    return [name for name, rx in _PETITION_RX.items() if rx.search(text)]


def stage_of(kind, text="", outcome=None):
    """The step of the case lifecycle a document belongs to. kind: sc, hc (court records) or dvac, cbi, ed (agency press releases)."""
    if kind == "sc":
        return "Final judgment" if outcome and _FINAL.search(outcome) else "Supreme Court"
    if kind == "hc":
        return "High Court"
    if _CHARGE.search(text):
        return "Chargesheet"
    if _FIR.search(text):
        return "FIR"
    return "Investigation"


def stage_rank(stage):
    return LIFECYCLE.index(stage) if stage in LIFECYCLE else len(LIFECYCLE)
