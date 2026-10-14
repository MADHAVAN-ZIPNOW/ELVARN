"""Propose facts from stored passages and let a person approve them.

The extractor only suggests: it finds sentences about an FIR, arrest, chargesheet, charges, conviction, acquittal,
discharge, quashing or bail, with any date, FIR number and law sections in the same sentence. A reviewer then links a
sentence to a case (and person) with `approve`. Rejected and pending rows never reach answers.

    python -m backend.knowledge.review extract [--include-reference]
    python -m backend.knowledge.review list [--limit 20]
    python -m backend.knowledge.review show ID
    python -m backend.knowledge.review approve ID --case "Case name" [--person "Name"] [--role accused] [--stage S] [--level L]
    python -m backend.knowledge.review reject ID
"""
import argparse
import json
import re
from datetime import datetime

from backend.knowledge.kb import COURT_DOC_TYPES, KnowledgeBase

# (stage, suggested status, pattern)
RULES = [
    ("fir", "registered", r"\bFIR\b|first information report|\bcrime no|registered a case|case was registered"),
    ("arrest", "arrested", r"\barrest(ed)?\b"),
    ("chargesheet", "charged", r"charge[- ]?sheet|final report|filed a charge"),
    ("trial", "charged", r"charges?\s+(were\s+|was\s+)?framed|framed charges?"),
    ("investigation", "on_bail", r"\bbail\b[^.]{0,40}\bgranted|granted\s+(interim\s+|anticipatory\s+|regular\s+)?bail"),
    ("final_judgment", "convicted", r"\bconvicted\b|found guilty|\bsentenced\b"),
    ("final_judgment", "acquitted", r"\bacquitted\b|\bacquittal\b"),
    ("final_judgment", "discharged", r"\bdischarged\b|discharge application"),
    ("final_judgment", "quashed", r"\bquash(ed|ing)\b"),
]
COURT_STATUSES = {"convicted", "acquitted", "discharged", "quashed", "on_bail"}
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
DATE_RES = [
    (re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b"), lambda m: (m[3], m[2], m[1])),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:day of\s+)?(" + "|".join(MONTHS) + r"),?\s+(\d{4})\b", re.I),
     lambda m: (m[3], MONTHS[m[2].lower()], m[1])),
    (re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I),
     lambda m: (m[3], MONTHS[m[1].lower()], m[2])),
]
SECTIONS = re.compile(r"\bsections?\s+[\d()A-Za-z, /&-]+?\s+(?:of\s+(?:the\s+)?)?(?:Prevention of Corruption Act|PC Act|P\.?C\.? Act"
                      r"|IPC|Indian Penal Code|PMLA|Prevention of Money[- ]Laundering Act)", re.I)
FIR_NO = re.compile(r"\b(?:crime|cr\.?|fir|ecr|cc|c\.c\.)\s*(?:no\.?|number)\s*[:.]?\s*([\w/.-]*\d[\w/.-]*)", re.I)
SENTENCE = re.compile(r"(?<=[.;])\s+(?=[A-Z(])|\n+")


def iso_date(sentence):
    for rx, parts in DATE_RES:
        m = rx.search(sentence)
        if m:
            y, mo, d = parts(m)
            try:
                return datetime(int(y), int(mo), int(d)).date().isoformat()
            except ValueError:
                pass
    return None


# Operative wording of a decision (what the court itself orders/holds), as opposed to recitals and arguments.
OPERATIVE = re.compile(r"\b(?:is|are|stands?|hereby|accordingly)\b[^.]{0,40}\b(?:convicted|acquitted|discharged|quashed|set aside)\b"
                       r"|\b(?:appeals?|petitions?|revisions?|writ petitions?)\b[^.]{0,60}\b(?:is|are|stands?)\s+(?:hereby\s+)?(?:allowed|dismissed)"
                       r"|\bsentenced to\b", re.I)


def propose(sentence, doc_type, operative=True):
    """List of proposals (dicts) for one sentence. In a court document a 'court_finding' is only suggested when the
    sentence is operative (see OPERATIVE); other mentions are at most 'reported', and non-operative talk of quashing is dropped
    (judgments discuss quashing in arguments far more often than they order it). Pass operative=True to skip this check."""
    out = []
    for stage, status, rx in RULES:
        if not re.search(rx, sentence, re.I):
            continue
        if doc_type in COURT_DOC_TYPES and status in COURT_STATUSES:
            if operative:
                level = "court_finding"
            elif status == "quashed":
                continue
            else:
                level = "reported"
        elif doc_type == "news" and status in ("registered", "arrested", "charged"):
            level = "alleged"
        else:
            level = "reported"
        sec, fir = SECTIONS.search(sentence), FIR_NO.search(sentence)
        out.append({"stage": stage, "status": status, "level": level, "event_date": iso_date(sentence),
                    "sections": sec.group(0) if sec else None, "fir_no": fir.group(1) if fir else None})
    return out


def extract(kb, include_reference=False):
    """Scan stored passages and store proposals as pending candidates. Returns the number of NEW proposals."""
    added = 0
    for r in kb.iter_passages(exclude_doc_types=() if include_reference else ("reference",)):
        for sentence in SENTENCE.split(r["text"]):
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 600:
                continue
            for p in propose(sentence, r["doc_type"]):
                added += kb.add_candidate(r["id"], p["stage"], p["status"], p["event_date"], sentence, p["level"], p["sections"], p["fir_no"])
    return added


def approve(kb, uid, case, person=None, role="accused", stage=None, level=None):
    kb.approve_candidate(uid, case, person, role, stage, level)


def reject(kb, uid):
    kb.reject_candidate(uid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "list", "show", "approve", "reject"])
    ap.add_argument("id", nargs="?", help="candidate id (shown by `list`)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--case")
    ap.add_argument("--person")
    ap.add_argument("--role", default="accused")
    ap.add_argument("--stage")
    ap.add_argument("--level", choices=["alleged", "reported", "court_finding"])
    ap.add_argument("--include-reference", action="store_true")
    a = ap.parse_args()
    kb = KnowledgeBase()
    if a.cmd == "extract":
        print(f"{extract(kb, a.include_reference)} new candidates")
    elif a.cmd == "list":
        for r in kb.pending_candidates(a.limit):
            c = r["c"]
            print(f"#{c['uid']} [{c['stage']}/{c['status']}/{c['level']}] {c['event_date'] or '-'} | {r['title']}\n    {c['sentence'][:200]}")
    elif a.cmd == "show":
        r = kb.get_candidate(a.id)
        print(json.dumps(r, indent=2, ensure_ascii=False) if r else "no such candidate")
    elif a.cmd == "approve":
        if not a.case:
            ap.error("approve needs --case")
        approve(kb, a.id, a.case, a.person, a.role, a.stage, a.level)
        print("approved")
    else:
        reject(kb, a.id)
        print("rejected")


if __name__ == "__main__":
    main()
