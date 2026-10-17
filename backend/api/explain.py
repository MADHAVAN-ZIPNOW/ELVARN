"""The grounded "model explanation" layer for stored-record answers.

The retrieved records stay the primary, always-visible evidence (printed by code, never changed). This module adds an OPTIONAL short explanation:

  1. the quotes come from the retrieval context (backend/knowledge/quotes.py: sentences the cross-encoder scored as relevant, copied exactly);
  2. at most MAX_PROMPT_QUOTES of them (the short, high-scoring ones) are given to the small model, which writes one cited claim per line;
  3. Python parses the lines and validates them (every claim cited, every citation a quote the model was given, every claim's words found in what it
     cites, no refusal, not empty, not too long). Anything that fails is dropped and the caller shows the records only; the reason is logged.

The model's own statement that it is grounded is never trusted. Result of explain(): {"status", "explanation", "reason"}. status is one of
ANSWERED, PARTIAL, NOT_ENOUGH_EVIDENCE, DISABLED, INVALID; `explanation` is the dict below (or None); `reason` says why an explanation was dropped.
"""
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from api.explain_prompts import NOT_ENOUGH_EVIDENCE, build_explanation_prompt
from knowledge.groundcheck import REFUSAL, check_answer

log = logging.getLogger("elvarn.explain")

ANSWERED, PARTIAL, NOT_ENOUGH, DISABLED, INVALID = "ANSWERED", "PARTIAL", "NOT_ENOUGH_EVIDENCE", "DISABLED", "INVALID"
MAX_PROMPT_QUOTES = 3            # how many quotes the model sees
SCORE_BAND = 3.0                 # "high-scoring": within this many logits of the best quote
MAX_NEW_TOKENS = 120             # the explanation is short; the window is 512 tokens in all
MAX_WORDS = 180
TIMEOUT_SECONDS = float(os.environ.get("ELVARN_EXPLAIN_TIMEOUT", "8"))
SUPPORT_SHARE = 0.6              # share of a claim's content words that must be found in the quotes it cites
REPAIR_SHARE = 0.8                # a claim without a citation gets one from the code when this share of its words is in one given quote
PARTIAL_COVERAGE = 0.5           # below this share of the question's words found in the cited quotes: PARTIAL

_CITATION = re.compile(r"\[(\d+)\]")
_LEAD = re.compile(r"^(?:according to\s*(?:\[\d+\]\s*(?:and\s*)?)+:?|from the (?:notes|records|quotes):?|[-*•]\s*|\d+[.)]\s+)\s*", re.I)
_STOP = {"the", "and", "for", "that", "with", "this", "from", "have", "has", "was", "were", "which", "shall", "their", "there", "not", "are", "what",
         "does", "did", "about", "under", "into", "its", "can", "may", "any", "all", "but", "also", "such", "than", "then", "when", "where", "who"}
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="explain")        # one explanation at a time (CPU, small memory)


def enabled():
    """Off unless ELVARN_EXPLAIN=1: measured on the benchmark, only about one explanation in ten contains the right fact (the records alone: four in ten)."""
    return os.environ.get("ELVARN_EXPLAIN", "0") == "1"


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if (len(w) > 3 or w.isdigit()) and w not in _STOP}


def choose_quotes(quotes, limit=MAX_PROMPT_QUOTES, band=SCORE_BAND):
    """The quotes to show the model: of the high-scoring ones (within `band` of the best) the shortest `limit`, in record order.
    `quotes` is [{"record": n, "quote": text, "score": logit}]."""
    if not quotes:
        return []
    best = max(q["score"] for q in quotes)
    high = [q for q in quotes if q["score"] >= best - band]
    shortest = sorted(high, key=lambda q: (len(q["quote"].split()), -q["score"]))[:limit]
    return sorted(shortest, key=lambda q: (q["record"], -q["score"]))


def parse_lines(text):
    """(claims, missing, refused) from the model's line format. claims = [{"claim": text without markers, "citations": [n, ...]}].
    A line with several cited sentences is split after each citation group; "MISSING: ..." gives the missing information."""
    claims, missing = [], ""
    refused = bool(NOT_ENOUGH_EVIDENCE.lower() in (text or "").lower() or REFUSAL.match((text or "").strip()))
    for raw in (text or "").splitlines():
        line = _LEAD.sub("", raw.strip())
        if not line:
            continue
        if re.match(r"^missing\b", line, re.I):
            missing = line.split(":", 1)[1].strip() if ":" in line else line[7:].strip()
            continue
        if line.upper().startswith(NOT_ENOUGH_EVIDENCE):
            continue
        for piece in re.split(r"(?<=\])\s+(?=[A-Z0-9\"“(])", line):                  # one claim per cited sentence
            cites = [int(n) for n in _CITATION.findall(piece)]
            claim = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s+", " ", _CITATION.sub("", piece))).strip(" -:•*.;")
            if claim or cites:
                claims.append({"claim": claim, "citations": cites})
    return claims, missing, refused


def validate(text, given, question="", all_records=None):
    """Check the model's text. given = {record number: quote text} (what the model was shown). Returns (status, explanation dict or None, reason)."""
    claims, missing, refused = parse_lines(text)
    if refused and given:
        return INVALID, None, "refusal although relevant quotes exist"
    if not claims or not any(c["claim"] for c in claims):
        return INVALID, None, "empty answer"
    valid_ids = set(given)
    known = set(all_records or given)
    repaired = 0
    for c in claims:                                                    # a claim the model forgot to cite, whose words are clearly one quote's: the CODE attaches that record
        need = words(c["claim"])
        if c["claim"] and not c["citations"] and need:
            best_n, best = max(((n, len(need & words(q)) / len(need)) for n, q in given.items()), key=lambda pair: pair[1])
            if best >= REPAIR_SHARE:
                c["citations"], repaired = [best_n], repaired + 1
    for c in claims:
        if not c["claim"]:
            return INVALID, None, "citation without a claim"
        if not c["citations"]:
            return INVALID, None, "claim without a citation"
        for n in c["citations"]:
            if n not in known:
                return INVALID, None, f"citation [{n}] is not a retrieved record"
            if n not in valid_ids:
                return INVALID, None, f"citation [{n}] is not one of the quotes given to the model"
    kept, total = [], 0
    for c in claims:                                                    # at most MAX_WORDS words: keep whole claims from the top
        total += len(c["claim"].split())
        if total > MAX_WORDS and kept:
            break
        kept.append(c)
    if total > MAX_WORDS and len(kept) == 1 and len(kept[0]["claim"].split()) > MAX_WORDS:
        return INVALID, None, "explanation too long"
    for c in kept:
        cited = " ".join(given[n] for n in c["citations"])
        need = words(c["claim"])
        if need and len(need & words(cited)) / len(need) < SUPPORT_SHARE:
            return INVALID, None, "unsupported claim: " + c["claim"][:60]
    answer = " ".join(c["claim"] + " " + "".join(f"[{n}]" for n in c["citations"]) + "." for c in kept)
    notes = {n: given[n] for n in given}
    ok, reasons = check_answer(answer, notes, question)
    if not ok:
        return INVALID, None, "failed the grounding check: " + "; ".join(reasons)[:80]
    used = sorted({n for c in kept for n in c["citations"]})
    q_words = words(question)
    coverage = len(q_words & words(" ".join(given[n] for n in used))) / len(q_words) if q_words else 1.0
    status = PARTIAL if (missing or coverage < PARTIAL_COVERAGE) else ANSWERED
    explanation = {"status": status, "answer": answer, "key_points": [{"claim": c["claim"], "citations": c["citations"]} for c in kept],
                   "used_records": used, "missing_information": missing, "citations_added_by_code": repaired}
    return status, explanation, ""


def explain(service, question, records, quotes, style="native", timeout=TIMEOUT_SECONDS):
    """Run the explanation layer. records = [{"n", "text"}] (the final records, numbered as shown); quotes = [{"record", "quote", "score"}] or None
    (None: the retrieval did not compute quotes, so there is nothing to explain from). Never raises: any failure is DISABLED or INVALID."""
    if not enabled():
        return {"status": DISABLED, "explanation": None, "reason": "switched off"}
    if quotes is None or service is None or not getattr(service, "ready", False):
        return {"status": DISABLED, "explanation": None, "reason": "no quotes or no model"}
    chosen = choose_quotes(quotes)
    if not chosen:
        return {"status": NOT_ENOUGH, "explanation": None, "reason": "no relevant quote"}      # no model call: nothing in the records answers it
    given = {}
    for q in chosen:
        given.setdefault(q["record"], q["quote"])
    prompt = build_explanation_prompt(question, [(q["record"], q["quote"]) for q in chosen], style)
    try:
        out = _pool.submit(service.generate, prompt, mode="chat", max_new_tokens=MAX_NEW_TOKENS, greedy=True, repetition_penalty=1.15,
                           no_repeat_ngram_size=4).result(timeout=timeout)
        text = (out.get("text") or "") if isinstance(out, dict) else ""
    except FutureTimeout:
        log.warning("explanation dropped: model call timed out after %.0f s", timeout)
        return {"status": INVALID, "explanation": None, "reason": "timeout"}
    except Exception as exc:  # noqa: BLE001 - the records are shown anyway
        log.warning("explanation dropped: model call failed: %s", exc.__class__.__name__)
        return {"status": INVALID, "explanation": None, "reason": "model error"}
    status, explanation, reason = validate(text, given, question, {r["n"] for r in records})
    if explanation is None:
        log.info("explanation rejected: %s", reason)
    return {"status": status, "explanation": explanation, "reason": reason}
