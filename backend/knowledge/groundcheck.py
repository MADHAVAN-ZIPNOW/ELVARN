"""Check that an answer written from notes really comes from those notes; if not, show the notes themselves.

The small model copies and blends notes unreliably: it can mix the date of one press release with the text of another, drop digits from an
amount ("Rs.3,50,800" -> "Rs.3,50/-"), name people that are not in the notes, or loop ("CBI, CBI, CBI"). These are the checks, all simple
and deterministic (no model):
  * every number in the answer (amount, date, year, section) appears in the notes;
  * every name (capitalised word / abbreviation, except the first word of a sentence) appears in the notes or in the question;
  * no word or three-word phrase is repeated in a loop;
  * a citation [n] points at a note that exists.
An honest refusal ("The notes do not say.") always passes. When an answer fails, `extractive_answer` returns the best notes word for
word, each with its [n]: slower to read than a fluent sentence, but never wrong.

    guard(answer, model_text, question) -> (text_to_show, "model" | "extractive", reasons)
"""
import re

NOTE_LINE = re.compile(r"^\[(\d+)\]\s+(.*)$", re.M)
CITATION = re.compile(r"\[\d+\]")
NUMBER = re.compile(r"\d[\d,./:-]*\d|\d")
NAME = re.compile(r"\b[A-Z][A-Za-z&@.]{2,}\b")
REFUSAL = re.compile(r"^\s*(the notes (do not|don't) (say|contain|answer)|i cannot find that in the notes|not mentioned in the notes)", re.I)
ALLOWED_NAMES = {"according", "from", "the", "this", "that", "these", "note", "notes", "rs", "also", "however", "there", "here", "see"}


def parse_notes(model_text):
    """{n: note text} from the 'Web notes' block of the prompt."""
    return {int(n): t.strip() for n, t in NOTE_LINE.findall(model_text or "")}


def _norm_number(s):
    return s.strip(".,;:-/")


def check_answer(answer, notes, question=""):
    """(ok, reasons). `notes` is a dict {n: text}."""
    text = (answer or "").strip()
    if not text or REFUSAL.match(text):
        return True, []
    reasons = []
    note_blob = " ".join(notes.values())
    note_low = note_blob.lower()
    bare = CITATION.sub(" ", text)

    for n in {int(x[1:-1]) for x in CITATION.findall(text)}:
        if n not in notes:
            reasons.append(f"cites [{n}] but there is no such note")

    for num in NUMBER.findall(bare):
        num = _norm_number(num)
        if num and num not in note_blob and num not in question:
            reasons.append(f"number {num} is not in the notes")
            break

    known = note_low + " " + question.lower()
    unknown = []
    for sentence in re.split(r"(?<=[.!?])\s+", bare):
        words = NAME.findall(sentence)
        first = re.match(r"\s*([A-Za-z&@.]+)", sentence)
        for w in words:
            if first and w == first.group(1):
                continue                               # the first word of a sentence is capitalised anyway
            if w.lower().strip(".") in ALLOWED_NAMES:
                continue
            if w.lower().strip(".") not in known:
                unknown.append(w)
    if unknown:
        reasons.append("names not in the notes: " + ", ".join(sorted(set(unknown))[:4]))

    words = re.findall(r"[A-Za-z0-9']+", bare.lower())
    if any(words[i] == words[i + 1] and len(words[i]) > 3 for i in range(len(words) - 1)):
        reasons.append("a word is doubled")
    pairs = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
    if any(pairs.count(g) >= 2 and len(g[0]) > 2 and len(g[1]) > 2 for g in set(pairs)):
        reasons.append("a word pair repeats")
    if any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2)):
        reasons.append("a word repeats three times in a row")
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if any(grams.count(g) >= 3 for g in set(grams)):
        reasons.append("a phrase repeats in a loop")
    return not reasons, reasons


STOP = {"what", "who", "whom", "when", "where", "which", "how", "why", "the", "and", "for", "are", "was", "were", "did", "does", "tell",
        "about", "with", "from", "that", "this", "have", "has", "had", "list", "give", "show"}


def _terms(text):
    return {w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower()) if w not in STOP}


def usable_note(text):
    """Not a bare link, not a one-line heading / question. A wiki heading such as '=== Chief Minister of Tamil Nadu (2026-present) ===' has six words
    but says nothing: the model then answers 'The notes do not say.' although the web could have answered."""
    t = (text or "").strip()
    if re.match(r"^=+.*=+$", t) or re.match(r"^#{1,6}\s", t):
        return False
    return len(t.split()) >= 6 and not re.match(r"^(https?://|www\.)\S+$", t) and not t.endswith("?")


def pick_notes(notes, question, max_notes=2):
    """Note numbers worth showing: usable ones, best match with the question first; a second note only if it matches nearly as well."""
    q = _terms(question)
    scored = []
    for n in sorted(notes):
        if usable_note(notes[n]):
            scored.append((len(q & _terms(notes[n])), -n, n))
    if not scored:
        scored = [(0, -n, n) for n in sorted(notes)]
    scored.sort(reverse=True)
    best = scored[0][0]
    keep = [n for s_, _, n in scored if s_ >= max(1, best * 0.7)][:max_notes] if best else [scored[0][2]]
    return sorted(keep) or [scored[0][2]]


def extractive_answer(notes, max_notes=2, max_chars=320, label="the sources", question=""):
    """The notes that match the question, word for word, each with its number: 'From the sources: [1] ... [2] ...'."""
    parts = []
    for n in (pick_notes(notes, question, max_notes) if question else sorted(notes)[:max_notes]):
        t = notes[n]
        if len(t) > max_chars:
            t = t[:max_chars].rsplit(" ", 1)[0] + " ..."
        parts.append(f"[{n}] {t}")
    return f"From {label}: " + " ".join(parts) if parts else ""


def guard(answer, model_text, question="", label="the sources"):
    """What to show: the model's answer when it passes the checks, else the notes themselves."""
    notes = parse_notes(model_text)
    if not notes:
        return answer, "model", []
    ok, reasons = check_answer(answer, notes, question)
    if ok:
        return answer, "model", []
    return extractive_answer(notes, label=label, question=question), "extractive", reasons
