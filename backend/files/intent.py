"""Rule-based (no model) check: is this chat message a request to convert an attached file?

Result kinds:
  none      -> not about files; the message goes to the normal chat flow untouched
  convert   -> convert the latest attachment to `target`
  edit      -> asked for summarising / translating / rewriting: refused honestly (this tool only changes format)
  unknown   -> asked for a format we do not make
  no_file   -> looks like a conversion request but nothing is attached
"""
import re
from dataclasses import dataclass

# phrase -> target. Longer phrases first so "markdown" wins over "md".
TARGET_WORDS = [
    (r"(?:ms\s*|microsoft\s*)?word(?:\s+(?:document|doc|file|format))?|docx", "docx"),
    (r"pdf", "pdf"),
    (r"markdown|md", "md"),
    (r"web\s*page|html?", "html"),
    (r"csv|comma[- ]separated(?:\s+values)?", "csv"),
    (r"json", "json"),
    (r"plain\s*text|text\s+(?:file|format)|txt", "txt"),
]
# file formats people ask for that we do not make
UNKNOWN_FORMATS = ("xlsx", "xls", "pptx", "ppt", "odt", "ods", "odp", "epub", "rtf", "png", "jpg", "jpeg", "gif", "svg", "mp3",
                   "mp4", "wav", "zip", "rar", "exe", "xml", "yaml", "yml", "tex", "latex", "doc", "excel", "powerpoint",
                   "mobi", "bmp", "webp", "tsv", "sql", "py", "js")

CONVERT_VERBS = r"convert|change|turn|transform|switch|export|save|download|make|create|generate|give|get|produce|need|want|put|send|write|output|render|rewrite|re-?save"
STRONG_VERBS = r"convert|change|turn|transform|switch|export|save"
EDIT_WORDS = (r"summari[sz]e|summary|tl;?dr|translate|rewrite|re-?write|paraphrase|proofread|fix\s+(?:the\s+)?(?:grammar|spelling|typos?)|"
              r"correct\s+(?:the\s+)?(?:grammar|spelling)|shorten|make\s+(?:it|this|that)\s+(?:shorter|longer|simpler|formal|casual|better|clearer)|"
              r"expand|improve|edit|polish|simplif(?:y|ied)|extract|analy[sz]e|explain|review")
REFERENCE = r"\b(?:it|this|that|these|the\s+file|my\s+file|file|document|doc|attachment|attached|upload(?:ed)?|the\s+text|my\s+text)\b"

_target_alt = "|".join(f"(?:{p})" for p, _ in TARGET_WORDS)
_unknown_alt = "|".join(UNKNOWN_FORMATS)


@dataclass
class Intent:
    kind: str
    target: str = ""
    word: str = ""


def _target_in(text):
    """The target format named after to/into/as/in (or 'a pdf', 'pdf version'). Last mention wins."""
    best = None
    for pattern, name in TARGET_WORDS:
        for m in re.finditer(rf"\b(?:{pattern})\b", text):
            if best is None or m.start() >= best[0]:
                best = (m.start(), name)
    return best[1] if best else ""


def _unknown_in(text):
    m = re.search(rf"\b(?:to|into|as|in)\s+(?:an?\s+|the\s+)?\.?({_unknown_alt})\b(?:\s+(?:file|format|document|sheet|spreadsheet))?", text)
    return m.group(1) if m else ""


def parse(message, has_attachment):
    text = " ".join(str(message or "").lower().split())
    if not text:
        return Intent("none")
    target = _target_in(text)
    unknown = "" if target else _unknown_in(text)
    has_verb = bool(re.search(rf"\b(?:{CONVERT_VERBS})\b", text))
    strong = bool(re.search(rf"\b(?:{STRONG_VERBS})\b", text))
    refers = bool(re.search(REFERENCE, text))
    short_format_only = bool(target) and len(text.split()) <= 4 and bool(re.fullmatch(
        rf"(?:(?:please|pls|to|as|in|into)\s+)*(?:an?\s+)?(?:{_target_alt})(?:\s+(?:format|version|file|please|pls|thanks))*[.!]?", text))
    format_words = bool(re.search(rf"\b(?:to|into|as|in)\s+(?:an?\s+|the\s+)?(?:{_target_alt}|{_unknown_alt})\b", text)) \
        or bool(re.search(rf"\b(?:an?\s+)?(?:{_target_alt})\s+(?:version|copy|file|format|document)\b", text)) \
        or bool(re.search(rf"\b(?:make|create|generate|give|get|produce|need|want)\s+(?:me\s+)?(?:an?\s+)?(?:{_target_alt})\b(?!\s+(?:of|about|on|for)\b)", text))

    asks_edit = bool(re.search(rf"\b(?:{EDIT_WORDS})\b", text))
    wants_convert = (target or unknown) and ((has_verb and format_words) or (strong and (target or unknown)) or short_format_only)
    # "make a pdf of the history of Rome" without any file is a normal chat question, not a file request
    if wants_convert and not has_attachment and not (refers or strong):
        wants_convert = False
    if short_format_only and not has_attachment:
        wants_convert = False

    if asks_edit and (has_attachment and (refers or wants_convert) or wants_convert and refers):
        return Intent("edit")
    if not wants_convert:
        return Intent("none")
    if not has_attachment:
        return Intent("no_file", target)
    if unknown:
        return Intent("unknown", word=unknown)
    return Intent("convert", target)
