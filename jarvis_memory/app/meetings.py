"""The Teams meeting notes the desktop app saves as memories titled "Meeting: <name>" (one memory per 10-minute part of a call), turned
into something the chat can answer from: which meetings there were, and what was said in them."""
from __future__ import annotations

import re
from typing import Any

HEAD_RE = re.compile(r"^Meeting: (?P<title>.*?) - (?P<day>\w{3} \d{1,2} \w{3} \d{4}), (?P<start>\d{2}:\d{2}) to (?P<end>\d{2}:\d{2})")
MAX_CHARS = 6000
MAX_MEETINGS = 12


def _is_meeting(m: dict[str, Any]) -> bool:
    return str(m.get("title") or "").startswith("Meeting:") or str(m.get("content") or "").startswith("Meeting:")


def _body(content: str) -> str:
    """The spoken lines of one saved part, without its two header lines."""
    return "\n".join(str(content).split("\n")[2:]).strip()


def digest(memories: list[dict[str, Any]], query: str = "") -> str:
    """Meetings (newest first) from the saved memories (newest first). With a ``query`` or when only one or two meetings match, what was said is included."""
    found = [m for m in memories if _is_meeting(m)]
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in {"meeting", "meetings", "the", "my", "of", "in", "a"}]
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for m in reversed(found):                       # oldest part first, so each meeting's lines stay in order
        content = str(m.get("content") or "")
        head = HEAD_RE.match(content)
        title = head["title"] if head else str(m.get("title") or "Meeting")[len("Meeting: "):]
        day = head["day"] if head else str(m.get("created_at") or "")[:10]
        g = groups.setdefault((title, day), {"title": title, "day": day, "start": head["start"] if head else "", "end": "", "parts": [], "text": ""})
        if head:
            g["end"] = head["end"]
        g["parts"].append(m)
        g["text"] += ("\n" if g["text"] else "") + _body(content)
    chosen = [g for g in groups.values() if all(w in f"{g['title']} {g['day']} {g['text']}".lower() for w in words)]
    chosen = list(reversed(chosen))[:MAX_MEETINGS]  # newest meeting first
    if not chosen:
        return "No meeting notes found" + (f" for '{query}'." if words else ". The desktop app saves them while a Teams call runs, if Settings -> Meeting notes is on.")
    lines = [f"Meetings with notes ({len(chosen)}):"]
    for i, g in enumerate(chosen, 1):
        span = f", {g['start']} to {g['end']}" if g["start"] else ""
        lines.append(f"{i}. {g['title']} - {g['day']}{span} ({len(g['text'].splitlines())} lines of notes)")
    if words or len(chosen) <= 2:
        budget = MAX_CHARS
        for g in chosen:
            if budget <= 0:
                break
            text = g["text"][:budget]
            lines += ["", f"What was said in {g['title']} ({g['day']}):", text]
            budget -= len(text)
    return "\n".join(lines)
