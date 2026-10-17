"""Decide whether a question needs a web search, and build the search query.

ELVARN is a tiny from-scratch model with no tool-calling, so it cannot decide
for itself. This is a BEST-EFFORT rule/keyword check on the question text and the
conversation. It cannot know whether a fact exists anywhere on the web, and it
will sometimes search when it did not need to, or miss a question that needed it.

It is deliberately one function (`decide`) behind a tiny interface, so it can be
replaced later (for example by a model that supports tool calling, using
WEB_SEARCH_TOOL below) without touching the chat code.
"""
import re
from dataclasses import dataclass

# Description for a FUTURE tool-calling model. Not used by ELVARN today.
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web and return a few short results (title, link, snippet). Use it for current, "
        "time-sensitive or externally verifiable facts: news, prices, scores, weather, recent events, "
        "facts about named people, companies, places or products. Do NOT use it for creative writing, "
        "greetings, casual chat, or questions that can be answered from the conversation alone. "
        "Retrieval only: it cannot buy, book, send or change anything."),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "maxLength": 200,
                                 "description": "A short, self-contained search query."}},
        "required": ["query"],
    },
}

MAX_QUERY_CHARS = 200

_EXPLICIT = re.compile(r"\b(search (the )?(web|internet|online)|web search|search online|google (it|this|that|for)|"
                       r"look (it |this |that )?up (online|on the web)|search for|find online)\b", re.I)
_TIME = re.compile(r"\b(latest|today|tonight|yesterday|currently|current|right now|news|breaking|recent(ly)?|"
                   r"price of|prices?|stock|share price|score|scores|weather|forecast|who won|results? of|"
                   r"this (week|month|year)|release date|exchange rate|trending|live|upcoming|"
                   r"20(2[4-9]|3\d))\b", re.I)
_GREETING = re.compile(r"^\s*(hi+|hello+|hey+|hii+|yo|sup|good (morning|afternoon|evening|night)|thanks?( you)?|"
                       r"thank you|bye|ok(ay)?|cool|nice|lol|how are you\??|what'?s up\??)\W*$", re.I)
_CREATIVE = re.compile(r"\b(write|compose|draft|poem|story|joke|riddle|song|lyrics|essay|rhyme|haiku|imagine|"
                       r"pretend|roleplay|role-play|make up|tell me a (joke|story)|translate)\b", re.I)
_ABOUT_CONVERSATION = re.compile(r"\b(what did i (say|ask)|our (chat|conversation)|summari[sz]e (this|our|the) |"
                                 r"repeat (that|it)|say that again|you just (said|told)|earlier)\b", re.I)
_DOMAIN_QUESTION = re.compile(r"^\s*(what|who|whom|when|where|which|why|how|define|explain|describe|tell me about|meaning of|is|are|can|does|do)\b", re.I)
# Informational questions ("what is photosynthesis?", "tell me about the Kaveri river", "capital of Brazil"): the 17M model cannot recall facts
# reliably (it copies key facts from notes only about 22% of the time and loops or invents without them), so these are looked up on the web.
_INFORMATIONAL = re.compile(r"^\s*((what|who|whom|whose|when|where|which|why|how)\b|define\b|explain\b|describe\b|tell me about\b|"
                            r"meaning of\b|(the )?(capital|population|area|length|height|history|origin|definition|meaning|founder|president|"
                            r"prime minister|chief minister) of\b)", re.I)
_ABOUT_ASSISTANT = re.compile(r"\b(you|your|yours|yourself|my|mine|myself)\b", re.I)               # "what is your name?", "what is my name?"
_CHAT_ABBREVIATION = re.compile(r"\bwhat (does|do) \w{2,4} (mean|stand for)\b", re.I)            # "what does btw mean?" is casual chat, not a lookup
_QUESTION_START =re.compile(r"^\s*(who|what|when|where|which|how (many|much|old|tall|far|long))\b", re.I)
_REFERENCE = re.compile(r"\b(it|its|that|this|they|them|their|he|him|his|she|her|there|those|these|the (company|"
                        r"person|team|country|city|product|game|movie|film))\b", re.I)
_STOP = {"the", "and", "for", "are", "was", "who", "what", "when", "where", "which", "how", "does", "did",
         "tell", "about", "please", "can", "you", "its", "with", "that", "this", "from", "have", "has",
         # question filler: "what is mean by arxiv" must not match a page about the mean, median and mode
         "mean", "means", "meant", "meaning", "define", "definition", "explain", "describe", "difference", "between"}


@dataclass
class RouteDecision:
    search: bool
    reason: str
    query: str = ""
    core: str = ""        # the user's own words (without the added legal context): used to judge whether a result is relevant


def _has_named_entity(text):
    """A capitalised word (not the first word) or an ALL-CAPS acronym: a rough 'named thing' signal."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9'&.-]*", text)
    for i, w in enumerate(words):
        if len(w) >= 2 and ((i > 0 and w[0].isupper() and w.lower() not in ("i", "i'm", "i'd")) or (w.isupper() and len(w) >= 3)):
            return True
    return False


def _content_words(text):
    """Words of 3+ letters that are not question filler: a question needs at least one to be searchable."""
    return [w for w in re.findall(r"[A-Za-z0-9]{3,}", text.lower()) if w not in _STOP and w not in ("tell", "capital", "history", "origin")]


def _worth_searching(text):
    """(bool, reason) for one message on its own."""
    if _EXPLICIT.search(text):
        return True, "explicit request"
    if _GREETING.match(text) or _CREATIVE.search(text) or _ABOUT_CONVERSATION.search(text):
        return False, "chat or creative or about the conversation"
    if _TIME.search(text):
        return True, "time-sensitive words"
    if _QUESTION_START.match(text) and _has_named_entity(text):
        return True, "fact question about a named thing"
    if (_INFORMATIONAL.match(text) and not _ABOUT_ASSISTANT.search(text) and not _CHAT_ABBREVIATION.search(text)
            and not _REFERENCE.search(text) and _content_words(text)):          # 'when was that company founded?' needs history to mean anything
        return True, "informational question (the small model cannot recall facts reliably)"
    return False, "answerable without the web"


def build_query(text, history=None, max_chars=MAX_QUERY_CHARS):
    """Search query from the question; a follow-up like 'when was that founded?' borrows words from the previous user turn."""
    q = re.sub(r"\s+", " ", re.sub(r"^\s*(please\s+)?(search (the )?(web|internet|online)( for)?|google)\s*", "", text, flags=re.I)).strip(" ?")
    if history and _REFERENCE.search(q):
        prev = history[-1][0] if history[-1] else ""
        prev = re.sub(r"\s+", " ", re.sub(r"^\s*(please\s+)?search (the )?(web|internet|online)( for)?\s*", "", prev, flags=re.I)).strip(" ?")
        if prev:
            q = f"{prev} {q}"
    return q[:max_chars].strip()


def decide(text, history=None, mode="chat"):
    """The routing step. history: list of (user, assistant) pairs, oldest first."""
    text = (text or "").strip()
    if mode == "raw" or len(text) < 4:
        return RouteDecision(False, "not a question")
    ok, reason = _worth_searching(text)
    if not ok and mode == "domain" and _DOMAIN_QUESTION.match(text) and not _GREETING.match(text) \
            and not _CREATIVE.search(text) and not _ABOUT_CONVERSATION.search(text):
        ok, reason = True, "India Law mode: a legal or government question"      # concept questions ("What is a writ petition?") are looked up too
    if not ok and history and _REFERENCE.search(text) and len(text.split()) <= 12 and not _GREETING.match(text) \
            and not _CREATIVE.search(text) and not _ABOUT_CONVERSATION.search(text):
        prev_ok, _ = _worth_searching(history[-1][0] or "")
        if prev_ok:
            ok, reason = True, "follow-up to a question that needed the web"
    if not ok:
        return RouteDecision(False, reason)
    query = build_query(text, history)
    if len(re.findall(r"[A-Za-z0-9]{3,}", query)) == 0:
        return RouteDecision(False, "no searchable words")
    core = query
    if mode == "domain" and not re.search(r"\bindia", query, re.I):
        # The India Law mode: "What is Article 21?" must not find Article 21 of other countries or random pages.
        # Search engines answer natural phrases much better than bare keywords (tested: 9 of 10 good results with these endings).
        if re.search(r"\barticles?\s+\d+", query, re.I) and not re.search(r"constitution", query, re.I):
            query = f"{query} of the Constitution of India"
        else:
            query = f"{query} under Indian law"
        query = query[:MAX_QUERY_CHARS]
    return RouteDecision(True, reason, query, core)


def keywords(text):
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in _STOP}


def is_small_talk(text):
    """True for a greeting, thanks, a creative request ("tell me a joke"), a question about the conversation or about the assistant itself
    ("who are you?"). None of these is a question for the stored records: the chat model answers them, and nothing is searched."""
    t = (text or "").strip()
    return bool(_GREETING.match(t) or _CREATIVE.search(t) or _ABOUT_CONVERSATION.search(t) or _CHAT_ABBREVIATION.search(t)
                or re.match(r"^\s*(who|what) (are|is) (you|your)\b", t, re.I) or re.match(r"^\s*(what can you do|how are you)\b", t, re.I))
