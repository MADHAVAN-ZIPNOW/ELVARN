"""Glue between the chat endpoints and the search pieces (router -> service -> prompt).

Kept separate from server.py so the chat code only deals with one small result object.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from .prompt import build_evidence_text
from .router import decide
from .service import SKIPPED, UNAVAILABLE, MESSAGES, USED_NO_EVIDENCE_MESSAGE, SearchService

WEB_SEARCH_MODES = ("auto", "off")


@dataclass
class SearchPlan:
    """What to do before generating. `query` is set only when a search should run."""
    query: Optional[str] = None
    reason: str = ""


@dataclass
class SearchResult:
    status: str = SKIPPED
    searched: bool = False
    sources: List[dict] = field(default_factory=list)
    message: str = ""
    model_text: Optional[str] = None      # question + evidence for the model (None: use the plain question)
    evidence_in_prompt: int = 0

    def fields(self):
        """The machine-readable fields added to every chat response."""
        return {"search_status": self.status, "searched": self.searched, "sources": self.sources,
                "search_message": self.message, "evidence_in_prompt": self.evidence_in_prompt}


def plan_search(text, history, mode, web_search):
    """Step 1 (cheap, no network): should we search, and for what?"""
    if web_search != "auto":
        return SearchPlan()
    d = decide(text, history, mode)
    return SearchPlan(d.query, d.reason) if d.search else SearchPlan(None, d.reason)


def run_search(plan, text, search: SearchService, model_service, max_new_tokens, allowed=True):
    """Step 2: do the search (if planned) and prepare the model text. Never raises."""
    if not plan.query:
        return SearchResult()
    if search is None or not allowed:
        return SearchResult(status=UNAVAILABLE, message=MESSAGES[UNAVAILABLE])
    outcome = search.search(plan.query)
    res = SearchResult(status=outcome.status, searched=outcome.searched,
                       sources=[s.as_dict() for s in outcome.sources], message=outcome.message)
    if outcome.searched:
        try:
            count, limit = model_service.prompt_budget()
            model_text, used = build_evidence_text(text, outcome.sources, count, limit, max_new_tokens)
        except Exception:
            model_text, used = None, []
        if used:
            res.model_text, res.evidence_in_prompt = model_text, len(used)
        else:
            res.message = USED_NO_EVIDENCE_MESSAGE      # results exist, but none fitted into the prompt: do not claim the answer came from them
    return res
