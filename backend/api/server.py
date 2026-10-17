"""ELVARN local API + web UI.

Run from the project root:
    python -m uvicorn backend.api.server:app --port 8000
then open http://127.0.0.1:8000
"""
import hmac
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from files.routes import install_file_routes  # noqa: E402
from knowledge.kb import CORRUPTION_WORDS, NOT_FOUND, UNAVAILABLE, KnowledgeBase  # noqa: E402
from knowledge import graphrag, groundcheck  # noqa: E402
from knowledge.vectorstore import RagStore  # noqa: E402
from knowledge.remote import KnowledgeUnavailable, RemoteKnowledge  # noqa: E402
from notify.mailer import MailError, Mailer, clean_address, one_line  # noqa: E402
from notify.templates import new_chat_email  # noqa: E402
from telephony.routes import install_call_routes  # noqa: E402
from usermemory.store import Memory, session_key  # noqa: E402
from files.service import FileService  # noqa: E402
from api.model_service import ModelNotReady, ModelService  # noqa: E402
from model.generation import MAX_NO_REPEAT_NGRAM, MAX_REPETITION_PENALTY, MODES  # noqa: E402
from search.orchestrate import WEB_SEARCH_MODES, SearchResult, plan_search, run_search  # noqa: E402
from search.prompt import KB_HEADER, build_evidence_text  # noqa: E402
from search.providers.searxng import SearxngProvider, is_loopback_host, wake_up  # noqa: E402
from search.service import USED, USED_NO_EVIDENCE_MESSAGE, SearchService, Source  # noqa: E402

# Questions about the present ("who is the current ...", "latest ...", a year from now on) cannot be answered from stored records that may be months old:
# when web search is on, the web is asked too and its answer is preferred (the stored records stay as the fallback).
FRESH = re.compile(r"\b(current(ly)?|now|today|latest|recent(ly)?|present(ly)?|newest|incumbent|this (year|month|week)|as of)\b|\b20(2[5-9]|3\d)\b", re.I)
KB_SLEEPING_NOTE = ("The stored records could not be reached just now (the service may be waking up, which takes up to a minute), "
                    "so this answer was not checked against them. Ask again in a moment.")

KB_RAG_MESSAGE = ("Answer written from verified knowledge-graph facts and stored documents (judgments, press releases, reference pages); "
                  "every note is cited and labelled alleged / reported / court finding. The small model can still be wrong, so check the sources.")

# The React app (frontend-react/, built into dist/) is served when it has been built; the earlier single-file page in frontend/ is the fallback.
REACT_DIST = BACKEND_DIR.parent / "frontend-react" / "dist"
FRONTEND_DIR = REACT_DIST if (REACT_DIST / "index.html").is_file() and os.environ.get("ELVARN_LEGACY_UI") != "1" else BACKEND_DIR.parent / "frontend"


class StartRequest(BaseModel):
    """Optional settings of the Start button: none = the politician-by-politician run; mode "judgments" = the all-India court judgments run."""
    mode: Optional[str] = Field(default=None, max_length=20)
    scope: Optional[str] = Field(default=None, max_length=10)
    max_gb: Optional[float] = Field(default=None, ge=1, le=500)
    from_year: Optional[int] = Field(default=None, ge=1947, le=2026)
    to_year: Optional[int] = Field(default=None, ge=1947, le=2026)
    topics: Optional[List[str]] = Field(default=None, max_length=12)
    then: Optional[str] = Field(default=None, max_length=10)


def start_args(body):
    if body is None:
        return None
    return body.model_dump() if hasattr(body, "model_dump") else body.dict()


class TargetedRequest(BaseModel):
    text: Optional[str] = Field(default=None, max_length=300)
    person: Optional[str] = Field(default=None, max_length=60)
    sources: Optional[List[str]] = Field(default=None, max_length=10)


class Turn(BaseModel):
    user: str = Field(max_length=1000)
    assistant: str = Field(max_length=2000)


class GenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    mode: str = "chat"
    max_new_tokens: int = Field(default=80, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=0, le=500)
    greedy: bool = False
    history: Optional[List[Turn]] = Field(default=None, max_length=6)
    # "off" (default, so old clients behave exactly as before) or "auto" (search the web when the question needs it)
    web_search: str = "off"
    # Repetition controls. None = "not set by the client": off normally, WEB_REPETITION defaults when web evidence is in the prompt.
    # ids returned by POST /api/files; lets the server answer "convert this to pdf" itself (the model is not used for that)
    file_ids: Optional[List[str]] = Field(default=None, max_length=5)
    # optional id that ties a user's turns together for the memory store (default: derived from the visitor)
    session_id: Optional[str] = Field(default=None, max_length=64)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=MAX_REPETITION_PENALTY)
    no_repeat_ngram_size: Optional[int] = Field(default=None, ge=0, le=MAX_NO_REPEAT_NGRAM)


# Used only when the answer is built from web notes (small models tend to loop there) and the client set nothing.
WEB_REPETITION = {"repetition_penalty": 1.15, "no_repeat_ngram_size": 4}


def repetition_settings(req, evidence_in_prompt):
    """(settings for generate, where they came from). Client values win; the web defaults apply only when evidence
    is really in the prompt; otherwise everything stays off (the old behaviour)."""
    web = evidence_in_prompt > 0
    base = WEB_REPETITION if web else {"repetition_penalty": 1.0, "no_repeat_ngram_size": 0}
    settings = {
        "repetition_penalty": base["repetition_penalty"] if req.repetition_penalty is None else req.repetition_penalty,
        "no_repeat_ngram_size": base["no_repeat_ngram_size"] if req.no_repeat_ngram_size is None else req.no_repeat_ngram_size,
    }
    client = req.repetition_penalty is not None or req.no_repeat_ngram_size is not None
    return settings, ("client" if client else "web_default" if web else "off")


# When the answer is written from web notes, random sampling (temperature 0.8, top-k 40-65) makes the small model drift away
# from the notes and swap in details from memory. In that case the sliders are capped (lower numbers are kept as they are).
WEB_SAMPLING_CAP = {"temperature": 0.4, "top_k": 20}


def sampling_settings(req, evidence_in_prompt):
    """(temperature, top_k, where they came from). Greedy decoding is never changed; without notes in the prompt nothing changes."""
    if evidence_in_prompt > 0 and not req.greedy:
        t = min(req.temperature, WEB_SAMPLING_CAP["temperature"])
        k = req.top_k if req.top_k == 0 else min(req.top_k, WEB_SAMPLING_CAP["top_k"])
        k = WEB_SAMPLING_CAP["top_k"] if req.top_k == 0 else k          # top_k 0 means "no limit": cap it too
        return t, k, ("web_cap" if (t, k) != (req.temperature, req.top_k) else "client")
    return req.temperature, req.top_k, "client"


def wake_remote(remote, min_gap=300):
    """The hosted service sleeps on a free plan: ping its public /healthz in the background, at most once per `min_gap` seconds."""
    return remote.wake(min_gap)


GENERATION_PATHS = ("/api/chat", "/api/chat/stream", "/api/generate")


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


WELCOME_TEXT = """Hi!

Thanks for trying ELVARN, a small AI model that answers questions about Tamil Nadu corruption-case records and always shows its sources.

{first}ELVARN is small and can be wrong. Allegations are not proof of guilt: please check the sources it shows.

You are getting this mail because you left your address on the site and agreed to it. Reply STOP and we will delete it.
"""

def create_app(checkpoint_dir=None, load_on_start=True, public=None, rate_limit_per_minute=None, max_concurrent=None, search_service=None, file_service=None, kb_service=None, rag_service=None, memory_service=None, remote_service=None, mail_service=None, call_config=None, owner_key=None, collect_service=None):
    """public=True (or ELVARN_PUBLIC=1) is for a website anyone can reach: it turns on a per-visitor rate limit,
    a cap on simultaneous generations, and disables the reload endpoint."""
    public = env_flag("ELVARN_PUBLIC") if public is None else public
    if rate_limit_per_minute is None:
        rate_limit_per_minute = int(os.environ.get("ELVARN_RATE_LIMIT", "20" if public else "0"))
    if max_concurrent is None:
        max_concurrent = int(os.environ.get("ELVARN_MAX_CONCURRENT", "2" if public else "0"))
    app = FastAPI(title="ELVARN", docs_url=None, redoc_url=None)
    # Lets the UI work when opened from another local server such as
    # VS Code Live Server (127.0.0.1:5500). Local origins only.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-File-Name"],
    )
    service = ModelService(checkpoint_dir)
    app.state.service = service
    app.state.public = public
    search = search_service if search_service is not None else SearchService.from_env()
    app.state.search = search
    # A remote search service on Render's free plan sleeps when idle: wake it in the background
    # (never blocks start-up, ignores errors). Not done for localhost, injected services (tests) or when switched off.
    provider = getattr(search, "provider", None)
    if (search_service is None and isinstance(provider, SearxngProvider) and not is_loopback_host(provider.base_url)
            and os.environ.get("ELVARN_SEARCH_WAKEUP", "1") != "0"):
        wake_up(provider.base_url)
    files = file_service if file_service is not None else FileService()
    kb, kb_expected = kb_service, kb_service is not None
    if kb is None and os.environ.get("NEO4J_URI"):
        kb_expected = True
        try:
            kb = KnowledgeBase.from_env()
        except Exception as exc:  # the chat still starts; corruption questions are refused until the graph is back
            print(f"WARNING: knowledge graph unavailable ({exc}); corruption questions will be refused", flush=True)
    app.state.kb = kb
    rag_store, rag_expected = rag_service, rag_service is not None
    if rag_store is None and (os.environ.get("CHROMA_HOST") or os.environ.get("CHROMA_API_KEY")):
        rag_expected = True
        try:
            rag_store = RagStore.from_env()
        except Exception as exc:  # the graph branch still works without the vector branch
            print(f"WARNING: RAG vector store unavailable ({exc}); answers will use the knowledge graph only", flush=True)
    app.state.rag = rag_store
    memory = memory_service
    if memory is None and os.environ.get("REDIS_URL"):
        try:
            memory = Memory.from_env()
        except Exception as exc:  # no memory is fine: the chat just does not remember
            print(f"WARNING: memory store unavailable ({exc}); the chat will not remember earlier turns", flush=True)
    app.state.memory = memory
    # A hosted knowledge service (knowledge_service/) replaces local stores: the chat site only asks it for the context.
    remote, remote_expected = remote_service, remote_service is not None
    if remote is None and os.environ.get("KNOWLEDGE_URL"):
        remote_expected = True
        try:
            remote = RemoteKnowledge.from_env()
            wake_remote(remote)
        except Exception as exc:  # a wrong address or token: corruption questions are refused, the rest of the chat still works
            print(f"WARNING: knowledge service not usable ({exc}); corruption questions will be refused", flush=True)
    app.state.remote = remote
    app.state.files = files

    # ---- per-visitor rate limit (in memory; one server process) ----
    hits = defaultdict(deque)
    hits_lock = threading.Lock()

    def visitor(request: Request):
        forwarded = request.headers.get("x-forwarded-for", "")
        return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")

    def take_hit(who):
        """Record one use for this visitor. Returns None if allowed, else the seconds to wait."""
        if rate_limit_per_minute <= 0:
            return None
        now = time.monotonic()
        with hits_lock:
            q = hits[who]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= rate_limit_per_minute:
                return max(1, int(60 - (now - q[0])))
            q.append(now)
        return None

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        path = request.url.path
        limited = ((request.method == "POST" and (path in GENERATION_PATHS or path == "/api/files" or path.endswith("/convert")
                                                   and path.startswith("/api/files/")))
                   or (request.method == "GET" and path.startswith("/api/files/") and (path.endswith("/download") or path.endswith("/preview"))))
        if limited:
            wait = take_hit(visitor(request))
            if wait is not None:
                return JSONResponse({"detail": f"Too many requests. Please wait {wait} seconds and try again."},
                                    status_code=429, headers={"Retry-After": str(wait)})
        return await call_next(request)

    # ---- cap on simultaneous generations (the CPU is small) ----
    slots = threading.BoundedSemaphore(max_concurrent) if max_concurrent > 0 else None

    def take_slot():
        if slots is not None and not slots.acquire(blocking=False):
            raise HTTPException(429, "ELVARN is busy answering other visitors. Please try again in a moment.")

    def give_slot():
        if slots is not None:
            slots.release()

    if load_on_start:
        try:
            service.load()
        except ModelNotReady:
            pass  # reported by /api/health; UI shows the message

    def check_web_search(req):
        if req.web_search not in WEB_SEARCH_MODES:
            raise HTTPException(422, f"web_search must be one of {list(WEB_SEARCH_MODES)}")

    def search_step(plan, req, request):
        """Run the planned web search. A search counts against the visitor's rate limit like a chat request,
        and it never raises: if it cannot run, the chat simply continues without it."""
        allowed = True
        if plan.query and rate_limit_per_minute > 0:
            allowed = take_hit(visitor(request)) is None
        return run_search(plan, req.text, search, service, req.max_new_tokens, allowed)

    def model_input(req, found, history):
        """Text and history the model sees. With evidence, earlier turns are dropped: a small context window cannot hold both."""
        if found.model_text:
            return found.model_text, None
        return req.text, history

    def file_turn(req, request):
        """A conversion request for an attached file is answered by plain code with a fixed message (no model)."""
        if not req.file_ids:
            return None
        return files.handle_chat(req.text, req.file_ids, visitor(request))

    def kb_reply(req, text, found=False):
        """A fixed reply written by plain code (a refusal or an 'unavailable' notice); the model is not used."""
        return {"text": text, "raw_text": text, "mode": req.mode, "files": [], "file_action": None,
                "model_used": False, "generated_tokens": 0, "prompt_tokens": 0, "seconds": 0,
                "stop_reason": "knowledge_base", "kb_found": found, "kb_sources": []}

    def session_of(req, request):
        return session_key(req.session_id or visitor(request))

    def has_data():
        try:
            return bool(kb.populated()) if kb is not None else bool(rag_store is not None and rag_store.count() > 0)
        except Exception:
            return False

    def graph_step(req, request):
        """GraphRAG: Chroma vector search and Neo4j graph traversal run in parallel, then merge into one ranked, cited context.
        Returns (SearchResult | None, Context | None, early reply | None, note). An early reply is a refusal or an 'unavailable'
        notice for a corruption question: the model is never asked to guess those. `note` is a sentence to show when the stored records
        could not be reached for another kind of question (the chat then goes on without them, and says so)."""
        if req.file_ids or not (kb_expected or rag_expected or remote_expected):
            return None, None, None, None
        corruption = bool(CORRUPTION_WORDS.search(req.text))
        if remote_expected:                                   # hosted knowledge service: it does the retrieval, we only get the context
            if remote is None:
                return None, None, (kb_reply(req, UNAVAILABLE) if corruption else None), KB_SLEEPING_NOTE
            try:
                ctx, populated = remote.context(req.text, session_of(req, request))
            except Exception:
                return None, None, (kb_reply(req, UNAVAILABLE) if corruption else None), KB_SLEEPING_NOTE
        else:
            if kb is None and rag_store is None:
                return None, None, (kb_reply(req, UNAVAILABLE) if corruption else None), KB_SLEEPING_NOTE
            try:
                ctx = graphrag.build_context(kb, rag_store, req.text, memory, session_of(req, request))
            except Exception:
                return None, None, (kb_reply(req, UNAVAILABLE) if corruption else None), KB_SLEEPING_NOTE
            populated = None
        # headings, bare links and one-line fragments scraped from web pages are not answers: drop them before the model or the user sees them
        ctx.items = [i for i in ctx.items if i["kind"] == "fact" or groundcheck.usable_note(i.get("text", ""))]
        if not ctx.items:
            if corruption and (populated if populated is not None else has_data()):
                return None, None, kb_reply(req, NOT_FOUND), None
            return None, ctx, None, None
        sources = [Source(title=i["source"]["title"], url=i["source"]["url"], domain=i["source"]["label"], snippet=i["text"]) for i in ctx.items]
        res = SearchResult(status=USED, searched=True, sources=[s.as_dict() for s in sources], message=KB_RAG_MESSAGE)
        try:
            count, limit = service.prompt_budget()
            model_text, used = build_evidence_text(req.text, sources, count, limit, req.max_new_tokens, header=KB_HEADER)
        except Exception:
            model_text, used = None, []
        if used:
            res.model_text, res.evidence_in_prompt = model_text, len(used)
        else:
            res.message = USED_NO_EVIDENCE_MESSAGE
        return res, ctx, None, None

    def web_plan(req, history, rag):
        """The web search to run for this question, or None when the stored records are used as they are. Without stored evidence the usual
        plan applies; with it, a question about the present also goes to the web (when web search is on)."""
        if rag is not None and not (req.web_search == "auto" and FRESH.search(req.text)):
            return None
        return plan_search(req.text, history, req.mode, req.web_search)

    def settle(rag, found_web, note):
        """-> (evidence to answer from, whether it is the stored records). The web answer wins for a present-day question; the stored
        records stay when the web gave nothing. A sleeping knowledge service is said so when nothing else backs the answer."""
        if rag is not None and not (found_web is not None and found_web.evidence_in_prompt):
            return rag, True
        found = found_web if found_web is not None else SearchResult()
        if note and not found.sources:
            found.status, found.message = "unavailable", note
        return found, False

    def check_grounding(result, found, question, label="the web results"):
        """An answer written from notes must match the notes: otherwise the stored record text itself is shown (and the model's own
        wording stays in raw_text). Chat without notes is untouched."""
        if not (getattr(found, "model_text", None) and found.evidence_in_prompt):
            return
        shown, grounding, why = groundcheck.guard(result.get("text", ""), found.model_text, question, label=label)
        result["grounding"] = grounding
        if grounding == "extractive":
            result["model_text_replaced"] = result.get("text", "")
            result["text"] = shown
            result["grounding_reasons"] = why

    def model_summary(req, found, history):
        """The model's short reading of the retrieved notes, or None. It is only an addition: a refusal, an empty reply or a reply that fails
        the grounding check against the notes is dropped, and it can never hide or replace the retrieved records."""
        if not (found.model_text and found.evidence_in_prompt) or not service.ready:
            return None
        try:
            text, hist = model_input(req, found, history)
            rep, _ = repetition_settings(req, found.evidence_in_prompt)
            temp, topk, _ = sampling_settings(req, found.evidence_in_prompt)
            out = service.generate(text, mode=req.mode, max_new_tokens=req.max_new_tokens, temperature=temp, top_k=topk,
                                   greedy=req.greedy, history=hist, **rep)
        except Exception:  # noqa: BLE001 - the records are shown anyway
            return None
        answer = (out.get("text") or "").strip()
        if not answer or groundcheck.REFUSAL.match(answer):         # a refusal is never shown: the records above are the answer
            return None
        shown, grounding, _ = groundcheck.guard(answer, found.model_text, req.text, "the stored records")
        return answer if grounding == "model" else None

    def stored_answer(req, found, ctx, summary=None):
        """The reply for a question the stored records (Chroma + knowledge graph) returned something for. Layer 1, written by plain code:
        every retrieved item, word for word, with its number and source; the model is not asked whether the records answer the question
        (a 24M-parameter model judged that unreliably and refused answers that were in the notes). Layer 2, optional: the model's own short
        summary of those same notes, added below when it passes the grounding check and says something the records do not already say."""
        parts = []
        for n, s in enumerate(found.sources, 1):
            title = (s.get("title") or "").strip()
            parts.append(f"[{n}] {(s.get('snippet') or '').strip()}" + (chr(10) + f"    Source: {title}" if title else ""))
        text = "From the stored records:" + chr(10) * 2 + (chr(10) * 2).join(parts)
        flat = re.sub(r"\W+", " ", text).lower()
        if summary and re.sub(r"\W+", " ", summary).lower().strip() in flat:
            summary = None                                        # a copy of a record: it adds nothing
        if summary:
            text += chr(10) * 2 + "Summary (written by the model from the records above): " + summary
        result = {"text": text, "raw_text": text, "mode": req.mode, "files": [], "file_action": None, "model_used": bool(summary),
                  "generated_tokens": 0, "prompt_tokens": 0, "seconds": 0, "stop_reason": "stored_records", "grounding": "stored_records"}
        result.update(found.fields())
        result.update(kb_extra(ctx))
        return result

    def kb_extra(ctx):
        """Extra response fields: the exact graph facts used (with their level and citation), the entities found, any failed branch."""
        return {"kb_facts": [{"text": i["text"], "level": i["level"], "source": i["source"]} for i in ctx.facts()],
                "kb_entities": [e["name"] for e in ctx.understanding.entities], "kb_notes": ctx.notes,
                "kb_followup": ctx.understanding.followup}

    def remember(req, request, answer, ctx):
        """Keep the user's latest turn (and the people / cases it was about) in the memory store."""
        entities = ctx.understanding.entities if ctx is not None else []
        if remote is not None:
            remote.remember(session_of(req, request), req.text, answer or "", entities)
        elif memory is not None:
            memory.remember(session_of(req, request), req.text, answer or "", entities)

    def file_result(handled, req):
        return {"text": handled["text"], "raw_text": handled["text"], "mode": req.mode, "files": handled["files"],
                "file_action": handled["file_action"], "model_used": False, "generated_tokens": 0, "prompt_tokens": 0,
                "seconds": 0, "stop_reason": "file_action"}

    def run(req: GenerateRequest, request: Request):
        if req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")
        if not req.text.strip():
            raise HTTPException(422, "text must not be blank")
        handled = file_turn(req, request)
        if handled is not None:
            return file_result(handled, req)
        rag, ctx, early, note = graph_step(req, request)
        if early is not None:
            return early
        check_web_search(req)
        history = [(t.user, t.assistant) for t in req.history] if req.history else None
        take_slot()
        try:
            plan = web_plan(req, history, rag)
            found, from_kb = settle(rag, search_step(plan, req, request) if plan is not None else None, note)
            if from_kb and found.sources:
                result = stored_answer(req, found, ctx, model_summary(req, found, history))
                remember(req, request, result["text"], ctx)
                return result
            text, hist = model_input(req, found, history)
            rep, rep_source = repetition_settings(req, found.evidence_in_prompt)
            temp, topk, sampling_source = sampling_settings(req, found.evidence_in_prompt)
            result = service.generate(
                text, mode=req.mode, max_new_tokens=req.max_new_tokens,
                temperature=temp, top_k=topk,
                greedy=req.greedy, history=hist, **rep,
            )
            result.update(found.fields())
            if from_kb:
                result.update(kb_extra(ctx))
            check_grounding(result, found, req.text, "the stored records" if from_kb else "the web results")
            result["repetition_source"] = rep_source
            result["sampling_source"] = sampling_source
            remember(req, request, result.get("text", ""), ctx)
            return result
        except ModelNotReady as exc:
            raise HTTPException(503, str(exc))
        finally:
            give_slot()

    # the API docs page gets the ELVARN logo as its tab icon too
    @app.get("/api/docs", include_in_schema=False)
    def docs():
        return get_swagger_ui_html(openapi_url="/openapi.json", title="ELVARN API", swagger_favicon_url="/favicon.png")

    @app.get("/favicon.png", include_in_schema=False)
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        logo = FRONTEND_DIR / "logo.png"
        if not logo.is_file():
            raise HTTPException(404, "logo not found")
        return FileResponse(logo, media_type="image/png")

    @app.post("/api/chat/stream")
    def chat_stream(req: GenerateRequest, request: Request):
        """Newline-delimited JSON events: start, token..., done (or error)."""
        if req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")
        if not req.text.strip():
            raise HTTPException(422, "text must not be blank")
        handled = file_turn(req, request)
        if handled is not None:
            result = file_result(handled, req)

            def file_events():
                if handled["files"]:
                    yield json.dumps({"event": "file", "files": handled["files"]}) + "\n"
                yield json.dumps({"event": "done", **result}) + "\n"

            return StreamingResponse(file_events(), media_type="application/x-ndjson",
                                     headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        rag, ctx, early, note = graph_step(req, request)
        if early is not None:
            return StreamingResponse(iter([json.dumps({"event": "done", **early}) + "\n"]), media_type="application/x-ndjson",
                                     headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        if not service.ready:
            raise HTTPException(503, service.last_error or "Model is not loaded. Train ELVARN first.")
        check_web_search(req)
        history = [(t.user, t.assistant) for t in req.history] if req.history else None
        take_slot()

        def events():
            try:
                plan = web_plan(req, history, rag)
                if plan is not None and plan.query:
                    yield json.dumps({"event": "status", "status": "searching", "message": "Searching the web..."}) + "\n"
                found, from_kb = settle(rag, search_step(plan, req, request) if plan is not None else None, note)
                if from_kb:
                    yield json.dumps({"event": "sources", **found.fields(), **kb_extra(ctx)}) + "\n"
                elif (plan is not None and plan.query) or note:
                    yield json.dumps({"event": "sources", **found.fields()}) + "\n"
                if from_kb and found.sources:                         # records shown by plain code, word for word; the model only adds a summary
                    result = stored_answer(req, found, ctx, model_summary(req, found, history))
                    remember(req, request, result["text"], ctx)
                    yield json.dumps({"event": "done", **result}) + "\n"
                    return
                text, hist = model_input(req, found, history)
                rep, rep_source = repetition_settings(req, found.evidence_in_prompt)
                temp, topk, sampling_source = sampling_settings(req, found.evidence_in_prompt)
                for event in service.generate_stream(
                    text, mode=req.mode, max_new_tokens=req.max_new_tokens,
                    temperature=temp, top_k=topk,
                    greedy=req.greedy, history=hist, **rep,
                ):
                    if event["event"] == "done":
                        event.update(found.fields())
                        if from_kb:
                            event.update(kb_extra(ctx))
                        check_grounding(event, found, req.text, "the stored records" if from_kb else "the web results")
                        event["repetition_source"] = rep_source
                        event["sampling_source"] = sampling_source
                        remember(req, request, event.get("text", ""), ctx)
                    yield json.dumps(event) + "\n"
            except Exception as exc:  # reported to the UI instead of a broken stream
                yield json.dumps({"event": "error", "message": str(exc)}) + "\n"
            finally:
                give_slot()

        return StreamingResponse(events(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    # ---- a visitor can leave an email at the first question: a welcome mail goes to them, a note goes to the owner (needs SMTP_* settings)
    mailer = mail_service if mail_service is not None else Mailer.from_env()
    lead_state = {"seen": set(), "hits": {}}

    class LeadRequest(BaseModel):
        email: str = Field(max_length=254)
        consent: bool = False
        question: str = Field(default="", max_length=300)

    @app.get("/api/lead")
    def lead_status():
        return {"enabled": mailer is not None}

    @app.post("/api/lead")
    def lead(req: LeadRequest, request: Request):
        if mailer is None:
            raise HTTPException(503, "Email is not set up on this server yet.")
        if not req.consent:
            raise HTTPException(400, "Please tick the box to agree before we use your email.")
        address = clean_address(req.email)
        if address is None:
            raise HTTPException(422, "That email address does not look right.")
        who, now = visitor(request), time.time()
        recent = [t for t in lead_state["hits"].get(who, []) if now - t < 3600]
        if len(recent) >= 3:
            raise HTTPException(429, "Too many tries. Please wait a while.")
        lead_state["hits"][who] = recent + [now]
        if address in lead_state["seen"]:
            return {"ok": True, "sent": False}
        question = one_line(req.question, 200)
        try:
            mailer.send(address, "Welcome to ELVARN", WELCOME_TEXT.format(first=(f"Your first question was: {question}" + chr(10) + chr(10)) if question else ""))
            if mailer.owner:
                mailer.send(mailer.owner, "ELVARN: a new visitor left an email", f"Email: {address}" + chr(10) + f"First question: {question or '(none)'}" + chr(10))
        except MailError as exc:
            raise HTTPException(502, str(exc))
        lead_state["seen"].add(address)
        return {"ok": True, "sent": True}

    # ---- the owner gets a colourful greeting email each time they open a new chat (owner key from the browser, address from the server's settings)
    owner_secret = owner_key if owner_key is not None else os.environ.get("OWNER_KEY", "")
    new_chat_mail = {"last": 0.0, "sent": []}

    def owner_request(request):
        got = request.headers.get("x-owner-key", "")
        return len(owner_secret) >= 16 and hmac.compare_digest(got.encode(), owner_secret.encode())

    # ---- politician-by-politician data collection: owner only.
    # On the owner's own machine the routes run the collector directly. On the deployed (public) site there is no collector: the owner's computer runs
    # an agent (backend/pipeline/agent.py) that polls this site, and the routes below pass the owner's button presses to it and show what it reports.
    collect_failures = {}

    def collect_guard(request):
        """Owner key or 403. Too many wrong keys from one address are answered 429 for ten minutes (the key is long, but nobody should be able to try)."""
        who, now_ = visitor(request), time.time()
        count, first = collect_failures.get(who, (0, now_))
        if now_ - first > 600:
            count, first = 0, now_
        if count >= 8:
            raise HTTPException(429, "Too many wrong keys. Try again later.")
        if not owner_request(request):
            collect_failures[who] = (count + 1, first)
            raise HTTPException(403, "Not allowed.")
        collect_failures.pop(who, None)

    if public:
        from backend.pipeline.bridge import CollectBridge       # standard library only: the hosted site carries no collector packages
        bridge = CollectBridge()
        ASK = 12.0                                              # seconds the site waits for the agent to answer a question

        def with_request(request_obj):
            return {**bridge.status(), "request": request_obj}

        @app.get("/api/collect/status")
        def collect_status(request: Request):
            collect_guard(request)
            return bridge.status()

        @app.post("/api/collect/start")
        def collect_start(request: Request, body: Optional[StartRequest] = None):
            collect_guard(request)
            if not bridge.online():
                return bridge.status()
            got = bridge.call("start", start_args(body), wait=ASK)                     # returns once the agent has acted and reported (about a second)
            note = "Start sent to your collection computer." if got is not None else "Your collection computer has not answered yet; it will start when it does."
            return {**bridge.status(), "message": note}

        @app.post("/api/collect/stop")
        def collect_stop(request: Request):
            collect_guard(request)
            if not bridge.online():
                return bridge.status()
            got = bridge.call("stop", wait=ASK)
            note = "Stop sent to your collection computer." if got is not None else "Your collection computer has not answered yet; it will stop when it does."
            return {**bridge.status(), "message": note}

        @app.post("/api/collect/targeted")
        def collect_targeted(request: Request, body: TargetedRequest):
            collect_guard(request)
            if not bridge.online():
                return with_request({"ok": False, "message": "Your collection computer is offline. Start the agent there, then ask again."})
            got = bridge.call("targeted", body.model_dump() if hasattr(body, "model_dump") else body.dict(), wait=ASK)
            if got is None or "error" in (got or {}):
                return with_request({"ok": False, "message": "Your collection computer did not answer in time." if got is None else got["error"]})
            return with_request(got)

        @app.get("/api/collect/people")
        def collect_people(request: Request, status: Optional[str] = None, q: Optional[str] = None, limit: int = 50, offset: int = 0):
            collect_guard(request)
            got = bridge.call("people", {"status": status, "q": q, "limit": limit, "offset": offset}, wait=ASK)
            if not isinstance(got, dict) or "items" not in got:
                raise HTTPException(503, "Your collection computer did not answer (is the agent running?).")
            return got

        @app.get("/api/collect/person/{key:path}")
        def collect_person(key: str, request: Request):
            collect_guard(request)
            got = bridge.call("person", {"key": key}, wait=ASK)
            if got is None:
                raise HTTPException(503, "Your collection computer did not answer (is the agent running?).")
            if not got or "error" in got:
                raise HTTPException(404, "Unknown person.")
            return got

        class AgentPoll(BaseModel):
            report: Optional[dict] = None
            results: Optional[list] = None

        @app.post("/api/collect/agent/poll")
        def collect_agent_poll(request: Request, body: AgentPoll):
            """The agent's call: its latest status and answers in, the waiting commands out."""
            collect_guard(request)
            return {"commands": bridge.poll(body.report, body.results or [])}
    else:
        collect_state = {"monitor": collect_service}

        def collect_monitor(request):
            collect_guard(request)
            if collect_state["monitor"] is None:
                from backend.pipeline.monitor import Monitor            # imported here: the hosted site does not carry the pipeline's packages
                collect_state["monitor"] = Monitor(kb=kb, rag=rag_store)
            return collect_state["monitor"]

        @app.get("/api/collect/status")
        def collect_status(request: Request):
            return collect_monitor(request).status()

        @app.post("/api/collect/start")
        def collect_start(request: Request, body: Optional[StartRequest] = None):
            from backend.pipeline import remote_ops
            monitor = collect_monitor(request)
            got = remote_ops.start(monitor, start_args(body))
            return {**monitor.status(), **({"message": got["message"]} if isinstance(got, dict) and not got.get("ok", True) else {})}

        @app.post("/api/collect/stop")
        def collect_stop(request: Request):
            from backend.pipeline import remote_ops
            monitor = collect_monitor(request)
            remote_ops.stop(monitor)
            return monitor.status()

        @app.post("/api/collect/targeted")
        def collect_targeted(request: Request, body: TargetedRequest):
            """'collect C. Joseph Vijay ED cases': one person, some sources only. Either a sentence (text), or the person and sources picked from the candidates."""
            from backend.pipeline import remote_ops
            monitor = collect_monitor(request)
            answer = remote_ops.targeted(monitor, body.model_dump() if hasattr(body, "model_dump") else body.dict())                # first (it may start the collector), then the status that shows it
            return {**monitor.status(), "request": answer}

        @app.get("/api/collect/people")
        def collect_people(request: Request, status: Optional[str] = None, q: Optional[str] = None, limit: int = 50, offset: int = 0):
            return collect_monitor(request).people_page(status, q, limit, offset)

        @app.get("/api/collect/person/{key:path}")
        def collect_person(key: str, request: Request):
            found = collect_monitor(request).person_detail(key)
            if found is None:
                raise HTTPException(404, "Unknown person.")
            return found

    @app.get("/api/notify/status")
    def notify_status():
        return {"enabled": bool(mailer is not None and mailer.owner and len(owner_secret) >= 16)}

    @app.post("/api/notify/newchat")
    def notify_new_chat(request: Request, hour: Optional[int] = None):
        if mailer is None or not mailer.owner or not owner_request(request):
            raise HTTPException(403, "Not allowed.")
        now = time.time()
        new_chat_mail["sent"] = [t for t in new_chat_mail["sent"] if now - t < 86400]
        if now - new_chat_mail["last"] < 20 or len(new_chat_mail["sent"]) >= 40:
            return {"ok": True, "sent": False}              # opened chats in quick succession (or very many today): one mail is enough
        base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/") or str(request.base_url).rstrip("/")
        subject, text, html = new_chat_email(os.environ.get("OWNER_NAME", ""), hour if hour is not None and 0 <= hour <= 23 else time.localtime().tm_hour, base)
        try:
            mailer.send(mailer.owner, subject, text, html=html)
        except MailError as exc:
            raise HTTPException(502, str(exc))
        new_chat_mail["last"] = now
        new_chat_mail["sent"].append(now)
        return {"ok": True, "sent": True}

    @app.delete("/api/memory")
    def forget_memory(request: Request, session_id: Optional[str] = None):
        """Delete this user's remembered turns and entities right away (privacy)."""
        if remote is not None:
            remote.forget(session_key(session_id or visitor(request)))
        elif memory is not None:
            memory.forget(session_key(session_id or visitor(request)))
        return {"forgotten": True}

    search_wake = {"last": 0.0}

    def wake_visit():
        """Wake the sleeping free services (knowledge + search), each at most every 5 minutes. Only called for page loads: /api/health is polled
        by Render all day and must not keep them awake (free instance hours are limited)."""
        if remote is not None:
            wake_remote(remote)
        if (isinstance(provider, SearxngProvider) and not is_loopback_host(provider.base_url) and os.environ.get("ELVARN_SEARCH_WAKEUP", "1") != "0"
                and time.time() - search_wake["last"] > 300):
            search_wake["last"] = time.time()
            wake_up(provider.base_url)

    @app.get("/api/health")
    def health():
        return service.health()

    @app.get("/api/model")
    def model_info():
        info = service.info()
        if info is None:
            raise HTTPException(503, service.last_error or "Model not loaded.")
        return {**info, "modes": list(MODES)}

    install_file_routes(app, files, visitor)

    @app.post("/api/chat")
    def chat(req: GenerateRequest, request: Request):
        return run(req, request)

    # ---- ELVARN can phone the owner and talk (Twilio); the answers come from the same pipeline as the web chat. Off without TWILIO_* settings.
    def ask_for_call(text, history, session, request):
        turns = [Turn(user=u, assistant=a) for u, a in history] or None
        return run(GenerateRequest(text=text[:1000], mode="chat", max_new_tokens=80, greedy=True, web_search="auto", session_id=session, history=turns),
                   request).get("text", "")

    install_call_routes(app, ask_for_call, call_config)

    @app.post("/api/generate")
    def generate_endpoint(req: GenerateRequest, request: Request):
        return run(req, request)

    @app.post("/api/reload")
    def reload_model():
        if public:
            raise HTTPException(403, "Reloading is disabled on the public site.")
        try:
            return {"reloaded": True, "model": service.load()}
        except ModelNotReady as exc:
            raise HTTPException(503, str(exc))

    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        @app.get("/")
        def index():
            wake_visit()                       # someone opened the page: start waking the free services now, ready by the time they type
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()
