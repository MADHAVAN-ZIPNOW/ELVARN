# ELVARN: a grounded, cited Q&A system for Tamil Nadu corruption-case records

**ELVARN** is a small language model (17 million parameters, trained from scratch) wrapped in a complete system that answers questions about
**Tamil Nadu corruption cases (1947 to 2026)** from real public records, **shows its sources for every answer, and refuses instead of inventing**.
It was built by Madhavan (madhavan.ai.research@gmail.com) as a learning and portfolio project in October 2026, and runs live on free hosting.

- Live demo: https://elvarn.onrender.com (free hosting: the first question after a quiet period can take up to a minute while services wake up)
- Code: https://github.com/MADHAVAN-ZIPNOW/ELVARN

---

## 1. The idea in one paragraph

Records about corruption cases are scattered: court judgments on court sites, vigilance press releases (DVAC), CBI press releases, news, Wikipedia.
Finding everything about one person or case is slow. ELVARN collects these public records every night, stores them in three specialised
databases, and answers questions by **retrieving the matching records first and letting the model write from them**, never from memory.
Because a 17M model cannot recall facts, it is only ever asked to read and copy what retrieval gives it. If the records do not contain the answer,
it says "Not found in my data." Every answer carries source cards (title, site, snippet, link) and a label saying where it came from.

## 2. What it can do today (all live)

| Feature | What it does |
|---|---|
| Cited answers | Each answer shows source cards on the right (like Perplexity) and a label: stored records / live web search / model alone / nothing |
| Refusal | Corruption questions with no matching record get "Not found in my data." |
| Answer check | The answer is compared with its notes (numbers, names, repeats). If it does not match, the original record text is shown word for word |
| Chat | Greetings, small talk, who ELVARN is, remembering your name; replies are warm and short with emoji |
| Voice conversation | Talk to ELVARN: it greets, listens, sends your question by itself when you stop speaking, answers on the page and aloud, listens again. Works on every page, floating panel, sound cues |
| Live web search | General and time-sensitive questions go to a self-hosted SearXNG |
| Sessions | Search, filter by date, pin, rename, download (PDF / Word / TXT) |
| Files | Upload txt/md/csv/json/html/docx and convert between formats |
| Email | Optional email card after the first answer (welcome mail); a colourful HTML greeting for the owner at each new chat |
| Phone call (built, off) | ELVARN phones the owner's number and talks, through Twilio; needs the owner's Twilio details |

## 3. Architecture

```
                         WEB SOURCES                                   NIGHTLY PIPELINE (Airflow, 02:00 IST)
  DVAC (dvac.tn.gov.in)  CBI (cbi.gov.in)  Supreme Court + Madras HC (open judgment archives)  Wikipedia (Tamil Nadu pages)
        |                     |                         |                                             |
        +---- collectors (polite: robots.txt, delays, size caps, "seen" state, SHA-256 dedupe) --------+
                                         |  raw files in backend/data/kb_raw (content-addressed)
                                         v
                     Kafka topic kb.raw_documents ---> Spark job: extract text, cut passages, propose facts
                                         |
                    +--------------------+--------------------+
                    v                    v                    v
                 Neo4j               Chroma                Redis
          (knowledge graph:     (every passage +      (per-user memory +
           facts, ontology,      vector, semantic      retrieval cache)
           citations)            search)
                    \                    |                    /
                     \                   v                   /
                      +------ GraphRAG: vector search || graph traversal -> merge and rank ------+
                                         |
   question --> router --> [stored records | live web search | model alone] --> 17M model writes from the notes
                                         |
                      answer check (numbers, names, repeats) --> answer or the source text word for word
                                         |
                                  React web app (cited, voice, sessions)
```

**Hosted design (free):** the chat site, a **knowledge service** (a token-protected Docker container holding a read-only snapshot of the knowledge
base, an embedded Chroma and a local Redis) and a **search service** (SearXNG) are three Render services. No outside accounts are needed for the data.
The snapshot is exported from the local databases (`knowledge_service/snapshot/`, last export 14 October 2026).

## 4. The three databases (separate on purpose)

| Store | Role | Why it exists |
|---|---|---|
| **Neo4j 5** (graph) | Documents, passages, people, cases, agencies, facts and their citations; ontology (`ontology.ttl`); each fact has an assertion level (alleged / reported / court finding) and a review status | Facts, relationships and provenance. The user rejected SQL, so a graph database holds the structure |
| **Chroma** (vector) | Every passage with an embedding (MiniLM, 384-d) | Finds passages by meaning |
| **Redis** | Short-term memory per user (so "what was the outcome?" works) and a retrieval cache keyed by question and knowledge-base version | Follow-ups and speed |

## 5. The model

- **Architecture:** decoder-only Transformer in PyTorch: 6 layers, 12 attention heads, embedding 384, feed-forward 1536, context 512 tokens, vocabulary 16,384
  (byte-level BPE tokenizer), tied embeddings; **17,152,000 parameters**.
- **Training history:** pretrained from scratch on 9.1M tokens (general English, domain text, a little chat); context extended to 512; fine-tuned to answer
  *from notes* (a notes + question + answer format with refusals); then fine-tuned again on chat and on knowledge-base notes.
- **Why the notes format matters:** the model is trained to copy and cite from "Web notes" (web results) or "Stored records" (knowledge-base notes) given in the prompt,
  and to say so when the notes do not help.
- **Training on a GPU:** a separate CUDA environment trains at about 0.25 s/step on an RTX 2050 (4 GB); on the CPU it was 5 to 13 s/step.
- **Data for the latest fine-tune:** about 9.9M tokens: all notes examples (including the knowledge-base passages) and about 5M tokens of chat
  (open datasets: HuggingFaceTB everyday-conversations, google Synthetic-Persona-Chat, allenai SODA, OpenAssistant OASST2; plus ELVARN's own greeting,
  small-talk and name-memory examples written from fixed phrase lists, no real people).

### Measured results (hold-out set built from validation topics only)

| Test | Before the chat + notes fine-tune | After |
|---|---|---|
| Key facts of the notes found in the answer | 22% | 66 to 71% |
| Fully correct answers | 4% | 52 to 56% |
| Wrong refusals (notes had the answer) | 25% | 8 to 15% |
| Answers with invented numbers or names | 47% | 13 to 17% (and about 0% of answers that pass the answer check) |
| Chat prompts answered sensibly (13 test prompts) | almost none ("The notes do not say.") | 12 of 13 |

## 6. GraphRAG and routing

1. **Understand:** find entities (people, cases, agencies), the intent, and whether it is a follow-up (using Redis memory).
2. **Retrieve in parallel:** Chroma vector search and Neo4j graph traversal (2 hops) run at the same time.
3. **Merge and rank:** the best fact and the best passage go first; per-source caps stop one document crowding out the rest; headings and bare links are dropped.
4. **Decide the route:** a corruption question with no match is refused; other informational questions go to live web search; chat stays with the model.
5. **Write and check:** the model writes from the notes; the answer check (`backend/knowledge/groundcheck.py`) verifies every number and name against the notes and rejects
   repeats; on a mismatch the app shows the original record text, labelled as such.

## 7. Data

- **Knowledge base content (14 October 2026 snapshot):** 7,024 graph nodes, 5,872 relationships, 9,781 searchable passages.
- **Collected raw files:** 1,148 files (277 MB): 1,103 PDFs (1,015 readable = about 1.2M tokens, 68 scans needing OCR, 20 unreadable) and 45 web pages.
- **Training text exported from the knowledge base:** 4,444 passages, about 1.4M tokens: Madras High Court judgments 2,226 passages, Wikipedia (Tamil Nadu pages) 1,764,
  DVAC 310, CBI 137, Supreme Court 7. Only passages the model has not already seen are exported.
- **Totals:** about 15M unique tokens of training text (9.1M original + 1.4M knowledge base + about 5M chat). A model this size would ideally have about 340M, so
  the model's own wording stays limited; retrieval and citations are the strength.
- **Licences:** listed in `DATA_SOURCES.md` (CC BY-SA, CC BY, Apache 2.0, public records). Collectors send a generic user agent, respect robots.txt and rate-limit themselves.

## 8. Engineering quality

- **907 automated tests** (backend, pipeline, retrieval, answer check, deploy script, email, phone call, export, a React build check).
- A **React + Vite** frontend split into components, hooks and CSS files; the built `dist/` is committed (the host has no Node build step); the earlier single-file page is the fallback.
- **Deploy by script:** `scripts/deploy_render.py` creates the knowledge service through Render's API, wires the site to it and waits for it to be healthy.
- **Security:** token-protected knowledge service (https only), per-visitor rate limits, a concurrency cap, owner key for owner-only actions, signature check on phone webhooks,
  secrets only in a git-ignored `.env`, input validation (email header injection, file types, sizes), answer text rendered as text never as HTML.
- **Honesty by design:** refusals, source cards, "Answered from" labels, assertion levels (alleged / reported / court finding), and a clear disclaimer that allegations are not proof of guilt.

## 9. Repository map

| Path | What is there |
|---|---|
| `backend/model/`, `backend/training/` | The Transformer, tokenizer, training and GPU fine-tuning scripts |
| `backend/knowledge/` | Ontology, Neo4j / Chroma stores, GraphRAG, answer check, snapshot export |
| `backend/pipeline/`, `dags/`, `docker/` | Collectors, Kafka, Spark, Airflow nightly DAG and the compose files |
| `backend/search/`, `search_service/` | Live web search router and the SearXNG service |
| `backend/api/` | FastAPI server (chat, streaming, files, export, email, voice call) |
| `backend/data_tools/` | Dataset builders (notes, chat, open datasets) and the chat-data collector |
| `backend/eval/` | Evaluation scripts (notes quality, chat prompts) |
| `knowledge_service/` | The free self-contained hosted knowledge service and its snapshot |
| `frontend-react/` | The React app (and `frontend/` the earlier single-file page) |
| `scripts/` | Deploy and settings scripts, the Google mail relay template |

## 10. Honest limits

- The model's own wording is weak (17M parameters, about 15M tokens): it copies and cites well, but cannot write polished summaries; broad questions fall back to the record text.
- No facts are human-approved yet (about 1,400 proposed facts wait in the review queue).
- Web search quality varies; ambiguous questions can find the wrong page.
- The live site reads a frozen snapshot (14 October 2026) and the nightly collection is paused, so the hosted knowledge does not grow by itself.
- Free hosting sleeps when idle; scans and Tamil-encoded PDFs need OCR that is not built yet.

## 11. What is next

Turn the nightly collection back on with an automatic snapshot refresh; fix web-search ambiguity; approve facts through the review queue; OCR for scanned PDFs and Tamil;
more sources (CAG reports, Assembly records); a larger model and far more data on a cloud GPU; the WhatsApp or phone-call channel.

## 12. How to run it

```
# databases (Docker): Neo4j, Chroma, Redis, SearXNG
powershell scripts/start_stack.ps1
# the app (loads NEO4J_/CHROMA_/REDIS_ variables from .env)
python -m uvicorn backend.api.server:app --port 8000        # http://127.0.0.1:8000
# the frontend: cd frontend-react && npm install && npm run build
# tests (against a disposable Neo4j on port 7688, never the real database)
NEO4J_TEST_URI=bolt://localhost:7688 python -m pytest backend/tests -q
```

*Written 15 October 2026.*
