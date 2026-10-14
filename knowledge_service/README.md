# ELVARN knowledge service

The Tamil Nadu corruption knowledge as a **separate, token-protected Docker service on Render**: the same pattern as `search_service/`.
It runs the GraphRAG pipeline (graph facts + passage search + user memory/cache) and hands the chat site a ranked, cited context.
The chat site then lets the model write the answer from it.

```
ELVARN site (Render, unchanged size)          knowledge service (Render FREE plan, this folder, ONE Docker container)
  backend/knowledge/remote.py                  +------------------------------------------------------------+
   POST /v1/context  ----------------------->  | app.py   checks Bearer token, rate limit, input checks      |
   Authorization: Bearer <token>               |            |-- graph    in memory, from snapshot/graph.jsonl.gz |
                                               |            |-- Chroma   embedded, filled from snapshot/        |
   <-- ranked, cited items (facts + passages)  |            '-- Redis    its own, 127.0.0.1 only              |
                                               +------------------------------------------------------------+
```
No other service and no outside account is needed, and it costs nothing: the data is **baked into the image** (like SearXNG's settings).
The chat site needs no neo4j / chromadb / redis packages and no embedding model.

## How it works, and the one trade-off
Neo4j needs a Java process that does not fit in a free 512 MB container. So the hosted service serves a **read-only snapshot** of your
knowledge base (`knowledge_service/snapshot/`, about 14 MB, committed to the repo):

| File | Content |
|---|---|
| `graph.jsonl.gz` | the lean graph exported from Neo4j: official documents, proposed + approved facts, entities (7,024 nodes) |
| `passages.jsonl.gz` + `vectors.npz` | 9,781 passages with their embeddings: all official documents, your own notes, and the Wikipedia articles about Tamil Nadu corruption |
| `manifest.json` | counts and a checksum of each file (checked at build and start) |

Left out of the hosted copy: general Wikipedia (the live web search answers general questions). **Your Neo4j + Chroma on your computer stay the
source of truth**; the snapshot is rebuilt from them. A test runs Neo4j and the in-memory graph on the same data and requires identical answers.

## Deploy it exactly like the search service (Render dashboard)
| Setting | Value |
|---|---|
| Type | Web Service |
| Runtime | Docker |
| Root directory | leave **empty** (the build needs `backend/knowledge`, `backend/usermemory` and `knowledge_service` from the repo root) |
| Dockerfile path | `knowledge_service/Dockerfile` |
| Branch | `main` |
| Plan | **Free** (512 MB). Measured with a hard 512 MB limit: 190 MB idle, 237 MB after queries, ready in 2 seconds |
| Health check path | `/healthz` |
| Env `KNOWLEDGE_TOKEN` | secret, long random (for example `openssl rand -hex 32`), at least 24 characters. **The only setting it needs.** |
| Optional env | `KNOWLEDGE_RATE_PER_MINUTE` (default 120), `KNOWLEDGE_MAX_CONCURRENT` (default 4) |

Render sets `PORT` itself. Do not set it. The first build downloads about 1 GB of Python packages (10 to 15 minutes on the free plan).

Then on the **ELVARN site service** add:
* `KNOWLEDGE_URL` = `https://<your-knowledge-service>.onrender.com`
* `KNOWLEDGE_TOKEN` = exactly the same value

The site only sends the token over `https://` (or to localhost). With plain `http://` to another host the client refuses to start, a warning
is logged, and corruption questions get the honest "knowledge base unavailable" message.

Cold start: on a free plan the service sleeps after about 15 minutes and takes about 30 to 60 seconds to wake. The site pings `/healthz` once at
start-up to wake it early. While it sleeps a corruption question gets "The knowledge base is unavailable right now, so I can't answer this
reliably."; everything else (web search, plain model) keeps working.

**Recommended Render setting:** on the ELVARN **site** service set *Build Filters > Ignored Paths* to `knowledge_service/snapshot/**` so refreshing the
snapshot does not redeploy the site.

## Refreshing the snapshot (when new documents or approved facts exist)
On your computer, with the local stores running:
```
python -m backend.knowledge.reindex                      # make sure every passage is in Chroma
python -m backend.knowledge.snapshot export              # rewrites knowledge_service/snapshot/ (same data gives byte-identical files)
python -m backend.knowledge.snapshot check               # verifies the files against the manifest
git add knowledge_service/snapshot && git commit -m "Refresh knowledge snapshot" && git push
```
Render rebuilds the knowledge service from the new commit. Nothing runs on Render at night: the nightly collection stays on your computer.

## Endpoints
* `GET /healthz` - no token, answers `ok` without touching any store (Render health check).
* `GET /readyz` - token; says which of graph / vector / memory are reachable.
* `POST /v1/context` - token; `{"question" (1-1000 chars), "session" (optional, letters/digits/-/_ up to 64)}`; answers `{items, notes, u, populated}` and nothing else.
* `POST /v1/remember`, `DELETE /v1/memory?session=...` - token; one user's memory.
* Everything else: 404 / 405 (the API docs pages are switched off). A missing or wrong token gives the same 401.
* 429 too many requests, 503 too many at once, 502 retrieval failed, 503 no store reachable.
* Logs hold only status codes, counts and timings: never the token, the question or the session id.
* No `KNOWLEDGE_TOKEN` (or one shorter than 24 characters): the service refuses to start.

## Measured (local Docker, hard 512 MB limit, your real data)
| | |
|---|---|
| Image | about 1.9 GB (the embedding runtime and the 80 MB model are inside) |
| Memory | 190 MB idle, 237 MB after queries, never killed |
| Start | ready in 2 seconds (the vectors are loaded when the image is built) |
| Speed | 0.2 to 0.45 s per question |
| Agreement with your local Neo4j + Chroma | the same top source on 9 of 10 test questions; the tenth was a vague 3-word question where the two returned different loosely related passages |

## Limits you should know
* The 17M model still writes weak answer text; the response always carries the cited facts and sources beside it.
* Only facts a person approved appear as graph facts; until the review queue is worked through the graph adds nothing.
* A public site that shows allegations about named people needs those facts reviewed first.
* Memory and cache live in the container's own Redis and are lost whenever it restarts (that is fine for a cache).

## Other ways to run it (optional)
* **Live stores:** set `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`, `CHROMA_HOST`/`CHROMA_PORT` (or `CHROMA_API_KEY`/`CHROMA_TENANT`/`CHROMA_DATABASE`)
  and `REDIS_URL` and it uses them instead of the snapshot (it prefers live stores whenever `NEO4J_URI` is set).
  `python -m backend.knowledge.migrate_stores --neo4j --official-only` and `--chroma` copy your local data to hosted stores (`TARGET_` prefix variables).
* **Try it locally:** `docker build -f knowledge_service/Dockerfile -t elvarn-knowledge .`, then
  `docker run --memory 512m -p 127.0.0.1:10000:10000 -e KNOWLEDGE_TOKEN=<24+ characters> elvarn-knowledge`; start the site with only
  `KNOWLEDGE_URL=http://127.0.0.1:10000` and `KNOWLEDGE_TOKEN` set (no store variables) to see the Render setup.
