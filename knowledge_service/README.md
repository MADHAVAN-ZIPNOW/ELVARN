# ELVARN knowledge service

The Tamil Nadu corruption knowledge, as a separate service behind a secret token: the same pattern as `search_service/`.
It runs the GraphRAG pipeline (graph facts from **Neo4j** + passages from **Chroma** + memory/cache in **Redis**) and hands the
chat site a ranked, cited context. The chat site then lets the model write the answer from it.

```
ELVARN site (Render, small, unchanged)        knowledge service (Render, this folder, one Docker container)
  backend/knowledge/remote.py                  +----------------------------------------------------------+
   POST /v1/context  ----------------------->  | app.py   checks Bearer token, rate limit, input checks    |
   Authorization: Bearer <token>               |            |-- Neo4j  (graph facts, relationships)         |
                                               |            |-- Chroma (passages + embeddings)               |
   <-- ranked, cited items (facts + passages)  |            '-- Redis  (user memory + retrieval cache)       |
                                               +----------------------------------------------------------+
```
The chat site needs **no** neo4j / chromadb / redis packages and no embedding model: it stays as small as today.

## What the service allows
* `GET /healthz` - no token, answers `ok` without touching any store (Render health check).
* `GET /readyz` - token; reports which of graph / vector / memory are reachable.
* `POST /v1/context` - token; `{"question" (1-1000 chars), "session" (optional, letters/digits/-/_ up to 64)}`.
  Answers `{items, notes, u, populated}` and nothing else.
* `POST /v1/remember`, `DELETE /v1/memory?session=...` - token; one user's memory.
* Everything else: 404 / 405 (the API docs pages are switched off). Missing or wrong token: the same 401 for both.
* 429 too many requests (global limit), 503 too many at once, 502 retrieval failed, 503 no store reachable at all.
* Logs hold only status codes, counts and timings: never the token, the question text or the session id.
* No `KNOWLEDGE_TOKEN` (or one shorter than 24 characters): the service refuses to start.
* A store that is down at start-up is left out; the answer is built from the ones that work.

## Measured (local Docker test against the real data, 2026-10-14)
| | |
|---|---|
| Image size | about 1.5 GB (mostly the embedding runtime; the 80 MB model is baked in, so no download at start) |
| Memory | **121 MB idle, 198 MB after several queries** (fits Render's 512 MB plans) |
| Speed | 0.2-0.5 s per question, 0.02 s when the answer is cached |
| Data size | Neo4j 63,891 nodes / 55,690 relationships (53 MB of text); Chroma 53,866 passages (about 470 MB on disk) |

## Render settings (create a NEW service; do not touch the ELVARN or search services)
| Setting | Value |
|---|---|
| Type | Web Service |
| Runtime | Docker |
| Root directory | leave EMPTY (the build needs `backend/knowledge` and `backend/usermemory` from the repo root) |
| Dockerfile path | `knowledge_service/Dockerfile` |
| Plan | Free (512 MB) is enough by the measurement above; a free service sleeps after ~15 min (see cold start) |
| Health check path | `/healthz` |
| Env `KNOWLEDGE_TOKEN` | secret, long random (for example `openssl rand -hex 32`), at least 24 characters |
| Env `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | your hosted Neo4j (for Aura the URI looks like `neo4j+s://xxxx.databases.neo4j.io`) |
| Env `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE` | Chroma Cloud; or for a Chroma server you host: `CHROMA_HOST`, `CHROMA_PORT=443`, `CHROMA_SSL=1`, optional `CHROMA_TOKEN` |
| Env `REDIS_URL` | your hosted Redis, `rediss://:password@host:6380/0` (Render Key Value gives you one) |
| Env `KNOWLEDGE_RATE_PER_MINUTE`, `KNOWLEDGE_MAX_CONCURRENT` | optional, defaults 120 and 4 |

Render sets `PORT` itself. Do not set it. Check the free-tier limits of each hosted store before you rely on them (they change).

## Settings on the ELVARN site service
* `KNOWLEDGE_URL` = `https://<your-knowledge-service>.onrender.com`
* `KNOWLEDGE_TOKEN` = exactly the same value as on the service

The site only sends the token over `https://` (or to localhost). With plain `http://` to another host the client refuses to
start, a warning is logged, and corruption questions get the honest "knowledge base unavailable" message.
When `KNOWLEDGE_URL` is set it replaces any local store settings on the site.

Cold start: on a free plan the service sleeps and needs about 30-60 seconds to wake. The site pings `/healthz` once at start-up
to wake it early. While it is asleep a corruption question gets "The knowledge base is unavailable right now, so I can't answer
this reliably."; every other question keeps working (web search, plain model).

## Moving your data to the hosted stores (one time, from this computer)
The local stores keep running; this copies them. Put the hosted details in `.env` with a `TARGET_` prefix:
```
TARGET_NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io   TARGET_NEO4J_USER=neo4j   TARGET_NEO4J_PASSWORD=...
TARGET_CHROMA_API_KEY=...  TARGET_CHROMA_TENANT=...  TARGET_CHROMA_DATABASE=...
```
```
python -m backend.knowledge.migrate_stores --neo4j --chroma --dry-run     # only report counts
python -m backend.knowledge.migrate_stores --neo4j                        # graph: refuses a target that is not empty
python -m backend.knowledge.migrate_stores --chroma                       # copies the passages WITH their embeddings, skips ids already there
python -m backend.knowledge.migrate_stores --verify                       # compare counts
```
After that, point the nightly pipeline (Airflow / Spark containers) at the hosted stores by setting `NEO4J_*`, `CHROMA_*`,
`REDIS_URL` in `.env`, so new documents go straight to the hosted copies.

## Try it locally (needs Docker; this is exactly what was measured above)
```
docker build -f knowledge_service/Dockerfile -t elvarn-knowledge .
docker run -d --name elvarn-knowledge-test -p 127.0.0.1:10000:10000 --env-file knowledge.env elvarn-knowledge
curl http://127.0.0.1:10000/healthz
curl -X POST http://127.0.0.1:10000/v1/context -H "Authorization: Bearer $KNOWLEDGE_TOKEN" -H "Content-Type: application/json" \
     -d '{"question":"village officer trapped for a bribe"}'
```
(`knowledge.env` holds `KNOWLEDGE_TOKEN`, `NEO4J_*` with host `host.docker.internal`, `CHROMA_HOST=host.docker.internal`, `CHROMA_PORT=8001`, `REDIS_URL`.)
Then start the site with only `KNOWLEDGE_URL=http://127.0.0.1:10000` and `KNOWLEDGE_TOKEN=...` (no store variables) to see the Render setup.

## Honest limits
* The 17M model still writes weak answer text; the response always carries the cited facts and sources next to it.
* Only facts a person approved appear as graph facts; until the review queue is worked through, the graph branch adds nothing.
* A public site that shows allegations about named people needs the facts reviewed first.
