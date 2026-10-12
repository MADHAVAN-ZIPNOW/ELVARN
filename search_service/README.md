# ELVARN search service (SearXNG behind a token gatekeeper)

Purpose: let the deployed ELVARN site do live web search. SearXNG is a free meta search engine. A public SearXNG
with no login would be abused, so it is hidden behind a small gatekeeper (`proxy.py`) that needs a secret token.

```
ELVARN service (Render)                 search service (Render, this folder, one Docker container)
 backend/search/providers/searxng.py     +-------------------------------------------------------+
   GET /search?q=...   -------------->   | proxy.py   0.0.0.0:$PORT   checks Bearer token,       |
   Authorization: Bearer <token>         |            rate limit, allow-listed parameters        |
                                         |               |  GET /search?...&format=json          |
                                         |               v                                       |
                                         | SearXNG    127.0.0.1:8081  (not reachable from outside)|
                                         +-------------------------------------------------------+
```

Files: `Dockerfile` (pinned SearXNG image), `settings.yml` (production SearXNG settings), `proxy.py` (gatekeeper, Python
standard library only), `entrypoint.sh` (starts both; if one dies the container exits so Render restarts it).

## What the gatekeeper allows
* `GET /healthz` - no token, answers `ok` without asking SearXNG (Render health check).
* `GET /readyz` - token needed, checks that SearXNG answers.
* `GET /search?q=...` - token needed. `format=json` is forced. Only `q` (1-300 chars), `language`, `safesearch` (0-2),
  `pageno` (1-5), `time_range` (day/week/month/year) and `categories` (general, news, science, it) are passed on.
* Everything else gets 404 (other paths) or 405 (other methods). No token -> 401 (same body for missing and wrong token).
* 429 too many requests (global limit), 503 too many searches at once (max 4), 504 SearXNG too slow (8 s), 502 SearXNG
  failed or answered with more than 1 MB.
* Logs contain only the status code, the duration and the kind of path. Never the token, the query or client addresses.
* No token configured (or shorter than 24 characters) -> the gatekeeper refuses to start.

## Render settings (create a NEW service; do not touch the ELVARN service)
| Setting | Value |
|---|---|
| Type | Web Service |
| Runtime | Docker |
| Root directory / build context | `search_service` |
| Dockerfile path | `./Dockerfile` (relative to the root directory; if Render asks for a path from the repo root: `search_service/Dockerfile`) |
| Plan | Free (512 MB RAM) |
| Health check path | `/healthz` |
| Env `SEARCH_PROXY_TOKEN` | secret, long random (for example `openssl rand -hex 32`) |
| Env `SEARXNG_SECRET` | secret, long random, a different value |
| Env `SEARCH_PROXY_RATE_PER_MINUTE` | optional, default 60 |

Render sets `PORT` itself. Do not set it.

## Settings on the ELVARN service
* `SEARXNG_BASE_URL` = `https://<your-search-service>.onrender.com`
* `SEARXNG_AUTH_TOKEN` = exactly the same value as `SEARCH_PROXY_TOKEN`

ELVARN only sends the token over `https://` (or to localhost). With `http://` to another host the search is switched off
and a warning is logged.

Cold start: the free plan sleeps after about 15 minutes without traffic and needs roughly 30-60 seconds to wake up. The
first search after a pause may time out; ELVARN then shows its honest "web search is not available / did not work"
message. ELVARN also pings `/healthz` once at start-up to wake the service early.

## Try it locally (needs Docker)
```
docker build -t elvarn-search search_service
docker run --rm --name elvarn-search-test -e SEARCH_PROXY_TOKEN=local-test-token-0123456789abcdef -e SEARXNG_SECRET=local-test-secret-0123456789abcdef -p 127.0.0.1:8090:10000 elvarn-search

curl http://127.0.0.1:8090/healthz                                   # ok
curl -i "http://127.0.0.1:8090/search?q=test"                        # 401
curl -i -H "Authorization: Bearer local-test-token-0123456789abcdef" "http://127.0.0.1:8090/search?q=python+language"   # 200 + JSON
curl -i -H "Authorization: Bearer local-test-token-0123456789abcdef" http://127.0.0.1:8090/config                         # 404
```
