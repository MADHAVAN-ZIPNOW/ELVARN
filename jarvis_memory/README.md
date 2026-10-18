# Jarvis Memory (Stage 1)

A personal memory assistant that lives in the cloud, so it works with your PC switched off.
Stage 1 = **save memories (typed or spoken), AI-organise them, search and filter them, edit them.**
Later stages add the hourly email (GitHub Actions), email replies (Gmail IMAP), the Android app and Daily Review.

It is a **separate, self-contained folder**: nothing in ELVARN's `backend/`, `frontend-react/` or `render.yaml` is used or changed.

## Free services only

| Job | Service | Cost |
|---|---|---|
| Database | **Neon** free Postgres | free |
| API + dashboard | **Render** free Docker web service | free |
| AI organising (title, summary, tags…) | **Groq** free tier | free |
| Hourly schedule (Stage 2) | **GitHub Actions** | free |
| Email replies (Stage 2) | **Gmail IMAP** | free |
| Media storage | none in Stage 1 (no R2) | – |

## Architecture

```
 phone / PC browser ──► Render (free): FastAPI + dashboard ──► Neon Postgres (one small table)
   type or speak            │   X-Owner-Key on every call           ▲
   (Web Speech API)         └──► Groq: extract title/summary/tags ──┘ (text only; failures never lose your memory)
```

## Deploy (about 15 minutes, all free, nothing to pay)

**1. Neon** — <https://neon.tech> → create a project → *Connection details* → turn on **Connection pooling** and copy the connection string (host contains `-pooler`, ends with `sslmode=require`). That is your `DATABASE_URL`. The tables are created automatically on first start.

**2. Owner key** — any random 16+ character secret. In PowerShell: `-join ((48..57)+(97..122) | Get-Random -Count 32 | % {[char]$_})`. Keep it private: it is the password to your memories.

**3. Groq key** — <https://console.groq.com/keys> (free). Without it memories are still saved, just without AI title/summary.

**4. Render** — New → **Web Service** → connect the `ELVARN` GitHub repo (the one with this folder) →
* Language/Runtime: **Docker**
* Dockerfile path: `./jarvis_memory/Dockerfile`, Docker context directory: `.`
* Instance type: **Free**; Health check path: `/api/jarvis/health`
* **Auto-deploy: off** (so it never redeploys when ELVARN changes; or limit it to the `jarvis_memory/` path)
* Environment variables: see the table below.

**5. Open it once with your key:** `https://<your-service>.onrender.com/?key=<your owner key>` — the browser remembers the key and hides it from the address bar.

### Environment variables

| Variable | Required | Notes |
|---|---|---|
| `DATABASE_URL` | yes | Neon **pooled** string |
| `JM_OWNER_KEY` | yes | 16+ characters; the API refuses to run without it |
| `GROQ_API_KEY` | recommended | free tier |
| `JARVIS_OWNER_NAME` / `JARVIS_OWNER_EMAIL` / `JARVIS_OWNER_PHONE` / `JARVIS_TIMEZONE` | no | defaults: Madhavan / – / – / Asia/Kolkata |
| `JM_LLM_MODEL` | no | default `qwen/qwen3.8-27b` (tested) |
| `JM_DB_SOFT_LIMIT_MB` | no | default **400**: writes are refused above this so the free database never fills (Neon free has been 0.5 GB; raise to 800 if yours is 1 GB) |
| `JM_RATE_LIMIT_PER_MINUTE`, `JM_ALLOWED_ORIGINS` | no | 60 / none |

Never commit real values (`.env` is git-ignored). Secrets only ever live in Render's environment settings.

## Free-tier behaviour to expect

* **Render free sleeps after ~15 min idle**; the first request afterwards takes ~30–60 s to wake it (the page says so if it can't connect). Stage 2's hourly GitHub Actions call keeps it awake while you are active.
* **Neon free suspends when idle** and wakes in about a second.
* **Groq free tier is rate limited.** When it says "slow down" the service waits as long as Groq asks, and if it still fails your memory is saved exactly as written, marked "AI unavailable", with a **Re-run AI** button.

## How small is the database?

One row per memory (tags/projects/people/next steps are arrays and tasks a JSON list on that row), text never stored twice, one search index. **Measured: about 1.5 KB per memory including all indexes** — roughly 650,000 memories per GB. The Settings page shows usage; `GET /api/jarvis/memory/stats` too. Deleted memories are hidden immediately and purged for good after 30 days.

*Deliberate difference from the original spec:* separate `memory_tags/projects/people/tasks` tables are replaced by arrays/JSON on the memory row (smaller, faster, no joins). Attachments, update requests, job runs and device tokens arrive with the stages that need them, so the free database holds only what is used.

## Using it

* **Memory tab:** type or tap 🎙️ **Speak** (browser speech recognition; edit the text before saving). Voice memories keep what was *heard* untouched as `raw_transcript` and your edited text as the memory.
* **Search tab:** words, `"exact phrase"`, `-exclude`, `or`; filters for date/time, source, project, person, tag, task status, sentiment, transcript/attachment; sort newest / oldest / best match; highlighted snippets; Load more.
* **Memory page:** original text (read-only, always preserved), AI summary and fields (editable), tasks, retry AI, delete.
* **Settings:** storage usage, download everything (JSON), delete everything (type DELETE).

**Never invents facts:** the model's answer is checked against your own words — names, projects and tags must appear in the text, tasks and next steps must use its words, summary sentences must be supported — and anything unsupported is dropped (what was dropped is recorded in the memory's metadata). Tested with prompt-injection text too.

## API (all need the `X-Owner-Key` header)

`POST/GET /api/jarvis/memory` · `GET/PATCH/DELETE /api/jarvis/memory/{id}` · `POST /api/jarvis/memory/{id}/extract` · `POST /api/jarvis/memory/search` · `GET/PATCH /api/jarvis/memory/settings` · `GET /api/jarvis/memory/stats` · `POST /api/jarvis/memory/export` · `DELETE /api/jarvis/memory/all?confirm=true` · public `GET /api/jarvis/health` (no data).
The prefix is `/api/jarvis/memory`, not `/api/memory`, because ELVARN already has a `DELETE /api/memory` route.

## Security

Owner key on every call (constant-time compare, fails closed if unset), brute-force lockout (10 wrong keys/min), per-address rate limit, 250 KB request cap, strict input validation, parameterised SQL only, owner-scoped queries, no docs/OpenAPI exposure, CORS closed by default, no content in logs, memory text always rendered as text (never HTML).

## Local development and tests

```powershell
cd jarvis_memory
python -m venv .venv ; .venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
docker run -d --name jm-test-pg -e POSTGRES_PASSWORD=jmtest -e POSTGRES_DB=jm_test -p 127.0.0.1:55432:5432 postgres:16-alpine
.venv\Scripts\python -m pytest tests -q          # 118 tests, real Postgres; never touches Neon
cd web ; npm install ; npm run build             # builds the dashboard into web/dist
```

Run it locally: set `DATABASE_URL`, `JM_OWNER_KEY` (and `GROQ_API_KEY`), then `python -m app.main` and open `http://127.0.0.1:8100/?key=<key>`.

## Roadmap

* **Stage 2** — hourly email (GitHub Actions → this API, quiet hours 22:00–09:00 IST, daily limits, retries, job log), email reply → memory via **Gmail IMAP** (only from your address, duplicate-proof).
* **Stage 3** — Flutter Android app (`com.zipnow.jarvis`), Firebase push, call-style full-screen alert, voice update. `google-services.json` goes in `mobile/android/app/`; the Firebase *service-account* key is only ever a Render environment variable.
* **Stage 4** — Daily Review (video/audio/screenshots) — needs storage; will use a free option and strict size limits.
* **Stage 5** — install scripts (`install_android.sh`, `run_dev.sh`), polish.
