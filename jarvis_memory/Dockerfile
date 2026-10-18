# Jarvis Memory: one small container (API + built dashboard). Build context = the REPOSITORY ROOT:
#   docker build -f jarvis_memory/Dockerfile -t jarvis-memory .
# On Render: New > Web Service > Docker, Dockerfile path ./jarvis_memory/Dockerfile, Docker context ., plan Free.

FROM node:20-alpine AS web
WORKDIR /web
COPY jarvis_memory/web/package.json jarvis_memory/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY jarvis_memory/web/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 MALLOC_ARENA_MAX=2
WORKDIR /srv
COPY jarvis_memory/requirements.txt .
RUN pip install -r requirements.txt
COPY jarvis_memory/app ./app
COPY jarvis_memory/migrations ./migrations
COPY --from=web /web/dist ./web/dist
RUN useradd --system --no-create-home jm && chown -R jm /srv
USER jm
ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT} --no-server-header --limit-concurrency 40 --workers 1"]
