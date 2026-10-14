# ELVARN knowledge service: GraphRAG over Neo4j (facts) + Chroma (passages) + Redis (memory), behind a secret token.
# Build context = the REPOSITORY ROOT (it needs backend/knowledge and backend/usermemory); the root .dockerignore is an allowlist.
#   docker build -f knowledge_service/Dockerfile -t elvarn-knowledge .
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY knowledge_service/requirements.txt /app/knowledge_service/requirements.txt
RUN pip install --no-cache-dir -r /app/knowledge_service/requirements.txt

# Download the embedding model INTO the image, so a cold start does not fetch 80 MB (and works without internet access to it).
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as D; print(len(D()(['warm up'])[0]))"

COPY knowledge_service /app/knowledge_service
COPY backend/knowledge /app/backend/knowledge
COPY backend/usermemory /app/backend/usermemory

# Not root.
RUN useradd --create-home --uid 10001 elvarn && chown -R elvarn /app
USER elvarn
# The user's own home now holds nothing: copy the model the root user downloaded so the first request does not download it again.
USER root
RUN mkdir -p /home/elvarn/.cache && cp -r /root/.cache/chroma /home/elvarn/.cache/chroma && chown -R elvarn /home/elvarn/.cache
USER elvarn

# Render sets PORT itself; 10000 is only the default for local runs.
ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn --factory knowledge_service.app:create_from_env --host 0.0.0.0 --port ${PORT} --workers 1"]
