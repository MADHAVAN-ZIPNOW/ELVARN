# ELVARN knowledge service: GraphRAG over the knowledge graph + vector passages + memory, behind a secret token.
# Like search_service/: ONE self-contained container. It bundles a snapshot of the knowledge base (knowledge_service/snapshot/),
# fills an embedded Chroma from it at BUILD time, and starts its own Redis, so it needs no other service and no outside account.
# (Set NEO4J_URI and friends instead, and it runs against live stores: see README.)
#
# Build context = the REPOSITORY ROOT (it needs backend/knowledge and backend/usermemory); the root .dockerignore is an allowlist.
#   docker build -f knowledge_service/Dockerfile -t elvarn-knowledge .
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends redis-server bash && rm -rf /var/lib/apt/lists/*

COPY knowledge_service/requirements.txt /app/knowledge_service/requirements.txt
RUN pip install --no-cache-dir -r /app/knowledge_service/requirements.txt

# Download the embedding model INTO the image, so a cold start does not fetch 80 MB (and works without internet access to it).
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as D; print(len(D()(['warm up'])[0]))"

COPY knowledge_service /app/knowledge_service
COPY backend/knowledge /app/backend/knowledge
COPY backend/usermemory /app/backend/usermemory

# Fill the embedded Chroma from the snapshot once, now (its embeddings are in the snapshot, so no model runs and start-up stays instant).
ENV CHROMA_PATH=/app/chroma_data
RUN python -m backend.knowledge.snapshot check --dir /app/knowledge_service/snapshot \
 && python -m backend.knowledge.snapshot load --dir /app/knowledge_service/snapshot --chroma-path /app/chroma_data

# Not root. The embedding model that the root user downloaded is copied to this user's cache.
RUN useradd --create-home --uid 10001 elvarn \
 && mkdir -p /home/elvarn/.cache && cp -r /root/.cache/chroma /home/elvarn/.cache/chroma \
 && chown -R elvarn /app /home/elvarn/.cache && chmod +x /app/knowledge_service/start.sh
USER elvarn

# Render sets PORT itself; 10000 is only the default for local runs.
ENV PORT=10000
EXPOSE 10000
CMD ["/bin/bash", "/app/knowledge_service/start.sh"]
