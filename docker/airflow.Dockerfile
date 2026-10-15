# Airflow image that can run the collectors (backend/pipeline), the Wikipedia loader and the Neo4j -> Chroma repair.
FROM apache/airflow:2.10.5-python3.11
# Airflow's own environment: the collectors (Kafka producer, S3 parquet reader, PDF text, Neo4j, Redis).
RUN pip install --no-cache-dir "confluent-kafka>=2.5" "neo4j>=5" "pypdf>=4" "pyarrow>=15" redis
# Chroma's client conflicts with Airflow's pinned packages, so it lives in its own virtualenv; the tasks that need it
# (load_wikipedia, check_stores) run there as `python -m backend.knowledge...`.
RUN python -m venv /home/airflow/venv-rag  && /home/airflow/venv-rag/bin/pip install --no-cache-dir chromadb==1.5.9 "neo4j>=5" "pypdf>=4" redis
ENV PYTHONPATH=/opt/elvarn
# Chroma downloads its small embedding model into ~/.cache/chroma the first time; the folder must belong to the airflow user
RUN mkdir -p /home/airflow/.cache/chroma
