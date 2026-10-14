# Spark image for the knowledge-graph loader (backend/pipeline/spark_job.py).
FROM apache/spark:3.5.3-python3
USER root
RUN pip install --no-cache-dir "neo4j>=5" "pypdf>=4" rdflib redis
ENV PYTHONPATH=/opt/elvarn
WORKDIR /opt/elvarn
