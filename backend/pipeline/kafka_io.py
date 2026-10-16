"""Kafka transport for collected documents: one JSON message per document on the topic below.

    KAFKA_BOOTSTRAP   broker address (default localhost:9094 from the host, kafka:9092 inside the compose network)
    KAFKA_TOPIC       default kb.raw_documents
"""
import json
import os

TOPIC = os.environ.get("KAFKA_TOPIC", "kb.raw_documents")
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9094")
# "localhost" resolves to ::1 first on this machine; the broker only listens on IPv4, so every connection first waited for a refused IPv6 attempt (about
# two seconds, which made status checks time out at random). Forcing IPv4 makes connecting instant.
CLIENT_OPTIONS = {"broker.address.family": "v4", "log_level": 0}


class KafkaSink:
    """Publishes messages and waits for the broker to acknowledge them (flush), so a collector never reports
    a document as collected before Kafka has it."""

    def __init__(self, bootstrap=None, topic=None):
        from confluent_kafka import Producer
        self.topic = topic or TOPIC
        self.producer = Producer({"bootstrap.servers": bootstrap or BOOTSTRAP, "enable.idempotence": True, "acks": "all",
                                  "message.max.bytes": 5_000_000, **CLIENT_OPTIONS})
        self.failed = []

    def _done(self, err, msg):
        if err is not None:
            self.failed.append(str(err))

    def send(self, msg):
        key = (msg.get("url") or msg.get("sha256") or "").encode("utf-8")
        self.producer.produce(self.topic, key=key, value=json.dumps(msg, ensure_ascii=False).encode("utf-8"), on_delivery=self._done)
        self.producer.poll(0)

    def flush(self, timeout=60):
        left = self.producer.flush(timeout)
        if left or self.failed:
            raise RuntimeError(f"Kafka delivery failed ({left} unsent; errors: {self.failed[:3]})")


def consume(group, bootstrap=None, topic=None, timeout=5.0, max_messages=None):
    """Read messages from the topic (used by tests and for debugging; the real consumer is the Spark job)."""
    from confluent_kafka import Consumer
    c = Consumer({"bootstrap.servers": bootstrap or BOOTSTRAP, "group.id": group, "auto.offset.reset": "earliest",
                  "enable.auto.commit": False, **CLIENT_OPTIONS})
    c.subscribe([topic or TOPIC])
    out = []
    try:
        while max_messages is None or len(out) < max_messages:
            m = c.poll(timeout)
            if m is None:
                break
            if m.error():
                raise RuntimeError(m.error())
            out.append(json.loads(m.value()))
    finally:
        c.close()
    return out
