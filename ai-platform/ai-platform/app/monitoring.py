import logging

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_seconds", "HTTP latency", ["method", "path"])
PREDICTIONS = Counter("fraud_predictions_total", "Fraud predictions", ["label"])
AGENT_RUNS = Counter("agent_runs_total", "Agent runs", ["status"])
RAG_QUERIES = Counter("rag_queries_total", "RAG queries", ["answered"])


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
