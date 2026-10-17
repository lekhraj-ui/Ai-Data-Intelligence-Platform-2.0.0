# Deployment and Operations

The service is packaged with a Dockerfile and started with docker compose together with a PostgreSQL database and Prometheus.

The GitHub Actions pipeline runs ruff linting, the pytest suite, the evaluation gate and a Docker image build on every push.

Monitoring exposes Prometheus metrics at the /metrics endpoint: request counts, latency histograms, fraud prediction counts and agent run counts.

The /health endpoint reports service status and is used by the container health check.
