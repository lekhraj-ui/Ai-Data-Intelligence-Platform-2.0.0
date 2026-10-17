# AI Platform: Python → ML → GenAI → Agentic AI → AI Engineering

One FastAPI service that implements every layer of the roadmap. It runs **fully offline** by default (mock LLM + TF-IDF embeddings); set an Anthropic key to switch to a real LLM.

| Layer | Where it lives |
|---|---|
| Python / FastAPI / SQL / PostgreSQL | `app/main.py`, `app/db.py` (SQLAlchemy; SQLite locally, PostgreSQL in Docker) |
| Scikit-learn, feature engineering, evaluation | `app/ml/` (RandomForest fraud model, CV, ROC AUC / PR / F1) |
| Anomaly detection | `app/ml/train.py` (IsolationForest on legitimate transactions) |
| LLM APIs, prompts | `app/genai/llm.py`, `prompts.py` (mock or Anthropic) |
| Embeddings, vector DB, RAG | `app/genai/embeddings.py`, `vectorstore.py`, `rag.py` |
| Tool calling, routing | `app/agents/tools.py`, `router.py` |
| Multi-agent + orchestration | `app/agents/multi_agent.py` (planner, researcher, writer, critic), `orchestrator.py` |
| Docker, CI/CD | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |
| Monitoring | `app/monitoring.py` → `/metrics`, `monitoring/prometheus.yml` |
| Evaluation | `app/evaluation/run.py` (CI quality gate), `tests/` |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
make install          # pip install -r requirements-dev.txt
make train            # optional: trains on first start anyway
make run              # http://localhost:8000/docs
```
Other commands: `make test`, `make eval`, `make demo`, `make lint`.

## Run with Docker (API + PostgreSQL + Prometheus)
```bash
docker compose up --build
```
API: http://localhost:8000/docs · Prometheus: http://localhost:9090

## Try it
```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"amount":2500,"hour":3,"tx_count_24h":12,"merchant_risk":0.9,"distance_km":400}'

curl -X POST localhost:8000/rag/query -H 'content-type: application/json' \
  -d '{"question":"Which algorithm is used for anomaly detection?"}'

curl -X POST localhost:8000/agent/run -H 'content-type: application/json' \
  -d '{"query":"calculate 12*(3+4); what does the critic do; how many events were logged"}'
```
Agent requests are split on `;`, `then`, `and also`; each part is routed to a tool.

## Use a real LLM
```bash
export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-...
```
`ToolRegistry.to_anthropic_tools()` returns schemas ready for the Messages API `tools=` parameter.

## Next steps
Swap `TfidfEmbedder` for a neural embedder, move the vector store to pgvector, add auth and a Grafana dashboard.
