import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app import monitoring as mon
from app.agents.multi_agent import AgentSystem
from app.agents.tools import build_default_registry
from app.config import settings
from app.db import init_db, log_event, recent_events
from app.genai.llm import LLMClient
from app.genai.rag import RAGPipeline
from app.ml.service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    llm = LLMClient()
    models = ModelService().load()
    rag = RAGPipeline(llm=llm)
    rag.reindex()
    app.state.models, app.state.rag = models, rag
    app.state.agents = AgentSystem(build_default_registry(rag, models), llm)
    yield


app = FastAPI(title="AI Platform", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start, status = time.perf_counter(), 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        mon.REQUESTS.labels(request.method, path, str(status)).inc()
        mon.LATENCY.labels(request.method, path).observe(time.perf_counter() - start)


class Transaction(BaseModel):
    amount: float = Field(gt=0, examples=[950.0])
    hour: int = Field(ge=0, le=23, examples=[3])
    tx_count_24h: int = Field(ge=0, examples=[9])
    merchant_risk: float = Field(ge=0, le=1, examples=[0.8])
    distance_km: float = Field(ge=0, examples=[120.0])


class Question(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=3, ge=1, le=10)


class Ingest(BaseModel):
    filename: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=10)


class AgentRequest(BaseModel):
    query: str = Field(min_length=3)


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": settings.llm_provider}


@app.get("/metrics")
def metrics():
    body, content_type = mon.render_metrics()
    return Response(body, media_type=content_type)


@app.get("/model/info")
def model_info(request: Request):
    return request.app.state.models.metrics


@app.post("/predict")
def predict(tx: Transaction, request: Request):
    t0 = time.perf_counter()
    result = request.app.state.models.predict(tx.model_dump())
    mon.PREDICTIONS.labels("fraud" if result["is_fraud"] else "legit").inc()
    log_event("predict", tx.model_dump(), result, (time.perf_counter() - t0) * 1000)
    return result


@app.post("/rag/query")
def rag_query(q: Question, request: Request):
    t0 = time.perf_counter()
    result = request.app.state.rag.query(q.question, q.k)
    mon.RAG_QUERIES.labels("yes" if result["sources"] else "no").inc()
    log_event("rag", q.model_dump(), {"answer": result["answer"]}, (time.perf_counter() - t0) * 1000)
    return result


@app.post("/rag/ingest")
def rag_ingest(doc: Ingest, request: Request):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", doc.filename)
    if not name.endswith((".md", ".txt")):
        name += ".md"
    target = settings.docs_dir / "uploads"
    target.mkdir(parents=True, exist_ok=True)
    (target / name).write_text(doc.content, encoding="utf-8")
    n = request.app.state.rag.reindex()
    return {"saved": name, "total_chunks": n}


@app.post("/agent/run")
def agent_run(req: AgentRequest, request: Request):
    t0 = time.perf_counter()
    try:
        result = request.app.state.agents.run(req.query)
    except Exception as exc:  # noqa: BLE001
        mon.AGENT_RUNS.labels("error").inc()
        raise HTTPException(500, f"agent failure: {exc}") from exc
    mon.AGENT_RUNS.labels("approved" if result["review"]["approved"] else "rejected").inc()
    log_event("agent", {"query": req.query}, {"answer": result["answer"]},
              (time.perf_counter() - t0) * 1000)
    return result


@app.get("/events")
def events(limit: int = 20):
    return recent_events(min(max(limit, 1), 100))
