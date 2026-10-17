"""Run every layer without starting the server:  python -m scripts.demo"""
import json

from app.agents.multi_agent import AgentSystem
from app.agents.tools import build_default_registry
from app.db import init_db
from app.genai.llm import LLMClient
from app.genai.rag import RAGPipeline
from app.ml.service import ModelService

init_db()
llm = LLMClient()
models = ModelService().load()
rag = RAGPipeline(llm=llm)
rag.reindex()
agents = AgentSystem(build_default_registry(rag, models), llm)

print("== ML ==")
print(models.predict({"amount": 2500, "hour": 3, "tx_count_24h": 12,
                      "merchant_risk": 0.9, "distance_km": 400}))
print("== RAG ==")
print(rag.query("How are documents chunked?")["answer"])
print("== Agents ==")
out = agents.run("calculate 12*(3+4); is amount=2500 hour=3 merchant_risk=0.9 fraud; "
                 "what does the critic do")
print(out["answer"])
print(json.dumps(out["trace"], indent=1))
