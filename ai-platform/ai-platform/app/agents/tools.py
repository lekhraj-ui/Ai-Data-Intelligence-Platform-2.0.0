import ast
import operator as op
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import text

from app.db import engine
from app.genai.rag import RAGPipeline
from app.ml.service import ModelService


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    input_schema: dict


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def call(self, name: str, **kwargs):
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name].fn(**kwargs)

    def names(self) -> list[str]:
        return list(self._tools)

    def to_anthropic_tools(self) -> list[dict]:
        """Schemas ready to pass as `tools=` to the Anthropic Messages API."""
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in self._tools.values()]


_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    raise ValueError("unsupported expression")


def calculator(expression: str) -> dict:
    value = _eval(ast.parse(expression.replace("^", "**").strip(), mode="eval"))
    return {"expression": expression.strip(), "value": value}


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|attach)\b", re.I)


def sql_query(query: str) -> dict:
    q = query.strip().rstrip(";")
    if not q.lower().startswith("select") or ";" in q or _FORBIDDEN.search(q):
        raise ValueError("only a single read-only SELECT is allowed")
    with engine.connect() as conn:
        result = conn.execute(text(q))
        rows = [dict(r._mapping) for r in result.fetchmany(50)]
    return {"query": q, "rows": rows}


def build_default_registry(rag: RAGPipeline, models: ModelService) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        "search_docs", "Answer a question from the indexed knowledge base (RAG).",
        lambda question, k=3: rag.query(question, k),
        {"type": "object", "properties": {"question": {"type": "string"}, "k": {"type": "integer"}},
         "required": ["question"]}))
    reg.register(Tool(
        "predict_fraud", "Score a card transaction for fraud probability and anomaly.",
        lambda **kw: models.predict(kw),
        {"type": "object", "properties": {
            "amount": {"type": "number"}, "hour": {"type": "integer"},
            "tx_count_24h": {"type": "integer"}, "merchant_risk": {"type": "number"},
            "distance_km": {"type": "number"}},
         "required": ["amount", "hour", "tx_count_24h", "merchant_risk", "distance_km"]}))
    reg.register(Tool(
        "calculator", "Evaluate an arithmetic expression.", calculator,
        {"type": "object", "properties": {"expression": {"type": "string"}},
         "required": ["expression"]}))
    reg.register(Tool(
        "sql_query", "Run a read-only SELECT against the event_logs table.", sql_query,
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}))
    reg.register(Tool(
        "utc_now", "Current UTC time.", lambda: {"utc": datetime.now(timezone.utc).isoformat()},
        {"type": "object", "properties": {}}))
    return reg
