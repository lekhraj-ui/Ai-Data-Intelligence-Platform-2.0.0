import re
from dataclasses import asdict, dataclass, field

from app.ml.features import FEATURES

_DEFAULTS = {"amount": 50.0, "hour": 12, "tx_count_24h": 2, "merchant_risk": 0.1, "distance_km": 5.0}
_MATH = re.compile(r"\d\s*[\+\*/\^%-]\s*\(?\s*\d")
_MATH_SPAN = re.compile(r"[\d\.\s\+\-\*/\(\)\^%]{3,}")


@dataclass
class Task:
    text: str
    tool: str
    args: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def route(text: str) -> Task:
    """Rule-based agent routing: pick the tool (and its arguments) for one sub-task."""
    low = text.lower()
    kv = {k: float(v) for k, v in re.findall(r"(\w+)\s*=\s*(-?[\d.]+)", text) if k in FEATURES}
    if kv:
        args = {**_DEFAULTS, **kv}
        args["hour"], args["tx_count_24h"] = int(args["hour"]), int(args["tx_count_24h"])
        return Task(text, "predict_fraud", args, "transaction features detected")
    if _MATH.search(text):
        expr = max(_MATH_SPAN.findall(text), key=len).strip()
        return Task(text, "calculator", {"expression": expr}, "arithmetic expression detected")
    if re.search(r"\b(how many|count|number of|recent|latest|last)\b.*\b(events?|logs?|predictions?|requests?)\b", low):
        if re.search(r"\b(recent|latest|last)\b", low):
            q = "SELECT id, kind, created_at FROM event_logs ORDER BY id DESC LIMIT 5"
        else:
            q = "SELECT kind, COUNT(*) AS n FROM event_logs GROUP BY kind"
        return Task(text, "sql_query", {"query": q}, "question about stored events")
    if re.search(r"\b(time|date)\b.*\b(now|today|current)\b|\bcurrent (time|date)\b", low):
        return Task(text, "utc_now", {}, "time question")
    return Task(text, "search_docs", {"question": text}, "default: knowledge-base question")
