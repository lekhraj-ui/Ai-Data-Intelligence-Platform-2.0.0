import re

from app.agents.orchestrator import Workflow
from app.agents.router import Task, route
from app.agents.tools import ToolRegistry
from app.genai.llm import LLMClient

_SPLIT = re.compile(r"\s*(?:;|\?\s+|\band then\b|\bthen\b|\band also\b)\s*", re.I)


class PlannerAgent:
    def plan(self, query: str) -> list[Task]:
        parts = [p.strip(" .?") for p in _SPLIT.split(query) if p and p.strip(" .?")]
        return [route(p) for p in parts[:4]] or [route(query)]


class ResearcherAgent:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, task: Task) -> dict:
        try:
            return {"task": task.text, "tool": task.tool, "output": self.registry.call(task.tool, **task.args)}
        except Exception as exc:  # noqa: BLE001
            return {"task": task.text, "tool": task.tool, "error": str(exc)}


def _fmt(r: dict) -> str:
    if r.get("error"):
        return f"- {r['task']}: failed ({r['error']})"
    out, tool = r["output"], r["tool"]
    if tool == "search_docs":
        srcs = ", ".join(sorted({s["source"] for s in out["sources"]})) or "no sources"
        return f"- {out['answer']} (sources: {srcs})"
    if tool == "predict_fraud":
        verdict = "FLAGGED as fraud" if out["is_fraud"] else "looks legitimate"
        return (f"- Transaction {verdict}: fraud probability {out['fraud_probability']:.1%}, "
                f"anomaly score {out['anomaly_score']} (anomaly: {out['is_anomaly']})")
    if tool == "calculator":
        return f"- {out['expression']} = {out['value']}"
    if tool == "sql_query":
        return f"- Query returned {len(out['rows'])} row(s): {out['rows']}"
    return f"- {out}"


class WriterAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def compose(self, query: str, results: list[dict]) -> str:
        draft = "\n".join(_fmt(r) for r in results)
        if self.llm.is_mock:
            return draft
        prompt = (f"User request: {query}\n\nFindings from tools:\n{draft}\n\n"
                  "Write a short, clear answer for the user using only these findings.")
        return self.llm.generate(prompt, system="You are a careful technical writer.")


class CriticAgent:
    def review(self, answer: str, results: list[dict]) -> dict:
        issues = []
        if not answer.strip():
            issues.append("empty answer")
        for r in results:
            if r.get("error"):
                issues.append(f"tool '{r['tool']}' failed: {r['error']}")
            elif r["tool"] == "search_docs" and not r["output"]["sources"]:
                issues.append("answer is not grounded in any document")
        return {"approved": not issues, "issues": issues}


class AgentSystem:
    def __init__(self, registry: ToolRegistry, llm: LLMClient):
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent(registry)
        self.writer = WriterAgent(llm)
        self.critic = CriticAgent()

    def run(self, query: str) -> dict:
        wf = (Workflow("agent_run", retries=1)
              .add("plan", lambda c: {"tasks": self.planner.plan(c["query"])})
              .add("research", lambda c: {"results": [self.researcher.execute(t) for t in c["tasks"]]})
              .add("write", lambda c: {"answer": self.writer.compose(c["query"], c["results"])})
              .add("review", lambda c: {"review": self.critic.review(c["answer"], c["results"])}))
        ctx = wf.run({"query": query})
        if ctx.get("error"):
            return {"answer": f"Workflow failed at '{ctx['failed_step']}': {ctx['error']}",
                    "tasks": [], "review": {"approved": False, "issues": [ctx["error"]]},
                    "trace": ctx["trace"]}
        return {"answer": ctx["answer"], "tasks": [t.to_dict() for t in ctx["tasks"]],
                "review": ctx["review"], "trace": ctx["trace"]}
