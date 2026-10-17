# Agentic Workflow

The planner splits a request into sub-tasks and the router assigns each sub-task to a tool: search_docs, predict_fraud, calculator or sql_query.

Tool calling is implemented with a registry that exposes JSON schemas, so the same tools can be handed to an LLM tool-use API.

The multi-agent system has four roles: planner, researcher, writer and critic. The researcher executes tools, the writer composes the final answer and the critic approves or rejects it.

Workflow orchestration runs the steps sequentially with retries and records a trace with duration and status for every step.
