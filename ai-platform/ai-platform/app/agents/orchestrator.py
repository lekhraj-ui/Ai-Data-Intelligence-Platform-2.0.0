import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)


class Workflow:
    """Tiny sequential workflow engine: ordered steps, retries and a per-step trace."""

    def __init__(self, name: str, retries: int = 1):
        self.name, self.retries = name, retries
        self.steps: list[tuple[str, Callable[[dict], dict | None]]] = []

    def add(self, name: str, fn: Callable[[dict], dict | None]) -> "Workflow":
        self.steps.append((name, fn))
        return self

    def run(self, ctx: dict) -> dict:
        trace: list[dict] = []
        for name, fn in self.steps:
            attempt = 0
            while True:
                t0 = time.perf_counter()
                try:
                    ctx.update(fn(ctx) or {})
                    trace.append({"step": name, "ok": True, "attempt": attempt + 1,
                                  "ms": round((time.perf_counter() - t0) * 1000, 2)})
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("step %s failed (attempt %d): %s", name, attempt + 1, exc)
                    trace.append({"step": name, "ok": False, "attempt": attempt + 1,
                                  "error": str(exc),
                                  "ms": round((time.perf_counter() - t0) * 1000, 2)})
                    attempt += 1
                    if attempt > self.retries:
                        ctx.update(failed_step=name, error=str(exc), trace=trace)
                        return ctx
        ctx["trace"] = trace
        return ctx
