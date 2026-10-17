import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)
_WORD = re.compile(r"[a-z0-9_]+")
_STOP = {"the", "a", "an", "is", "are", "of", "what", "how", "which", "does", "do", "for",
         "in", "to", "and", "by", "with", "where", "are", "used", "use"}


def _extractive(question: str, context: list[str]) -> str:
    """Offline stand-in for an LLM: return the context sentences that best match the question."""
    q = set(_WORD.findall(question.lower())) - _STOP
    sentences: list[str] = []
    for c in context:
        sentences += re.split(r"(?<=[.!?])\s+", c.replace("\n", " "))
    ranked = sorted(
        ((len(q & set(_WORD.findall(s.lower()))), i, s) for i, s in enumerate(sentences)),
        key=lambda t: (-t[0], t[1]),
    )
    best = [s for score, _, s in ranked[:2] if score > 0]
    return " ".join(best) or "I could not find an answer in the provided context."


class LLMClient:
    """Provider switch: 'mock' (offline, default) or 'anthropic'."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self._client = None
        if self.provider == "anthropic":
            if not settings.anthropic_api_key:
                logger.warning("ANTHROPIC_API_KEY missing - falling back to mock LLM")
                self.provider = "mock"
            else:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=settings.anthropic_api_key)

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    def generate(self, user: str, system: str = "", context: list[str] | None = None,
                 question: str | None = None, max_tokens: int = 500) -> str:
        if self.is_mock:
            if context:
                return _extractive(question or user, context)
            return f"[mock-llm] {user[:300]}"
        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
