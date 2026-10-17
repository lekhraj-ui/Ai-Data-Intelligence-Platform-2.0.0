import re
from pathlib import Path

from app.config import settings
from app.genai.llm import LLMClient
from app.genai.prompts import RAG_SYSTEM, build_rag_prompt
from app.genai.vectorstore import VectorStore


def chunk_text(text: str, size: int = 600) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    cleaned = []
    for p in paras:
        body = "\n".join(l for l in p.splitlines() if not l.lstrip().startswith("#")).strip()
        if body:
            cleaned.append(body)
    chunks, cur = [], ""
    for p in cleaned:
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


class RAGPipeline:
    def __init__(self, docs_dir: Path | None = None, llm: LLMClient | None = None):
        self.docs_dir = Path(docs_dir or settings.docs_dir)
        self.llm = llm or LLMClient()
        self.store = VectorStore()

    def reindex(self) -> int:
        chunks = []
        for path in sorted(self.docs_dir.rglob("*")):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            for i, text in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
                chunks.append({"source": path.name, "chunk_id": i, "text": text})
        self.store.build(chunks)
        return len(chunks)

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        return self.store.search(question, k)

    def query(self, question: str, k: int = 3) -> dict:
        hits = self.retrieve(question, k)
        if not hits or hits[0]["score"] < settings.rag_min_score:
            return {"answer": "I don't know based on the indexed documents.", "sources": []}
        answer = self.llm.generate(
            user=build_rag_prompt(question, hits), system=RAG_SYSTEM,
            context=[h["text"] for h in hits], question=question,
        )
        sources = [{"source": h["source"], "score": round(h["score"], 3),
                    "snippet": h["text"][:200]} for h in hits]
        return {"answer": answer, "sources": sources}
