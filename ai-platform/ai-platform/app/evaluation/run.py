"""Evaluation gate: python -m app.evaluation.run  (exit code 1 if quality drops - used in CI)."""
import json
import sys
import tempfile
from pathlib import Path

from app.config import settings
from app.genai.llm import LLMClient
from app.genai.rag import RAGPipeline
from app.ml.service import ModelService

MIN_ROC_AUC = 0.90
MIN_RAG_HIT = 0.75


def eval_rag(rag: RAGPipeline, k: int = 3) -> dict:
    cases = json.loads(Path(settings.eval_file).read_text())
    hits = kw = 0
    for c in cases:
        retrieved = rag.retrieve(c["question"], k)
        hits += any(h["source"] == c["expected_source"] for h in retrieved)
        blob = " ".join(h["text"] for h in retrieved).lower()
        kw += c["keyword"].lower() in blob
    n = len(cases)
    return {"cases": n, f"hit_at_{k}": hits / n, "keyword_recall": kw / n}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        models = ModelService(models_dir=Path(tmp)).load()  # fresh train => evaluates current code
    ml = models.metrics
    rag = RAGPipeline(llm=LLMClient(provider="mock"))
    rag.reindex()
    rg = eval_rag(rag)
    print("ML :", {k: round(v, 3) for k, v in ml.items() if isinstance(v, float)})
    print("RAG:", rg)
    ok = ml["roc_auc"] >= MIN_ROC_AUC and rg["hit_at_3"] >= MIN_RAG_HIT
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
