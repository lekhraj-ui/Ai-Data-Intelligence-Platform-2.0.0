from app.evaluation.run import eval_rag
from app.genai.llm import LLMClient
from app.genai.rag import RAGPipeline, chunk_text


def test_chunking_respects_size():
    text = "\n\n".join(["word " * 60] * 5)
    assert all(len(c) <= 700 for c in chunk_text(text, 600))


def test_retrieval_quality():
    rag = RAGPipeline(llm=LLMClient(provider="mock"))
    assert rag.reindex() > 0
    assert eval_rag(rag)["hit_at_3"] >= 0.75


def test_unknown_question():
    rag = RAGPipeline(llm=LLMClient(provider="mock"))
    rag.reindex()
    assert rag.query("zzzz qqqq xxxx")["sources"] == []
