RAG_SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know. Be concise."
)


def build_rag_prompt(question: str, hits: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] ({h['source']}) {h['text']}" for i, h in enumerate(hits))
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer using the context and cite [n]."
