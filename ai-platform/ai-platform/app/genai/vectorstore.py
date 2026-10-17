import numpy as np

from app.genai.embeddings import Embedder, TfidfEmbedder


class VectorStore:
    """In-memory cosine-similarity store. Replace with pgvector / a vector DB at scale."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or TfidfEmbedder()
        self.chunks: list[dict] = []
        self.matrix = np.zeros((0, 0))

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        if not chunks:
            self.matrix = np.zeros((0, 0))
            return
        texts = [c["text"] for c in chunks]
        self.embedder.fit(texts)
        self.matrix = self.embedder.embed(texts)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q  # vectors are L2-normalised -> dot product == cosine
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]
