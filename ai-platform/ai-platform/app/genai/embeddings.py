from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Offline embeddings (L2-normalised TF-IDF). Swap for a neural/API embedder in production."""

    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")

    def fit(self, texts: list[str]) -> None:
        self.vec.fit(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray()
