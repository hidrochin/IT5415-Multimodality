"""Text embeddings (spec module 6.5).

Wraps sentence-transformers. Embeddings are L2-normalized so that FAISS inner
product == cosine similarity. Query/passage prefixes support models like E5.
"""
from __future__ import annotations

import numpy as np


class TextEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        # Local import keeps `import mmrag` cheap and torch optional until used.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    @property
    def dim(self) -> int:
        # Method was renamed in sentence-transformers 5.x; fall back for older.
        fn = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(fn())

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return emb.astype("float32")
