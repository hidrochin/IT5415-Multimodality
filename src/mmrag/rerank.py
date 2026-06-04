"""Reranking (spec module 6.7).

A cross-encoder (default bge-reranker-base) re-scores the candidate passages
pulled from FAISS. Cross-encoders read the (query, passage) pair jointly, so
they are far more precise than the bi-encoder used for first-stage retrieval —
at the cost of running one forward pass per candidate.
"""
from __future__ import annotations


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        # Local import keeps `import mmrag` cheap.
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.batch_size = batch_size

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        """Return chunks sorted by cross-encoder relevance, with `rerank_score` set."""
        if not chunks:
            return []
        pairs = [(query, c.get("text", "")) for c in chunks]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        ranked = sorted(zip(chunks, scores), key=lambda cs: float(cs[1]), reverse=True)
        out: list[dict] = []
        for chunk, score in ranked:
            hit = dict(chunk)
            hit["rerank_score"] = float(score)
            out.append(hit)
        return out[:top_k] if top_k is not None else out
