"""Hybrid retrieval (spec module 6.6) — Phase 1: dense FAISS over text.

Phase 2 will add image embeddings and score fusion
(``0.7 * text_score + 0.3 * image_score``) plus the bge-reranker.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.json"


class FaissIndex:
    """A flat inner-product FAISS index paired with its chunk metadata."""

    def __init__(self, dim: int) -> None:
        import faiss  # local import: faiss optional until used

        self._faiss = faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[dict] = []

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must be the same length")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        scores, idxs = self.index.search(query_emb, top_k)
        results: list[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            hit = dict(self.chunks[idx])
            hit["score"] = float(score)
            results.append(hit)
        return results

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(index_dir / _INDEX_FILE))
        with open(index_dir / _CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "chunks": self.chunks}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str | Path) -> "FaissIndex":
        import faiss

        index_dir = Path(index_dir)
        chunks_path = index_dir / _CHUNKS_FILE
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"No index found in {index_dir}. Run ingest first (scripts/ingest.py)."
            )
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(data["dim"])
        obj.index = faiss.read_index(str(index_dir / _INDEX_FILE))
        obj.chunks = data["chunks"]
        return obj
