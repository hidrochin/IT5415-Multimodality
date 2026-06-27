"""Hybrid retrieval (spec module 6.6) — Phase 1: dense FAISS over text.

Phase 2 will add image embeddings and score fusion
(``0.7 * text_score + 0.3 * image_score``) plus the bge-reranker.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens — the shared tokenizer for BM25."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi-BM25 lexical baseline (eval B0) over the same chunk store.

    The dense retrievers are the system under test; BM25 is the term-matching
    floor they must clear to justify the embedding cost (PROPOSAL §6.1). It mirrors
    ``FaissIndex``'s ``.search`` contract — returns chunk dicts with a ``score`` —
    but is searched by the **raw query string** (no embedder), so ``evaluate`` can
    treat it uniformly via its ``lexical=True`` path.
    """

    def __init__(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        corpus = [_tokenize(c["text"]) for c in chunks]
        # rank_bm25 can't index an empty corpus; guard so an all-empty chunk set
        # (shouldn't happen post-ingest) degrades to "no hits" instead of raising.
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top_k = min(top_k, len(self.chunks))
        top_idx = np.argsort(scores)[::-1][:top_k]
        results: list[dict] = []
        for idx in top_idx:
            hit = dict(self.chunks[int(idx)])
            hit["score"] = float(scores[idx])
            results.append(hit)
        return results


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
