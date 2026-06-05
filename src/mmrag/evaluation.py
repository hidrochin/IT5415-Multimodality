"""Retrieval evaluation (spec section 9): Recall@K and MRR.

These metrics let us answer the research questions empirically by comparing a
text-only index against a text+figure-caption index over a hand-authored gold
set (RQ1/RQ2). A retrieved chunk is "relevant" when its page is one of the
question's ``evidence_pages``.
"""
from __future__ import annotations

from typing import Sequence


def recall_at_k(ranked_pages: Sequence[int], gold_pages: Sequence[int], k: int) -> float:
    """1.0 if any of the top-k retrieved pages is a gold evidence page, else 0.0."""
    gold = set(gold_pages)
    return 1.0 if any(p in gold for p in ranked_pages[:k]) else 0.0


def reciprocal_rank(ranked_pages: Sequence[int], gold_pages: Sequence[int]) -> float:
    """1 / rank of the first relevant page in the ranking (0.0 if none)."""
    gold = set(gold_pages)
    for rank, page in enumerate(ranked_pages, start=1):
        if page in gold:
            return 1.0 / rank
    return 0.0


def aggregate(per_query: list[dict], ks: Sequence[int]) -> dict[str, float]:
    """Mean Recall@k (for each k) and MRR across all queries."""
    n = len(per_query) or 1
    out: dict[str, float] = {
        f"recall@{k}": sum(q[f"recall@{k}"] for q in per_query) / n for k in ks
    }
    out["mrr"] = sum(q["rr"] for q in per_query) / n
    return out


def build_index(embedder, chunks: list[dict]):
    """Embed chunk texts and build an in-memory FAISS index over them.

    Re-embedding from the stored chunks (rather than re-ingesting) lets us swap
    chunk sets — e.g. drop figure chunks for the text-only baseline — without
    re-running OCR or figure captioning.
    """
    from .retrieval import FaissIndex

    embeddings = embedder.encode_passages([c["text"] for c in chunks])
    index = FaissIndex(embedder.dim)
    index.add(embeddings, chunks)
    return index


def evaluate(
    qa_pairs: list[dict],
    index,
    embedder,
    ks: Sequence[int] = (1, 3, 5),
    top_k: int = 5,
    candidate_k: int = 20,
    reranker=None,
) -> dict:
    """Run every gold question through retrieval and score it.

    Returns ``{"per_query": [...], "metrics": {...}}``. When ``reranker`` is
    given, the top ``candidate_k`` FAISS hits are re-scored before truncating to
    ``top_k`` — matching the live retrieval path so metrics reflect production.
    """
    per_query: list[dict] = []
    for qa in qa_pairs:
        question = qa["question"]
        gold = qa["evidence_pages"]

        q_emb = embedder.encode_queries([question])
        if reranker is not None:
            hits = index.search(q_emb, top_k=max(candidate_k, top_k))
            hits = reranker.rerank(question, hits, top_k=top_k)
        else:
            hits = index.search(q_emb, top_k=top_k)

        ranked_pages = [h["page"] for h in hits]
        row = {
            "id": qa.get("id"),
            "type": qa.get("type"),
            "gold": gold,
            "ranked_pages": ranked_pages,
            "ranked_ids": [h.get("id", "") for h in hits],
            # Did an actual figure chunk (OCR/caption) surface in the results?
            "figure_hit": any(h.get("image_path") for h in hits),
            "rr": reciprocal_rank(ranked_pages, gold),
        }
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked_pages, gold, k)
        per_query.append(row)

    return {"per_query": per_query, "metrics": aggregate(per_query, ks)}
