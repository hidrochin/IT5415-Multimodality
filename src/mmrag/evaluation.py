"""Retrieval evaluation (spec section 9): Recall@K and MRR.

These metrics let us answer the research questions empirically by comparing a
text-only index against a text+figure-caption index over a hand-authored gold
set (RQ1/RQ2). A retrieved chunk is "relevant" when it matches the question's
evidence. For a single-document index that key is just the page; for the
**multi-deck** slide index pages collide across decks, so the key is the
``(source, page)`` pair (T3). The matching functions below are generic over the
key type — pass plain pages or ``(source, page)`` tuples.
"""
from __future__ import annotations

from typing import Hashable, Sequence


def recall_at_k(ranked: Sequence[Hashable], gold: Sequence[Hashable], k: int) -> float:
    """1.0 if any of the top-k retrieved keys is a gold evidence key, else 0.0."""
    gold_set = set(gold)
    return 1.0 if any(key in gold_set for key in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[Hashable], gold: Sequence[Hashable]) -> float:
    """1 / rank of the first relevant key in the ranking (0.0 if none)."""
    gold_set = set(gold)
    for rank, key in enumerate(ranked, start=1):
        if key in gold_set:
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
        gold_pages = qa["evidence_pages"]
        # Source-aware scoring (T3): on the multi-deck slide index a page number
        # alone is ambiguous (every deck has a "page 7"), so when the question
        # names a `source` we score on (source, page) keys. Falls back to plain
        # pages for single-doc gold sets (e.g. attention_qa.json) that omit it.
        gold_source = qa.get("source")

        q_emb = embedder.encode_queries([question])
        if reranker is not None:
            hits = index.search(q_emb, top_k=max(candidate_k, top_k))
            hits = reranker.rerank(question, hits, top_k=top_k)
        else:
            hits = index.search(q_emb, top_k=top_k)

        ranked_pages = [h["page"] for h in hits]
        if gold_source is not None:
            ranked_keys = [(h.get("source"), h["page"]) for h in hits]
            gold_keys = [(gold_source, p) for p in gold_pages]
        else:
            ranked_keys, gold_keys = ranked_pages, gold_pages
        gold_key_set = set(gold_keys)

        row = {
            "id": qa.get("id"),
            "type": qa.get("type"),
            "source": gold_source,
            "gold": gold_pages,
            "ranked_pages": ranked_pages,
            "ranked_ids": [h.get("id", "") for h in hits],
            # Did the *relevant* figure/slide chunk (right source + gold page)
            # actually surface? The mechanism check behind RQ2 — captions only
            # help if the visual chunk for the evidence page is retrieved.
            "figure_hit": any(
                h.get("image_path") and (h.get("source"), h["page"]) in gold_key_set
                for h in hits
            ),
            "rr": reciprocal_rank(ranked_keys, gold_keys),
        }
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked_keys, gold_keys, k)
        per_query.append(row)

    return {"per_query": per_query, "metrics": aggregate(per_query, ks)}
