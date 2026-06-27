"""Retrieval evaluation (spec section 9): Recall@K, MRR, nDCG@K.

These metrics let us answer the research questions empirically by comparing a
text-only index against a text+figure-caption index over a hand-authored gold
set (RQ1/RQ2). A retrieved chunk is "relevant" when it matches the question's
evidence. For a single-document index that key is just the page; for the
**multi-deck** slide index pages collide across decks, so the key is the
``(source, page)`` pair (T3). The matching functions below are generic over the
key type — pass plain pages or ``(source, page)`` tuples.

T4 adds the rigor the proposal's evaluation protocol asks for (PROPOSAL §6.1/§6.4):
a **BM25 lexical baseline** (``BM25Index``, B0), **nDCG@K** (rank-aware, graded),
and **bootstrap confidence intervals** so the small-n slide results carry an
uncertainty band rather than a bare point estimate.
"""
from __future__ import annotations

import math
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


def ndcg_at_k(ranked: Sequence[Hashable], gold: Sequence[Hashable], k: int) -> float:
    """Normalized DCG@k with binary relevance over evidence keys.

    Unlike Recall@k (did *any* relevant key appear) and MRR (where is the
    *first*), nDCG rewards ranking *every* relevant evidence key high — the right
    lens when a question cites several pages. Relevance is binary (a key is gold
    or not), and each gold key is credited **once**: the multi-deck index can
    surface the same (source, page) via both a text chunk and a slide chunk, and
    double-crediting it would push DCG above the ideal. IDCG is the best
    achievable ordering given ``min(#gold, k)`` relevant keys.
    """
    gold_set = set(gold)
    seen: set = set()
    dcg = 0.0
    for i, key in enumerate(ranked[:k], start=1):
        if key in gold_set and key not in seen:
            seen.add(key)
            dcg += 1.0 / math.log2(i + 1)
    n_rel = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(per_query: list[dict], ks: Sequence[int]) -> dict[str, float]:
    """Mean Recall@k, nDCG@k (for each k) and MRR across all queries."""
    n = len(per_query) or 1
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = sum(q[f"recall@{k}"] for q in per_query) / n
        out[f"ndcg@{k}"] = sum(q[f"ndcg@{k}"] for q in per_query) / n
    out["mrr"] = sum(q["rr"] for q in per_query) / n
    return out


def bootstrap_ci(
    per_query: list[dict],
    ks: Sequence[int],
    metrics: Sequence[str],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CI for each metric by resampling *questions*.

    With only ~33 gold questions a point estimate is noisy, so we resample the
    per-query rows with replacement ``n_boot`` times, re-aggregate, and take the
    central ``1 - alpha`` percentile band. Resampling at the question level is the
    right unit — queries are the independent observations, chunks are not.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(per_query)
    if n == 0:
        return {m: (0.0, 0.0) for m in metrics}
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = [per_query[i] for i in idx]
        agg = aggregate(boot, ks)
        for m in metrics:
            samples[m].append(agg[m])
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        m: (float(np.percentile(s, lo_q)), float(np.percentile(s, hi_q)))
        for m, s in samples.items()
    }


def score_query(qa: dict, hits: list[dict], ks: Sequence[int]) -> dict:
    """Build the per-query metric row for one already-ranked, already-truncated
    ``hits`` list (the top-k results, however they were produced).

    Centralises the **source-aware** relevance logic (T3) so every retrieval path
    — single-index dense/lexical (:func:`evaluate`) and the T7 fusion combiners
    (:func:`score_candidates`) — scores identically. When the gold question names a
    ``source`` the key is the ``(source, page)`` pair (page numbers collide across
    decks); otherwise it falls back to the bare page for single-doc gold sets.
    """
    gold_pages = qa["evidence_pages"]
    gold_source = qa.get("source")
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
        # Did the *relevant* figure/slide chunk (right source + gold page) actually
        # surface? The mechanism check behind RQ2/RQ3 — figures only help if the
        # visual chunk for the evidence page is retrieved.
        "figure_hit": any(
            h.get("image_path") and (h.get("source"), h["page"]) in gold_key_set
            for h in hits
        ),
        "rr": reciprocal_rank(ranked_keys, gold_keys),
    }
    for k in ks:
        row[f"recall@{k}"] = recall_at_k(ranked_keys, gold_keys, k)
        row[f"ndcg@{k}"] = ndcg_at_k(ranked_keys, gold_keys, k)
    return row


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
    lexical: bool = False,
) -> dict:
    """Run every gold question through retrieval and score it.

    Returns ``{"per_query": [...], "metrics": {...}}``. When ``reranker`` is
    given, the top ``candidate_k`` first-stage hits are re-scored before
    truncating to ``top_k`` — matching the live retrieval path so metrics reflect
    production. ``lexical=True`` selects a term-matching first stage (``BM25Index``,
    searched by the raw question string) instead of the dense embedder; the
    ``embedder`` is then unused for first-stage retrieval.
    """
    per_query: list[dict] = []
    for qa in qa_pairs:
        question = qa["question"]
        fetch_k = max(candidate_k, top_k) if reranker is not None else top_k
        if lexical:
            hits = index.search(question, top_k=fetch_k)
        else:
            q_emb = embedder.encode_queries([question])
            hits = index.search(q_emb, top_k=fetch_k)
        if reranker is not None:
            hits = reranker.rerank(question, hits, top_k=top_k)
        else:
            hits = hits[:top_k]
        per_query.append(score_query(qa, hits, ks))

    return {"per_query": per_query, "metrics": aggregate(per_query, ks)}


# ─────────────────────────── Cross-modal fusion eval (T7 / E3, H3) ──────────────────────────
# The decisive H3 test (PROPOSAL §3.2/§5.3): fuse the *describe-then-embed* text
# ranking (BGE-M3 over caption+OCR chunks) with the *embed-the-image* ranking
# (CLIP over slide images). Because re-encoding the CLIP query and re-searching
# both indexes is the expensive part, candidates are collected **once** and then
# any number of fusion combiners / α values are applied as pure re-scoring over
# the cached candidate lists (:mod:`mmrag.fusion`).

def collect_fusion_candidates(
    qa_pairs: list[dict],
    text_index,
    text_embedder,
    image_index,
    image_embedder,
    candidate_k: int = 50,
) -> list[dict]:
    """Retrieve the deep text and image candidate lists for every gold question.

    Returns one row per question: ``{"qa", "text_hits", "image_hits"}`` where each
    hit list is the top ``candidate_k`` from that modality (dicts carrying ``id``,
    ``score``, ``page``, ``source``, ``image_path``). Encoders are queried here and
    only here; downstream fusion is score arithmetic over these lists.
    """
    rows: list[dict] = []
    for qa in qa_pairs:
        q = qa["question"]
        text_hits = text_index.search(text_embedder.encode_queries([q]), top_k=candidate_k)
        image_hits = image_index.search(image_embedder.encode_queries([q])[0], top_k=candidate_k)
        rows.append({"qa": qa, "text_hits": text_hits, "image_hits": image_hits})
    return rows


def score_candidates(
    candidates: list[dict],
    rank_fn,
    ks: Sequence[int] = (1, 3, 5),
    top_k: int = 5,
) -> dict:
    """Score a set of collected candidates under one ranking strategy.

    ``rank_fn`` maps a candidate row to a full ranked list of hit dicts (e.g. the
    text-only list, the image-only list, or a fused list from
    :func:`mmrag.fusion.fuse_hits`). It is truncated to ``top_k`` and scored with
    the shared source-aware :func:`score_query`, so fusion conditions are directly
    comparable to the single-index conditions from :func:`evaluate`.
    """
    per_query = [score_query(c["qa"], rank_fn(c)[:top_k], ks) for c in candidates]
    return {"per_query": per_query, "metrics": aggregate(per_query, ks)}
