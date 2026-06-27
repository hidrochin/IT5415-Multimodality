"""Cross-space late fusion of two retrieval rankings (spec module / PROPOSAL §3.2,
§5.3 — the H3 combiners).

The text side (`s_T`, BGE-M3 over caption+OCR chunks — *describe-then-embed*) and
the image side (`s_V`, CLIP over slide images — *embed-the-image*) come from
**different encoders with different score distributions**, so their raw scores are
not directly additive. This module implements the principled combiners H3 puts on
trial, plus the naive baseline it must beat:

- :func:`rrf` — Reciprocal Rank Fusion `Σ_m 1/(k_0 + rank_m(c))`. Rank-based, so it
  needs no score calibration at all; absence from a list contributes nothing.
- :func:`linear_fuse` — distribution-aware linear `α·z(s_T) + (1−α)·z(s_V)`, with
  per-query standardization `z` (``zscore`` or ``minmax``). ``method="raw"`` skips
  standardization, which (with ``alpha=0.7``) reproduces the original spec's fixed
  `0.7·s_T + 0.3·s_V` blend — the **naive baseline to be beaten**, not the method.

All functions are **pure** (no torch/numpy-model deps): they take ranked hit
dicts / score maps and return fused scores, so they are cheap to sweep over α and
trivially unit-testable. :func:`fuse_hits` is the convenience wrapper the eval
script calls — it maps the fused scores back onto the original hit dicts.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable


def rrf(rankings: dict[str, list[Hashable]], k0: int = 60) -> dict[Hashable, float]:
    """Reciprocal Rank Fusion over named ranked id-lists.

    ``rankings`` maps a modality name to its ranked list of chunk ids (best first).
    Each list contributes ``1/(k0 + rank)`` (1-indexed) to every id it contains; an
    id absent from a list simply gets no term from it. ``k0`` damps the influence of
    top ranks (Cormack et al. 2009 use 60).
    """
    fused: dict[Hashable, float] = defaultdict(float)
    for ids in rankings.values():
        for rank, cid in enumerate(ids, start=1):
            fused[cid] += 1.0 / (k0 + rank)
    return dict(fused)


def _standardize(scores: dict[Hashable, float], method: str) -> dict[Hashable, float]:
    """Per-list score standardization. ``raw`` is identity (no calibration)."""
    if not scores or method == "raw":
        return dict(scores)
    vals = list(scores.values())
    if method == "zscore":
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sd = var ** 0.5
        if sd == 0:
            return {k: 0.0 for k in scores}
        return {k: (v - mu) / sd for k, v in scores.items()}
    if method == "minmax":
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 0.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
    raise ValueError(f"unknown standardization method: {method!r}")


def linear_fuse(
    text_scores: dict[Hashable, float],
    image_scores: dict[Hashable, float],
    alpha: float,
    method: str = "zscore",
) -> dict[Hashable, float]:
    """Distribution-aware linear fusion ``α·z(s_T) + (1−α)·z(s_V)``.

    Each modality's scores are standardized independently (so the strong text
    cosines and the weak, tightly-clustered CLIP cosines are put on one scale),
    then blended. ``alpha=1`` → text only, ``alpha=0`` → image only. A chunk present
    in only one modality (the common case: the image index covers only visual
    chunks) is filled in the *other* modality with that modality's **minimum**
    standardized score, so a one-sided hit is neither rewarded nor unduly punished
    by the absent side. ``method="raw"`` skips standardization and fills with 0.0,
    reproducing the fixed `0.7·s_T + 0.3·s_V` baseline.
    """
    t = _standardize(text_scores, method)
    v = _standardize(image_scores, method)
    fill_t = 0.0 if method == "raw" else (min(t.values()) if t else 0.0)
    fill_v = 0.0 if method == "raw" else (min(v.values()) if v else 0.0)
    ids = set(t) | set(v)
    return {
        cid: alpha * t.get(cid, fill_t) + (1.0 - alpha) * v.get(cid, fill_v)
        for cid in ids
    }


def fuse_hits(
    text_hits: list[dict],
    image_hits: list[dict],
    method: str,
    *,
    alpha: float = 0.5,
    k0: int = 60,
) -> list[dict]:
    """Fuse two hit-dict lists into one ranked list of hit dicts.

    ``method`` is one of ``"rrf"`` | ``"zscore"`` | ``"minmax"`` | ``"raw"`` (the
    last is the fixed blend, used with ``alpha=0.7``). Hits are keyed on ``id``; the
    returned dicts are copies of the originals (text fields win when a chunk appears
    in both lists) carrying an added ``fused_score``. Ties break on the id string for
    deterministic ordering.
    """
    id2hit: dict[Hashable, dict] = {}
    for h in image_hits:
        id2hit[h["id"]] = h
    for h in text_hits:  # text wins on shared metadata (caption text, etc.)
        id2hit[h["id"]] = h

    if method == "rrf":
        scores = rrf(
            {
                "text": [h["id"] for h in text_hits],
                "image": [h["id"] for h in image_hits],
            },
            k0=k0,
        )
    else:
        scores = linear_fuse(
            {h["id"]: h["score"] for h in text_hits},
            {h["id"]: h["score"] for h in image_hits},
            alpha=alpha,
            method=method,
        )

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    out: list[dict] = []
    for cid, s in ranked:
        hit = dict(id2hit[cid])
        hit["fused_score"] = s
        out.append(hit)
    return out
