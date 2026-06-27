"""Faithfulness / citation metrics (spec section 6.3).

Retrieval recall measures whether the *evidence* is found; faithfulness measures
whether the *answer* stays inside it. The grounded prompt (``qa.py``) instructs
Gemini to cite only page numbers that appear in the evidence headers, and never a
page it was not shown. PROPOSAL §1.4/§6.3 makes this a property to be **measured,
not assumed** — a retrieval win that does not translate into faithful answers is
of limited value. This module turns the previously anecdotal "no-leakage"
observation into numbers:

- **Citation precision** = fraction of an answer's cited pages that are actually
  present in the retrieved evidence. The direct test of the no-leakage property.
- **Leakage** = the complement: a citation to a page never shown to the model.
  Reported both micro (leaked citations / all citations) and answer-level
  (fraction of answers carrying *any* leak) — the headline number.
- **Citation recall (page-level proxy)** = of the gold evidence pages that were
  *retrievable* (their (source, page) chunk actually surfaced in the evidence),
  the fraction the answer cited. The §6.3 definition is claim-level ("evidence-
  supported claims actually cited"); enumerating claims needs an LLM judge (T8),
  so here we use a defensible page-level proxy: did the model cite the supporting
  page it was handed? It is bounded above by retrieval — you cannot cite a page
  that was never retrieved — which is exactly why it is scored over the
  *retrievable* gold subset, not all gold.

Everything here is pure/string-only (no torch, no API), so it is unit-checkable
offline; ``scripts/faithfulness.py`` supplies the live answers and evidence.
"""
from __future__ import annotations

import re
from typing import Sequence

# The prompt asks for ``[pN]`` citations; tolerate ``[p7, p9]``, ``[7]``, ``[pp. 7]``.
# Two branches, scanned in order so multi-citation answers keep their sequence:
#   (1) a ``p``/``pp`` prefixed page number anywhere (``\b`` so "step7"/"top 7"
#       don't match) — the canonical form, and
#   (2) a bracket containing *only* digits and separators (``[7]``, ``[7, 8]``).
# Crucially this does **not** treat a bracket with letters as a citation, so an
# evidence-index reference like ``[Evidence 2]`` is ignored rather than miscounted
# as a citation to page 2. (Genuine ``[p2]`` markers the model emits — e.g. when it
# conflates the evidence index with the page — are still caught: that is real
# leakage, not a parser artifact.)
_CITATION = re.compile(
    r"\bp{1,2}\.?\s*(\d+)"               # group 1: p7, pp7, p.7, p 7
    r"|\[\s*(\d[\d\s,;:\.\-]*?)\s*\]",   # group 2: [7], [7, 8] — digits only
    re.IGNORECASE,
)
_DIGITS = re.compile(r"\d+")


def parse_citations(answer: str) -> list[int]:
    """Extract cited page numbers from an answer's ``[pN]`` markers (in order).

    Duplicates are kept (a page cited twice counts twice for micro precision);
    callers that want the distinct set can ``set(...)`` the result.
    """
    pages: list[int] = []
    for m in _CITATION.finditer(answer):
        if m.group(1) is not None:
            pages.append(int(m.group(1)))
        else:
            pages.extend(int(n) for n in _DIGITS.findall(m.group(2)))
    return pages


def score_answer(
    answer: str,
    evidence_chunks: Sequence[dict],
    gold_pages: Sequence[int],
    gold_source: str | None = None,
) -> dict:
    """Score one grounded answer for citation faithfulness.

    ``evidence_chunks`` are the retrieved chunks shown to the model (each a dict
    with ``page`` and ``source``). Precision/leakage compare cited pages against
    the page numbers present in that evidence; citations are page-only, so the
    evidence page set spans all sources (a cited ``[p7]`` is "valid" if *any*
    shown chunk is on page 7 — matching what the model could see in the headers).

    The recall proxy is **source-aware**: the retrievable gold subset is the gold
    pages whose ``(gold_source, page)`` chunk actually surfaced, so a same-numbered
    page from the wrong deck does not count as the model having seen the evidence.
    """
    cited = parse_citations(answer)
    cited_set = set(cited)
    evidence_pages = {c.get("page") for c in evidence_chunks}
    gold_set = set(gold_pages)

    valid = [p for p in cited if p in evidence_pages]
    leaked = sorted(cited_set - evidence_pages)

    # Source-aware retrievable-gold subset for the recall proxy.
    evidence_keys = {(c.get("source"), c.get("page")) for c in evidence_chunks}
    retrievable_gold = {p for p in gold_set if (gold_source, p) in evidence_keys}
    cited_gold = retrievable_gold & cited_set

    n_cited = len(cited)
    precision = len(valid) / n_cited if n_cited else None
    recall = (
        len(cited_gold) / len(retrievable_gold) if retrievable_gold else None
    )

    return {
        "cited_pages": cited,
        "leaked_pages": leaked,
        "evidence_pages": sorted(p for p in evidence_pages if p is not None),
        "retrievable_gold": sorted(retrievable_gold),
        "n_cited": n_cited,
        "n_valid": len(valid),
        "n_leaked": len(cited_set - evidence_pages),
        "n_retrievable_gold": len(retrievable_gold),
        "abstained": n_cited == 0,
        "has_leak": bool(leaked),
        "precision": precision,
        "recall": recall,
    }


def aggregate(rows: Sequence[dict]) -> dict:
    """Corpus-level faithfulness metrics over per-answer rows from ``score_answer``.

    - ``citation_precision`` / ``citation_recall`` are **macro** (mean over the
      answers where the metric is defined: precision over answers that cited,
      recall over answers with a retrievable gold page).
    - ``micro_precision`` pools all citations (total valid / total cited), so a
      verbose multi-citation answer weighs more than a terse one.
    - ``leakage_rate`` is the headline no-leakage check: fraction of *citing*
      answers that contain at least one leaked citation. ``micro_leakage`` is its
      citation-pooled complement (1 - micro_precision).
    - ``abstention_rate`` flags answers with no citations at all (e.g. the model
      declared the evidence insufficient) — excluded from precision so an
      abstention is neither rewarded nor punished as a leak.
    """
    n = len(rows)
    with_cit = [r for r in rows if r["n_cited"] > 0]
    with_gold = [r for r in rows if r["n_retrievable_gold"] > 0]
    total_cited = sum(r["n_cited"] for r in rows)
    total_valid = sum(r["n_valid"] for r in rows)

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    micro_precision = total_valid / total_cited if total_cited else 0.0
    return {
        "n": n,
        "n_citing": len(with_cit),
        "n_abstained": n - len(with_cit),
        "total_citations": total_cited,
        "total_leaked": total_cited - total_valid,
        "citation_precision": _mean([r["precision"] for r in with_cit]),
        "citation_recall": _mean([r["recall"] for r in with_gold]),
        "micro_precision": micro_precision,
        "micro_leakage": 1.0 - micro_precision if total_cited else 0.0,
        "leakage_rate": _mean([1.0 if r["has_leak"] else 0.0 for r in with_cit]),
        "abstention_rate": _mean([1.0 if r["abstained"] else 0.0 for r in rows]),
    }


def bootstrap_leakage_ci(
    rows: Sequence[dict],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for the headline faithfulness numbers.

    Same rationale as ``evaluation.bootstrap_ci`` (T4): with a hand-authored gold
    set n is small, so we resample *answers* with replacement and report a 95%
    band rather than a bare point estimate. Bands this wide on a leakage rate of
    zero are expected and honest — they say "no leak observed in n answers", not
    "leakage is impossible".
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(rows)
    metrics = ("citation_precision", "leakage_rate", "citation_recall")
    if n == 0:
        return {m: (0.0, 0.0) for m in metrics}
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = [rows[i] for i in idx]
        agg = aggregate(boot)
        for m in metrics:
            samples[m].append(agg[m])
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        m: (float(np.percentile(s, lo_q)), float(np.percentile(s, hi_q)))
        for m, s in samples.items()
    }
