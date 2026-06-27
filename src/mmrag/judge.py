"""QA-accuracy via LLM-as-judge + human-agreement validation (PROPOSAL §6.2, task T8).

Retrieval eval (`evaluation.py`) asks whether the *evidence* is found; faithfulness
(`faithfulness.py`) asks whether the answer *stays inside* it. Neither asks the
question a reader actually cares about: **is the answer correct?** That needs to
compare a generated answer to a known-good reference, which is a judgement call —
so we use an LLM judge (Gemini) and, crucially, **validate the judge** against the
human author before trusting it (§6.2). A judge we have not checked is just another
unvalidated model.

This module is the pure half (no torch, no API), so the grading logic — prompt
construction, verdict parsing, accuracy aggregation, and Cohen's κ — is
unit-checkable offline. ``scripts/qa_accuracy.py`` supplies the live answers and
judge calls.

Design choices that keep the judge honest about its known biases (§6.2):
- **Reference-based, not reference-free.** The judge sees a hand-authored gold
  answer and grades coverage/correctness against it, rather than re-deriving truth
  from the question alone (which would just re-test the same model's knowledge).
- **Style is explicitly out of scope.** The prompt tells the judge to ignore
  length, phrasing, and citations and to *not* reward verbosity — the standard
  length/verbosity bias mitigation. There is no self-preference confound to anonymize
  because a single fixed reference is graded, not two candidates ranked against
  each other.
- **Three ordered labels** (correct / partial / incorrect) rather than a raw 1–10
  score: coarse labels are far more reproducible across judge runs and far easier
  for the human validator to apply consistently, which is what κ measures.
"""
from __future__ import annotations

import re
from typing import Sequence

# Ordered worst→best so a graded score is just the index / (k-1).
LABELS = ("incorrect", "partial", "correct")
_LABEL_SCORE = {"incorrect": 0.0, "partial": 0.5, "correct": 1.0}

JUDGE_SYSTEM = (
    "You are a strict grader for a question-answering system about multimodal "
    "machine learning. You are given a QUESTION, a REFERENCE answer that is known "
    "to be correct, and a CANDIDATE answer produced by the system. Decide how well "
    "the CANDIDATE matches the REFERENCE on factual content.\n"
    "Grade ONLY factual correctness and coverage of the key points in the reference. "
    "IGNORE writing style, length, wording, ordering, and any [pN] page citations. "
    "Do NOT reward an answer for being longer or more detailed; a concise answer that "
    "covers the key points is fully correct.\n"
    "Use exactly one of these labels:\n"
    "  correct   - captures all the key facts of the reference, with no factual error "
    "(paraphrase and partial extra detail are fine).\n"
    "  partial   - on topic and partly right, but misses a key point or contains a "
    "minor factual error.\n"
    "  incorrect - wrong, contradicts the reference, off-topic, or the candidate "
    "abstains / says it cannot answer.\n"
    "Respond with a single line in exactly this form:\n"
    "VERDICT: <correct|partial|incorrect> | REASON: <short justification>"
)

_VERDICT_RE = re.compile(r"verdict\s*[:\-]?\s*(correct|partial|incorrect)", re.IGNORECASE)
# Fallback: the first standalone label word anywhere in the reply.
_ANY_LABEL_RE = re.compile(r"\b(correct|partial|incorrect)\b", re.IGNORECASE)


def build_judge_prompt(question: str, reference: str, candidate: str) -> str:
    """Render the user turn for the judge: question + gold reference + candidate."""
    return (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE answer (correct):\n{reference}\n\n"
        f"CANDIDATE answer (to grade):\n{candidate}\n\n"
        "Grade the CANDIDATE against the REFERENCE."
    )


def parse_verdict(reply: str) -> str | None:
    """Extract the verdict label from a judge reply, or ``None`` if unparseable.

    Prefers the canonical ``VERDICT: <label>`` form; falls back to the first
    standalone label word so a judge that drops the prefix is still scored rather
    than silently discarded.
    """
    if not reply:
        return None
    m = _VERDICT_RE.search(reply)
    if m:
        return m.group(1).lower()
    m = _ANY_LABEL_RE.search(reply)
    return m.group(1).lower() if m else None


def label_to_score(label: str | None) -> float | None:
    """Map an ordered label to a graded score in [0,1] (None passes through)."""
    return _LABEL_SCORE.get(label) if label is not None else None


def aggregate_accuracy(rows: Sequence[dict]) -> dict:
    """QA-accuracy summary over per-answer judge rows (each with a ``label``).

    Reports three views so the headline does not hide the partials:
    - ``strict_accuracy``  — fraction graded exactly ``correct``.
    - ``lenient_accuracy`` — fraction ``correct`` OR ``partial`` (on-topic & mostly right).
    - ``graded_score``     — mean of the ordered score (incorrect 0 / partial .5 / correct 1).
    Unparseable verdicts are counted (``n_unparsed``) and excluded from the rates
    rather than guessed at.
    """
    scored = [r for r in rows if r.get("label") in _LABEL_SCORE]
    n = len(scored)
    if n == 0:
        return {
            "n": 0, "n_unparsed": len(rows),
            "strict_accuracy": 0.0, "lenient_accuracy": 0.0, "graded_score": 0.0,
            "counts": {lab: 0 for lab in LABELS},
        }
    counts = {lab: sum(1 for r in scored if r["label"] == lab) for lab in LABELS}
    return {
        "n": n,
        "n_unparsed": len(rows) - n,
        "strict_accuracy": counts["correct"] / n,
        "lenient_accuracy": (counts["correct"] + counts["partial"]) / n,
        "graded_score": sum(_LABEL_SCORE[r["label"]] for r in scored) / n,
        "counts": counts,
    }


def cohens_kappa(a: Sequence, b: Sequence, categories: Sequence | None = None) -> float | None:
    """Cohen's κ between two label sequences (chance-corrected agreement).

    κ = (p_o − p_e) / (1 − p_e), where p_o is the observed agreement and p_e the
    agreement expected if each rater labelled independently at their own marginal
    rates. κ=1 is perfect agreement, 0 is chance-level, <0 is worse than chance.
    Returns ``None`` for an empty input. The degenerate case where p_e=1 (both
    raters used a single category for everything) is reported as κ=1.0 if they
    fully agree, else 0.0 — there is no chance-corrected signal to extract.
    """
    if len(a) != len(b):
        raise ValueError("label sequences must be the same length")
    n = len(a)
    if n == 0:
        return None
    cats = list(categories) if categories is not None else sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum(
        (sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n)
        for c in cats
    )
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def to_binary(label: str | None) -> str | None:
    """Collapse the 3-way label to correct-vs-not for a binary κ / accuracy view."""
    if label is None:
        return None
    return "correct" if label == "correct" else "not_correct"


def bootstrap_accuracy_ci(
    rows: Sequence[dict],
    metric: str = "graded_score",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a QA-accuracy metric (same rationale as T4).

    Resamples the per-answer rows with replacement so the small-n point estimate
    carries an honest uncertainty band. ``metric`` is any key produced by
    :func:`aggregate_accuracy` (e.g. ``graded_score``, ``strict_accuracy``).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(rows)
    if n == 0:
        return (0.0, 0.0)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = [rows[i] for i in idx]
        vals.append(aggregate_accuracy(boot)[metric])
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return (float(np.percentile(vals, lo_q)), float(np.percentile(vals, hi_q)))
