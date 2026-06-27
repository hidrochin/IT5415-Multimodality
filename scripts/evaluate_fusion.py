"""CLI: the full E3 cross-modal fusion evaluation (T7 — the decisive H3 test).

Fuses the *describe-then-embed* text ranking (BGE-M3 over caption+OCR chunks) with
the *embed-the-image* ranking (CLIP over slide images) and asks H3's question
(PROPOSAL §3.2/§5.3/§4): does direct image embedding add anything over captions,
and if so only under principled (rank/distribution-aware) fusion?

Conditions reported (Recall@K, MRR, nDCG@K with 95% bootstrap CIs):

    text-only      dense BGE-M3 over text+caption chunks   (E2 best — the bar to beat)
    image-only     CLIP embed-the-image                    (the OOD arm on its own)
    RRF            reciprocal rank fusion (text ⊕ image)   (rank-based, calibration-free)
    z-linear @α*   α·z(s_T)+(1−α)·z(s_V), best α by nDCG@5  (distribution-aware)
    fixed 0.7/0.3  raw 0.7·s_T + 0.3·s_V                    (naive baseline to beat)

Plus the α-sensitivity curve for z-linear. No reranker: the bge cross-encoder is
text-only and already shown (T3/T4) to demote caption chunks, so fusion is
evaluated as a clean first-stage combiner.

    python scripts/evaluate_fusion.py --gold data/eval/slides_qa.json

Needs the FAISS chunk store (scripts/ingest_corpus.py) AND the CLIP image index
(scripts/build_image_index.py). Windows note: faiss+torch want the guards
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 (set below as a fallback).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path, forces UTF-8 stdout)
from mmrag.config import load_config
from mmrag.evaluation import (
    aggregate,
    bootstrap_ci,
    collect_fusion_candidates,
    score_candidates,
)
from mmrag.fusion import fuse_hits
from mmrag.image_embeddings import ImageEmbedder, ImageIndex
from mmrag.pipeline import MultimodalRAG
from mmrag.retrieval import FaissIndex

KS = (1, 3, 5)
CI_METRICS = ("recall@1", "recall@5", "mrr", "ndcg@5")
ALPHAS = [round(i / 10, 1) for i in range(11)]  # 0.0 .. 1.0


def _print_table(name: str, metrics: dict, ci: dict | None = None) -> None:
    def cell(key: str, label: str) -> str:
        if ci and key in ci:
            lo, hi = ci[key]
            return f"{label}={metrics[key]:.3f} [{lo:.3f},{hi:.3f}]"
        return f"{label}={metrics[key]:.3f}"

    cols = " | ".join(
        [cell("recall@1", "R@1"), cell("recall@5", "R@5"),
         cell("mrr", "MRR"), cell("ndcg@5", "nDCG@5")]
    )
    print(f"  {name:<16} {cols}")


def main() -> None:
    ap = argparse.ArgumentParser(description="E3 cross-modal fusion eval (H3).")
    ap.add_argument("--gold", default="data/eval/slides_qa.json", help="Gold QA JSON")
    ap.add_argument("--candidate-k", type=int, default=50, help="Per-modality candidate depth before fusion")
    ap.add_argument("--k0", type=int, default=60, help="RRF rank damping constant")
    ap.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for CIs")
    ap.add_argument("--out", default="data/eval/fusion_results.json", help="Where to write per-answer rows")
    args = ap.parse_args()

    cfg = load_config()
    rag = MultimodalRAG(cfg)

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    qa_pairs = gold["qa_pairs"]

    # Text side: the production FAISS index *is* dense BGE-M3 over text + caption
    # chunks (the E2 best / "dense text+cap" reference fusion must beat), so load
    # it from disk rather than re-embedding all ~1200 chunks on CPU.
    if not (cfg.index_dir / "chunks.json").exists():
        raise SystemExit(f"No index at {cfg.index_dir}. Run scripts/ingest_corpus.py first.")
    text_index = FaissIndex.load(cfg.index_dir)
    all_chunks = text_index.chunks

    # Image side: prebuilt CLIP index (embed-the-image arm, T6) + the SAME CLIP
    # encoder for the text query, so query and slide live in one space.
    icfg = cfg.get("image_embeddings", {})
    img_dir = cfg._resolve(icfg.get("index_dir", "data/processed/image_index"))
    image_index = ImageIndex.load(img_dir)
    img_device = icfg.get("device", "auto")
    if not img_device or img_device == "auto":
        img_device = cfg.resolve_device()
    image_embedder = ImageEmbedder(
        model_name=image_index.model_name or icfg.get("model", "clip-ViT-B-32"),
        device=img_device,
        batch_size=int(icfg.get("batch_size", 32)),
    )

    top_k = cfg["retrieval"]["top_k"]
    print(
        f"E3 fusion eval over '{gold['source']}': {len(qa_pairs)} questions\n"
        f"text index: {len(all_chunks)} chunks (BGE-M3) | image index: "
        f"{len(image_index.chunks)} visual chunks ({image_index.model_name}, {image_index.dim}-d)\n"
        f"candidate_k={args.candidate_k}, top_k={top_k}, rerank=off (first-stage fusion), "
        f"n_boot={args.n_boot}\n"
    )

    # Expensive step once: encode queries in both spaces, search both indexes.
    candidates = collect_fusion_candidates(
        qa_pairs, text_index, rag.embedder, image_index, image_embedder,
        candidate_k=args.candidate_k,
    )

    # Single-modality references.
    res = {
        "text-only": score_candidates(candidates, lambda c: c["text_hits"], KS, top_k),
        "image-only": score_candidates(candidates, lambda c: c["image_hits"], KS, top_k),
    }
    # Rank-based fusion.
    res["rrf"] = score_candidates(
        candidates, lambda c: fuse_hits(c["text_hits"], c["image_hits"], "rrf", k0=args.k0), KS, top_k
    )
    # Naive fixed-weight blend (the baseline to beat).
    res["fixed 0.7/0.3"] = score_candidates(
        candidates, lambda c: fuse_hits(c["text_hits"], c["image_hits"], "raw", alpha=0.7), KS, top_k
    )

    # α sensitivity for distribution-aware (z-score) linear fusion.
    curve = []
    for a in ALPHAS:
        r = score_candidates(
            candidates, lambda c, a=a: fuse_hits(c["text_hits"], c["image_hits"], "zscore", alpha=a), KS, top_k
        )
        curve.append((a, r))
    best_alpha, best_res = max(curve, key=lambda ar: ar[1]["metrics"]["ndcg@5"])
    res[f"z-linear a={best_alpha}"] = best_res

    print("=== E3 retrieval metrics, all questions (95% bootstrap CI) ===")
    for name, r in res.items():
        ci = bootstrap_ci(r["per_query"], KS, CI_METRICS, n_boot=args.n_boot)
        _print_table(name, r["metrics"], ci)

    # Figure-grounded subset — where the image arm has any chance of helping.
    fig_ids = {qa["id"] for qa in qa_pairs if qa.get("type") == "figure"}
    if fig_ids:
        print(f"\n=== Figure-grounded questions only ({len(fig_ids)}, RQ3 focus) ===")
        for name, r in res.items():
            sub = [q for q in r["per_query"] if q["id"] in fig_ids]
            ci = bootstrap_ci(sub, KS, CI_METRICS, n_boot=args.n_boot)
            _print_table(name, aggregate(sub, KS), ci)

    print("\n=== α sensitivity (z-linear: α·z(s_T)+(1−α)·z(s_V); α=1 text-only, α=0 image-only) ===")
    for a, r in curve:
        m = r["metrics"]
        star = "  <= best nDCG@5" if a == best_alpha else ""
        print(f"  α={a:.1f}  R@1={m['recall@1']:.3f}  MRR={m['mrr']:.3f}  nDCG@5={m['ndcg@5']:.3f}{star}")

    # H3 verdict, stated against the pre-registered hypothesis.
    t, v = res["text-only"]["metrics"], res["image-only"]["metrics"]
    best_fuse_name = max(
        ("rrf", f"z-linear a={best_alpha}", "fixed 0.7/0.3"),
        key=lambda n: res[n]["metrics"]["ndcg@5"],
    )
    bf = res[best_fuse_name]["metrics"]
    print("\n=== H3 read-out ===")
    print(f"  describe-then-embed (text-only) nDCG@5 {t['ndcg@5']:.3f}  vs  embed-the-image (CLIP) {v['ndcg@5']:.3f}")
    print(f"    -> text {'>=' if t['ndcg@5'] >= v['ndcg@5'] else '<'} image  "
          f"({'consistent with' if t['ndcg@5'] >= v['ndcg@5'] else 'counter to'} H3's direction)")
    print(f"  best fusion: {best_fuse_name} nDCG@5 {bf['ndcg@5']:.3f}  vs text-only {t['ndcg@5']:.3f}  "
          f"(Δ {bf['ndcg@5'] - t['ndcg@5']:+.3f})")
    print(f"  fixed 0.7/0.3 nDCG@5 {res['fixed 0.7/0.3']['metrics']['ndcg@5']:.3f}  "
          f"(principled {'beats' if bf['ndcg@5'] > res['fixed 0.7/0.3']['metrics']['ndcg@5'] else 'ties/loses to'} fixed)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gold": gold["source"],
        "n_questions": len(qa_pairs),
        "candidate_k": args.candidate_k,
        "top_k": top_k,
        "k0": args.k0,
        "best_alpha": best_alpha,
        "metrics": {name: r["metrics"] for name, r in res.items()},
        "alpha_curve": [{"alpha": a, **r["metrics"]} for a, r in curve],
        "per_query": {name: r["per_query"] for name, r in res.items()},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
