"""CLI: evaluate retrieval quality across baselines and conditions.

Answers RQ1/RQ2 from PROPOSAL.md by scoring four first-stage conditions against
the hand-authored gold set:

    BM25 (lexical baseline B0) x {text-only, text+caption}
    dense BGE-M3               x {text-only, text+caption}

Each reports Recall@K, MRR and nDCG@K with bootstrap 95% CIs (T4). Run:

    python scripts/evaluate.py --gold data/eval/slides_qa.json
    python scripts/evaluate.py --gold data/eval/attention_qa.json --no-rerank

Requires an existing index (run scripts/ingest.py first) built with figures on.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from mmrag.config import load_config
from mmrag.evaluation import aggregate, bootstrap_ci, build_index, evaluate
from mmrag.pipeline import MultimodalRAG
from mmrag.retrieval import BM25Index

KS = (1, 3, 5)
# Headline metrics we attach a 95% CI to (keeps the table readable vs CI-on-all).
CI_METRICS = ("recall@1", "recall@5", "mrr", "ndcg@5")


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
    print(f"  {name:<18} {cols}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval: BM25 vs dense, text-only vs +captions.")
    ap.add_argument("--gold", default="data/eval/attention_qa.json", help="Gold QA JSON")
    ap.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder reranker (dense conditions)")
    ap.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for CIs")
    args = ap.parse_args()

    cfg = load_config()
    rag = MultimodalRAG(cfg)

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    qa_pairs = gold["qa_pairs"]

    chunks_path = cfg.index_dir / "chunks.json"
    if not chunks_path.exists():
        raise SystemExit(f"No index at {cfg.index_dir}. Run scripts/ingest.py first.")
    all_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))["chunks"]
    text_chunks = [c for c in all_chunks if not c.get("image_path")]
    figure_chunks = [c for c in all_chunks if c.get("image_path")]

    rcfg = cfg["retrieval"]
    top_k, candidate_k = rcfg["top_k"], rcfg.get("candidate_k", 20)
    use_rerank = (not args.no_rerank) and cfg.get("rerank", {}).get("enabled", False)
    reranker = rag.reranker if use_rerank else None

    print(
        f"Evaluating {len(qa_pairs)} questions over '{gold['source']}' "
        f"({len(text_chunks)} text + {len(figure_chunks)} figure chunks)\n"
        f"top_k={top_k}, candidate_k={candidate_k}, "
        f"dense-rerank={'on' if reranker else 'off'}, n_boot={args.n_boot}\n"
        f"(BM25 baseline is pure lexical — never reranked)\n"
    )

    # (name, chunk set, lexical?, reranker). BM25 is the lexical floor (B0), run
    # without rerank so it stays a clean first-stage baseline; dense conditions
    # take the configured reranker so they mirror the live path.
    conditions = [
        ("bm25 text-only", text_chunks, True, None),
        ("bm25 text+cap", all_chunks, True, None),
        ("dense text-only", text_chunks, False, reranker),
        ("dense text+cap", all_chunks, False, reranker),
    ]
    results: dict[str, dict] = {}
    for name, chunks, lexical, rr in conditions:
        index = BM25Index(chunks) if lexical else build_index(rag.embedder, chunks)
        results[name] = evaluate(
            qa_pairs, index, rag.embedder,
            ks=KS, top_k=top_k, candidate_k=candidate_k, reranker=rr, lexical=lexical,
        )

    print("=== Retrieval metrics, all questions (95% bootstrap CI) ===")
    for name, res in results.items():
        ci = bootstrap_ci(res["per_query"], KS, CI_METRICS, n_boot=args.n_boot)
        _print_table(name, res["metrics"], ci)

    # Figure-grounded subset: where captions are expected to help most (RQ2).
    fig_ids = {qa["id"] for qa in qa_pairs if qa.get("type") == "figure"}
    if fig_ids:
        print(f"\n=== Figure-grounded questions only ({len(fig_ids)}, RQ2 focus) ===")
        for name, res in results.items():
            sub = [q for q in res["per_query"] if q["id"] in fig_ids]
            ci = bootstrap_ci(sub, KS, CI_METRICS, n_boot=args.n_boot)
            _print_table(name, aggregate(sub, KS), ci)

    # Chunk-level signal: does the *relevant* slide/figure chunk actually
    # surface (right source + gold page)? Page-level recall can be carried by a
    # text chunk on the same page, so this isolates whether the caption chunk
    # itself is retrieved — the RQ2 mechanism (source-aware since T3).
    if fig_ids:
        mm_fig = [q for q in results["dense text+cap"]["per_query"] if q["id"] in fig_ids]
        hits = sum(1 for q in mm_fig if q["figure_hit"])
        print(
            f"\n=== Slide-chunk retrieval (dense text+cap) ===\n"
            f"  relevant slide/figure chunk in top-{top_k} for {hits}/{len(mm_fig)} figure-grounded questions"
        )

    # Per-question MRR delta (dense), so regressions are visible, not just averages.
    print("\n=== Per-question MRR (dense text-only -> text+cap) ===")
    base = {q["id"]: q for q in results["dense text-only"]["per_query"]}
    mm = {q["id"]: q for q in results["dense text+cap"]["per_query"]}
    for qa in qa_pairs:
        qid = qa["id"]
        b, m = base[qid]["rr"], mm[qid]["rr"]
        flag = "  <= improved" if m > b + 1e-9 else ("  <= regressed" if m < b - 1e-9 else "")
        print(f"  {qid} [{qa['type']:>6}] {b:.3f} -> {m:.3f}{flag}")


if __name__ == "__main__":
    main()
