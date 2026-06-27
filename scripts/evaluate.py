"""CLI: evaluate retrieval quality, text-only vs text+figure-caption.

Answers RQ1/RQ2 from PROPOSAL.md by re-embedding the ingested chunks two ways
and scoring both against the hand-authored gold set:

    python scripts/evaluate.py
    python scripts/evaluate.py --gold data/eval/attention_qa.json --no-rerank

Requires an existing index (run scripts/ingest.py first) built with figures on.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from mmrag.config import load_config
from mmrag.evaluation import build_index, evaluate
from mmrag.pipeline import MultimodalRAG

KS = (1, 3, 5)


def _print_table(name: str, metrics: dict) -> None:
    cols = " | ".join(f"R@{k}={metrics[f'recall@{k}']:.3f}" for k in KS)
    print(f"  {name:<14} {cols} | MRR={metrics['mrr']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval: text-only vs +captions.")
    ap.add_argument("--gold", default="data/eval/attention_qa.json", help="Gold QA JSON")
    ap.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder reranker")
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
        f"top_k={top_k}, candidate_k={candidate_k}, rerank={'on' if reranker else 'off'}\n"
    )

    configs = {
        "text-only": text_chunks,
        "text+caption": all_chunks,
    }
    results: dict[str, dict] = {}
    for name, chunks in configs.items():
        index = build_index(rag.embedder, chunks)
        results[name] = evaluate(
            qa_pairs, index, rag.embedder,
            ks=KS, top_k=top_k, candidate_k=candidate_k, reranker=reranker,
        )

    print("=== Retrieval metrics (higher is better) ===")
    for name, res in results.items():
        _print_table(name, res["metrics"])

    # Highlight the figure-grounded subset, where captions are expected to help most.
    fig_ids = {qa["id"] for qa in qa_pairs if qa.get("type") == "figure"}
    if fig_ids:
        print("\n=== Figure-grounded questions only (RQ2 focus) ===")
        from mmrag.evaluation import aggregate
        for name, res in results.items():
            sub = [q for q in res["per_query"] if q["id"] in fig_ids]
            _print_table(name, aggregate(sub, KS))

    # Chunk-level signal: does the *relevant* slide/figure chunk actually
    # surface (right source + gold page)? Page-level recall can be carried by a
    # text chunk on the same page, so this isolates whether the caption chunk
    # itself is retrieved — the RQ2 mechanism (source-aware since T3).
    if fig_ids:
        mm_fig = [q for q in results["text+caption"]["per_query"] if q["id"] in fig_ids]
        hits = sum(1 for q in mm_fig if q["figure_hit"])
        print(
            f"\n=== Slide-chunk retrieval (text+caption) ===\n"
            f"  relevant slide/figure chunk in top-{top_k} for {hits}/{len(mm_fig)} figure-grounded questions"
        )

    # Per-question MRR delta, so regressions are visible, not just averages.
    print("\n=== Per-question MRR (text-only -> text+caption) ===")
    base = {q["id"]: q for q in results["text-only"]["per_query"]}
    mm = {q["id"]: q for q in results["text+caption"]["per_query"]}
    for qa in qa_pairs:
        qid = qa["id"]
        b, m = base[qid]["rr"], mm[qid]["rr"]
        flag = "  <= improved" if m > b + 1e-9 else ("  <= regressed" if m < b - 1e-9 else "")
        print(f"  {qid} [{qa['type']:>6}] {b:.3f} -> {m:.3f}{flag}")


if __name__ == "__main__":
    main()
