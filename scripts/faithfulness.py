"""CLI: faithfulness / citation eval over the gold set (PROPOSAL §6.3, task T5).

Generates a grounded Gemini answer for every gold question through the **live
retrieval path** (dense FAISS -> reranker -> top_k, mirroring `ask.py`), then
scores each answer's ``[pN]`` citations against the evidence it was actually
shown:

    citation precision  -- cited pages that are in the evidence
    leakage rate        -- answers citing a page never shown (the no-leakage check)
    citation recall     -- of retrievable gold pages, the fraction cited (proxy)

Run (needs GEMINI_API_KEY in .env, and an index from scripts/ingest_corpus.py):

    python scripts/faithfulness.py --gold data/eval/slides_qa.json
    python scripts/faithfulness.py --gold data/eval/slides_qa.json --no-rerank

Results (per-answer rows + aggregates) are written to
``data/eval/faithfulness_results.json`` for inspection.
"""
import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from mmrag.config import load_config
from mmrag.faithfulness import aggregate, bootstrap_leakage_ci, score_answer
from mmrag.pipeline import MultimodalRAG
from mmrag.retrieval import FaissIndex


def _answer_with_retry(qa, question, chunks, retries: int = 8):
    """Gemini answer with capped exponential backoff.

    The shared flash-lite tier 503s/429s heavily under load, so a few quick
    retries aren't enough — we back off up to ``retries`` times (2,4,8,…,capped
    at 30 s) before giving up, mirroring the captioner's resilience in
    ``pipeline.py``.
    """
    delay = 2.0
    for attempt in range(retries):
        try:
            return qa.answer(question, chunks)
        except Exception as e:  # noqa: BLE001 - retry any transient API failure
            if attempt == retries - 1:
                raise
            msg = str(e).split("{")[0].strip() or type(e).__name__
            print(f"    (retry {attempt + 1}/{retries - 1} after: {msg})")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Faithfulness / citation eval over the gold set.")
    ap.add_argument("--gold", default="data/eval/slides_qa.json", help="Gold QA JSON")
    ap.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder reranker")
    ap.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for CIs")
    ap.add_argument(
        "--out", default="data/eval/faithfulness_results.json", help="Where to write per-answer rows"
    )
    args = ap.parse_args()

    cfg = load_config()
    api_key = cfg.gemini_api_key
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")

    rag = MultimodalRAG(cfg)

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    qa_pairs = gold["qa_pairs"]

    if not (cfg.index_dir / "chunks.json").exists():
        raise SystemExit(f"No index at {cfg.index_dir}. Run scripts/ingest_corpus.py first.")
    index = FaissIndex.load(cfg.index_dir)  # load once, reuse for every question

    rcfg = cfg["retrieval"]
    top_k, candidate_k = rcfg["top_k"], rcfg.get("candidate_k", 20)
    use_rerank = (not args.no_rerank) and cfg.get("rerank", {}).get("enabled", False)

    from mmrag.qa import GeminiQA

    qcfg = cfg["qa"]
    qa = GeminiQA(
        api_key=api_key,
        model=qcfg["model"],
        temperature=qcfg.get("temperature", 0.2),
        max_output_tokens=qcfg.get("max_output_tokens", 1024),
    )

    print(
        f"Faithfulness eval: {len(qa_pairs)} questions over '{gold['source']}'\n"
        f"top_k={top_k}, candidate_k={candidate_k}, "
        f"rerank={'on' if use_rerank else 'off'}, qa_model={qcfg['model']}\n"
    )

    rows: list[dict] = []
    for qa_pair in qa_pairs:
        question = qa_pair["question"]
        # Mirror the live retrieval path (rag.retrieve) but reuse the loaded index.
        q_emb = rag.embedder.encode_queries([question])
        fetch_k = max(candidate_k, top_k) if use_rerank else top_k
        hits = index.search(q_emb, top_k=fetch_k)
        if use_rerank:
            hits = rag.reranker.rerank(question, hits, top_k=top_k)
        else:
            hits = hits[:top_k]

        answer = _answer_with_retry(qa, question, hits)
        row = score_answer(
            answer, hits, qa_pair["evidence_pages"], qa_pair.get("source")
        )
        row.update(
            {"id": qa_pair.get("id"), "type": qa_pair.get("type"),
             "source": qa_pair.get("source"), "answer": answer}
        )
        rows.append(row)

        leak = " LEAK" if row["has_leak"] else ""
        prec = "n/a" if row["precision"] is None else f"{row['precision']:.2f}"
        print(
            f"  {row['id']} [{row['type']:>6}] cited={row['cited_pages']} "
            f"prec={prec} leaked={row['leaked_pages']}{leak}"
        )

    metrics = aggregate(rows)
    ci = bootstrap_leakage_ci(rows, n_boot=args.n_boot)

    def band(key: str) -> str:
        lo, hi = ci[key]
        return f"[{lo:.3f},{hi:.3f}]"

    print("\n=== Faithfulness, all questions (95% bootstrap CI) ===")
    print(f"  answers={metrics['n']}  citing={metrics['n_citing']}  abstained={metrics['n_abstained']}")
    print(f"  total citations={metrics['total_citations']}  leaked={metrics['total_leaked']}")
    print(f"  citation precision (macro) = {metrics['citation_precision']:.3f} {band('citation_precision')}")
    print(f"  citation precision (micro) = {metrics['micro_precision']:.3f}")
    print(f"  citation recall (macro,proxy) = {metrics['citation_recall']:.3f} {band('citation_recall')}")
    print(f"  LEAKAGE RATE (answer-level)   = {metrics['leakage_rate']:.3f} {band('leakage_rate')}")
    print(f"  leakage (micro, per-citation) = {metrics['micro_leakage']:.3f}")
    print(f"  abstention rate = {metrics['abstention_rate']:.3f}")

    # Figure-grounded subset (RQ2 focus): faithfulness on visually-grounded answers.
    fig_rows = [r for r in rows if r["type"] == "figure"]
    if fig_rows:
        fm = aggregate(fig_rows)
        print(f"\n=== Figure-grounded subset ({fm['n']}) ===")
        print(f"  citation precision (macro) = {fm['citation_precision']:.3f}")
        print(f"  leakage rate (answer-level) = {fm['leakage_rate']:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"gold": gold["source"], "rerank": use_rerank, "metrics": metrics,
             "ci": {k: list(v) for k, v in ci.items()}, "rows": rows},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote per-answer rows + metrics to {out_path}")


if __name__ == "__main__":
    main()
