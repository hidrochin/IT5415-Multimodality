"""CLI: QA-accuracy eval via LLM-as-judge, with human-agreement validation (T8).

Three things happen here, mirroring PROPOSAL §6.2:

1. **Answer.** For each gold question, generate a grounded answer through the live
   retrieval path (dense FAISS -> reranker -> top_k -> Gemini), under each retrieval
   **condition** — ``text-only`` vs ``text+caption`` — so we can see whether the
   caption-augmented retrieval that won on Recall/nDCG (H1/H2) also produces *more
   correct answers*, which is what a reader ultimately cares about.
2. **Judge.** Grade every answer against a hand-authored gold reference with an LLM
   judge (``mmrag.judge``), one tier *up* from the answer model (gemini-2.5-flash
   judging gemini-2.5-flash-lite) so the judge is not grading itself.
3. **Validate.** Emit a random-subset worksheet for the human author to label; once
   filled, ``--kappa <worksheet>`` reports judge–human Cohen's κ. Per §6.2 the judge
   is only trusted where κ is adequate.

The two retrieval conditions reuse the on-disk FAISS index: the text-only index is
built by **reconstructing** the stored text-chunk vectors (no re-embedding), so the
only cost is the Gemini answer + judge calls.

Run (needs GEMINI_API_KEY and an index from scripts/ingest_corpus.py):

    python scripts/qa_accuracy.py                       # both conditions, judge, write worksheet
    python scripts/qa_accuracy.py --no-rerank
    python scripts/qa_accuracy.py --kappa data/eval/qa_accuracy_worksheet.json   # κ after labelling

Windows: prefix with KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 (faiss+torch guard).
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
import numpy as np

from mmrag.config import load_config
from mmrag.judge import (
    JUDGE_SYSTEM, LABELS, aggregate_accuracy, bootstrap_accuracy_ci,
    build_judge_prompt, cohens_kappa, parse_verdict, to_binary,
)
from mmrag.pipeline import MultimodalRAG
from mmrag.retrieval import FaissIndex


def _with_retry(fn, what: str, retries: int = 8):
    """Call ``fn`` with capped exponential backoff (flash-lite 503/429s under load)."""
    delay = 2.0
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - retry any transient API failure
            if attempt == retries - 1:
                raise
            msg = str(e).split("{")[0].strip() or type(e).__name__
            print(f"    (retry {attempt + 1}/{retries - 1} {what} after: {msg})")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return None


def _build_text_only_index(full: FaissIndex) -> FaissIndex:
    """Text-only index from the stored text-chunk vectors — no re-embedding.

    The on-disk index holds every chunk's BGE-M3 vector in chunk order; we
    reconstruct them and keep only the non-figure (no ``image_path``) chunks, so the
    text-only retrieval condition is exact, not a re-encode (which would be ~1h on
    CPU for ~1k chunks).
    """
    n = full.index.ntotal
    vecs = full.index.reconstruct_n(0, n)
    keep = [i for i, c in enumerate(full.chunks) if not c.get("image_path")]
    text_only = FaissIndex(full.dim)
    text_only.add(np.ascontiguousarray(vecs[keep]), [full.chunks[i] for i in keep])
    return text_only


def _make_judge(api_key: str, model: str, temperature: float = 0.0):
    """A minimal Gemini judge call: (question, reference, candidate) -> reply text."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        system_instruction=JUDGE_SYSTEM, temperature=temperature, max_output_tokens=256
    )

    def judge(question: str, reference: str, candidate: str) -> str:
        prompt = build_judge_prompt(question, reference, candidate)
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        return (resp.text or "").strip()

    return judge


def _compute_kappa(out_path: Path) -> None:
    """Report judge–human Cohen's κ from a filled worksheet, then exit."""
    data = json.loads(out_path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    paired = [it for it in items if it.get("human_label") in LABELS and it.get("judge_label") in LABELS]
    if not paired:
        raise SystemExit(
            f"No labelled rows in {out_path}: fill each item's \"human_label\" with one of "
            f"{LABELS} and re-run with --kappa."
        )
    judge = [it["judge_label"] for it in paired]
    human = [it["human_label"] for it in paired]
    k3 = cohens_kappa(judge, human, categories=list(LABELS))
    k2 = cohens_kappa([to_binary(x) for x in judge], [to_binary(x) for x in human])
    agree3 = sum(1 for a, b in zip(judge, human) if a == b) / len(paired)
    print(f"\n=== Judge–human agreement on {len(paired)} labelled answers ===")
    print(f"  raw agreement (3-way) = {agree3:.3f}")
    print(f"  Cohen's kappa (3-way: {'/'.join(LABELS)}) = {k3:.3f}")
    print(f"  Cohen's kappa (binary correct vs not)      = {k2:.3f}")
    verdict = (
        "substantial+ (>0.6): trust the judge" if k3 is not None and k3 >= 0.6
        else "moderate (0.4-0.6): judge usable with caution" if k3 is not None and k3 >= 0.4
        else "weak (<0.4): do NOT trust the automatic judge"
    )
    print(f"  -> {verdict}")
    data["kappa"] = {"three_way": k3, "binary": k2, "raw_agreement": agree3, "n": len(paired)}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote κ back to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="QA-accuracy eval (LLM-judge + human-κ).")
    ap.add_argument("--gold", default="data/eval/slides_qa.json")
    ap.add_argument("--gold-answers", default="data/eval/slides_qa_gold_answers.json")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--judge-model", default="gemini-2.5-flash",
                    help="Judge model (default one tier above the answer model)")
    ap.add_argument("--subset-n", type=int, default=12, help="Human-label worksheet size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="data/eval/qa_accuracy_results.json")
    ap.add_argument("--worksheet", default="data/eval/qa_accuracy_worksheet.json")
    ap.add_argument("--kappa", default=None,
                    help="Path to a filled worksheet: compute judge–human κ and exit")
    args = ap.parse_args()

    if args.kappa:
        _compute_kappa(Path(args.kappa))
        return

    cfg = load_config()
    api_key = cfg.gemini_api_key
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    qa_pairs = gold["qa_pairs"]
    references = json.loads(Path(args.gold_answers).read_text(encoding="utf-8"))["answers"]
    missing = [qa["id"] for qa in qa_pairs if qa["id"] not in references]
    if missing:
        raise SystemExit(f"Gold answers missing for ids: {missing}")

    if not (cfg.index_dir / "chunks.json").exists():
        raise SystemExit(f"No index at {cfg.index_dir}. Run scripts/ingest_corpus.py first.")

    rag = MultimodalRAG(cfg)
    full = FaissIndex.load(cfg.index_dir)
    text_only = _build_text_only_index(full)

    rcfg = cfg["retrieval"]
    top_k, candidate_k = rcfg["top_k"], rcfg.get("candidate_k", 20)
    use_rerank = (not args.no_rerank) and cfg.get("rerank", {}).get("enabled", False)

    qcfg = cfg["qa"]
    from mmrag.qa import GeminiQA

    answerer = GeminiQA(
        api_key=api_key, model=qcfg["model"],
        temperature=qcfg.get("temperature", 0.2),
        max_output_tokens=qcfg.get("max_output_tokens", 1024),
    )
    judge = _make_judge(api_key, args.judge_model)

    def retrieve(index: FaissIndex, question: str) -> list[dict]:
        q_emb = rag.embedder.encode_queries([question])
        fetch_k = max(candidate_k, top_k) if use_rerank else top_k
        hits = index.search(q_emb, top_k=fetch_k)
        return rag.reranker.rerank(question, hits, top_k=top_k) if use_rerank else hits[:top_k]

    conditions = {"text-only": text_only, "text+caption": full}
    print(
        f"QA-accuracy: {len(qa_pairs)} questions x {len(conditions)} conditions over "
        f"'{gold['source']}'\ntop_k={top_k}, candidate_k={candidate_k}, "
        f"rerank={'on' if use_rerank else 'off'}, answer={qcfg['model']}, "
        f"judge={args.judge_model}\n"
    )

    results: dict[str, dict] = {}
    for cond_name, index in conditions.items():
        print(f"--- condition: {cond_name} ---")
        rows: list[dict] = []
        for qa in qa_pairs:
            q, qid = qa["question"], qa["id"]
            ref = references[qid]
            hits = retrieve(index, q)
            answer = _with_retry(lambda: answerer.answer(q, hits), f"answer {qid}")
            reply = _with_retry(lambda: judge(q, ref, answer), f"judge {qid}")
            label = parse_verdict(reply)
            rows.append({
                "id": qid, "type": qa.get("type"), "source": qa.get("source"),
                "question": q, "reference": ref, "answer": answer,
                "judge_label": label, "judge_raw": reply,
            })
            print(f"  {qid} [{qa.get('type','?'):>6}] -> {label or 'UNPARSED'}")
        results[cond_name] = rows

    # ── Aggregates ───────────────────────────────────────────────────────────
    def label_rows(rows):
        return [{"label": r["judge_label"]} for r in rows]

    summary: dict[str, dict] = {}
    print("\n=== QA-accuracy per condition (95% bootstrap CI on graded score) ===")
    for cond_name, rows in results.items():
        agg = aggregate_accuracy(label_rows(rows))
        lo, hi = bootstrap_accuracy_ci(label_rows(rows), "graded_score", n_boot=args.n_boot)
        agg["graded_ci"] = [lo, hi]
        summary[cond_name] = agg
        c = agg["counts"]
        tail = f"/unparsed={agg['n_unparsed']}" if agg["n_unparsed"] else ""
        print(
            f"  {cond_name:<13} graded={agg['graded_score']:.3f} [{lo:.3f},{hi:.3f}]  "
            f"strict={agg['strict_accuracy']:.3f}  lenient={agg['lenient_accuracy']:.3f}  "
            f"(correct={c['correct']}/partial={c['partial']}/incorrect={c['incorrect']}{tail})"
        )

    # Figure-grounded subset (RQ2 focus): is visual grounding answered correctly?
    print("\n=== Figure-grounded subset ===")
    fig_summary: dict[str, dict] = {}
    for cond_name, rows in results.items():
        frows = [r for r in rows if r["type"] == "figure"]
        agg = aggregate_accuracy([{"label": r["judge_label"]} for r in frows])
        fig_summary[cond_name] = agg
        print(f"  {cond_name:<13} graded={agg['graded_score']:.3f}  strict={agg['strict_accuracy']:.3f}  (n={agg['n']})")

    # ── Human-validation worksheet (random subset, pooled over conditions) ─────
    pooled = [
        {"key": f"{cond}::{r['id']}", "condition": cond, "id": r["id"], "type": r["type"],
         "question": r["question"], "reference": r["reference"], "candidate": r["answer"],
         "judge_label": r["judge_label"], "human_label": ""}
        for cond, rows in results.items() for r in rows
    ]
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(pooled), size=min(args.subset_n, len(pooled)), replace=False)
    worksheet = {
        "instructions": (
            "Human validation for the LLM judge (PROPOSAL §6.2). For each item, read the "
            "QUESTION, the REFERENCE (known-correct), and the CANDIDATE, then set "
            "\"human_label\" to one of: correct | partial | incorrect (same rubric the judge "
            "used: judge factual coverage vs the reference, ignore style/length/citations). "
            "Do NOT look at \"judge_label\" while deciding. Then run: "
            "python scripts/qa_accuracy.py --kappa data/eval/qa_accuracy_worksheet.json"
        ),
        "seed": args.seed,
        "items": [pooled[i] for i in sorted(pick)],
    }
    Path(args.worksheet).write_text(json.dumps(worksheet, ensure_ascii=False, indent=2), encoding="utf-8")

    out = {
        "gold": gold["source"], "rerank": use_rerank,
        "answer_model": qcfg["model"], "judge_model": args.judge_model,
        "summary": summary, "figure_summary": fig_summary,
        "conditions": results,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote per-answer rows + metrics to {args.out}")
    print(
        f"Wrote {len(worksheet['items'])}-item human-label worksheet to {args.worksheet}\n"
        f"  -> label each item's \"human_label\", then: "
        f"python scripts/qa_accuracy.py --kappa {args.worksheet}"
    )


if __name__ == "__main__":
    main()
