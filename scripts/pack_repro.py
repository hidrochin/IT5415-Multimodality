"""CLI: build the reproducibility pack (T10).

Bundles everything a fresh machine (or the Colab notebook
``notebooks/reproduce_eval.ipynb``) needs to **reproduce the eval tables** — code,
config, the hand-authored gold set, and the *prebuilt* indexes — into a single
zip, plus a ``REPRO_MANIFEST.json`` that pins model IDs / config / seeds / dataset
stats / package versions / git commit (PROPOSAL §9).

    python scripts/pack_repro.py                 # -> data/processed/repro_pack.zip (~6 MB)
    python scripts/pack_repro.py --with-payload  # also embed image_payload.zip (~63 MB,
                                                 #   the slide PNGs, to GPU-rebuild the T6 index)

Why ship the prebuilt indexes rather than re-ingest? Ingest re-runs ~250 Gemini
caption calls (cost + nondeterminism); the eval reproduce is meant to be
deterministic and key-free for the retrieval/fusion tables. The text FAISS index
(BGE-M3 over 1226 chunks) and the CLIP image index (252 visual chunks) are small
and seed-independent, so shipping them makes the eval one-click. Live-answer evals
(faithfulness, QA-accuracy) still need GEMINI_API_KEY at run time.

This script imports no torch / mmrag heavy modules — it just reads files — so it
always runs locally and fast.
"""
import argparse
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Files/dirs copied verbatim into the pack (relative to ROOT). Globs expanded below.
CODE_GLOBS = [
    "src/mmrag/*.py",
    "scripts/*.py",
]
DATA_FILES = [
    "config.yaml",
    "requirements.txt",
    "README.md",
    "data/eval/slides_qa.json",
    "data/eval/slides_qa_gold_answers.json",
    "data/eval/attention_qa.json",
    "data/eval/qa_accuracy_worksheet.json",
    "data/processed/index/index.faiss",
    "data/processed/index/chunks.json",
    "data/processed/image_index/image_vecs.npy",
    "data/processed/image_index/image_index.json",
    "data/processed/ingest_stats.json",
]
# Key third-party packages whose versions we want pinned in the manifest.
PINNED_PKGS = [
    "sentence-transformers", "faiss-cpu", "torch", "transformers",
    "google-genai", "rank-bm25", "numpy", "pymupdf", "gradio",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _pkg_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for name in PINNED_PKGS:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _safe_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_manifest(cfg: dict) -> dict:
    """Pin model IDs, seeds, config, dataset stats, and env — the §9 record."""
    emb = cfg.get("embeddings", {})
    img = cfg.get("image_embeddings", {})
    fig = cfg.get("figures", {})
    qa = cfg.get("qa", {})
    rr = cfg.get("rerank", {})
    ret = cfg.get("retrieval", {})

    stats = _safe_json(ROOT / "data/processed/ingest_stats.json") or {}
    gold = _safe_json(ROOT / "data/eval/slides_qa.json") or {}
    qa_pairs = gold.get("qa_pairs", [])
    img_index = _safe_json(ROOT / "data/processed/image_index/image_index.json") or {}

    return {
        "name": "multimodal-rag reproducibility pack",
        "git_commit": _git_commit(),
        "created_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "models": {
            # Every pinned checkpoint / API model the pipeline touches.
            "text_embedder": emb.get("text_model"),
            "reranker": rr.get("model"),
            "image_embedder": img.get("model"),
            "captioner": fig.get("model"),
            "answerer": qa.get("model"),
            "judge": "gemini-2.5-flash",  # scripts/qa_accuracy.py --judge-model default
        },
        "seeds": {
            # Bootstrap CIs are the only stochastic step; pinned to 0 everywhere
            # (mmrag.evaluation / .faithfulness / .judge default seed=0).
            "bootstrap_seed": 0,
            "bootstrap_n_resamples": 1000,
        },
        "retrieval": {
            "candidate_k": ret.get("candidate_k"),
            "top_k": ret.get("top_k"),
            "chunk_size": cfg.get("chunking", {}).get("chunk_size"),
            "chunk_overlap": cfg.get("chunking", {}).get("chunk_overlap"),
            "rerank_enabled": rr.get("enabled"),
            "fusion_k0": 60,            # RRF rank-damping (scripts/evaluate_fusion.py)
            "fusion_best_alpha": 0.8,   # z-linear optimum found in T7
        },
        "dataset": {
            "decks": stats.get("docs"),
            "pages": stats.get("pages"),
            "chunks": stats.get("chunks"),
            "text_chunks": stats.get("text_chunks"),
            "figure_chunks": stats.get("figure_chunks"),
            "captioned_decks": [d["source"] for d in stats.get("per_doc", []) if d.get("captioned")],
            "image_index_visual_chunks": len(img_index.get("chunks", [])),
            "image_index_dim": img_index.get("dim"),
            "gold_qa_total": len(qa_pairs),
            "gold_qa_figure": sum(1 for q in qa_pairs if q.get("type") == "figure"),
            "gold_qa_text": sum(1 for q in qa_pairs if q.get("type") == "text"),
        },
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _pkg_versions(),
        },
        "reproduce": {
            # The exact commands the notebook runs, in order. The first two are
            # key-free (retrieval/fusion); the last two need GEMINI_API_KEY.
            "retrieval_eval": "python scripts/evaluate.py --gold data/eval/slides_qa.json",
            "fusion_eval": "python scripts/evaluate_fusion.py --gold data/eval/slides_qa.json",
            "faithfulness_eval": "python scripts/faithfulness.py --gold data/eval/slides_qa.json  # needs GEMINI_API_KEY",
            "qa_accuracy_eval": "python scripts/qa_accuracy.py --gold data/eval/slides_qa.json  # needs GEMINI_API_KEY",
            "rebuild_image_index_gpu": "notebooks/build_image_index.ipynb  # GPU; consumes image_payload.zip",
            "windows_guard": "set KMP_DUPLICATE_LIB_OK=TRUE and OMP_NUM_THREADS=1 before faiss+torch scripts",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the reproducibility pack zip (T10).")
    ap.add_argument("--out", default="data/processed/repro_pack.zip", help="Output zip path")
    ap.add_argument("--with-payload", action="store_true",
                    help="Also embed data/processed/image_payload.zip (slide PNGs) for the GPU T6 rebuild")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    manifest = build_manifest(cfg)

    # Gather the file list (skip __pycache__, warn on anything missing).
    members: list[tuple[Path, str]] = []
    for glob in CODE_GLOBS:
        for p in sorted(ROOT.glob(glob)):
            if "__pycache__" in p.parts:
                continue
            members.append((p, p.relative_to(ROOT).as_posix()))
    missing = []
    for rel in DATA_FILES:
        p = ROOT / rel
        if p.exists():
            members.append((p, rel))
        else:
            missing.append(rel)
    if args.with_payload:
        payload = ROOT / "data/processed/image_payload.zip"
        if payload.exists():
            members.append((payload, "data/processed/image_payload.zip"))
        else:
            missing.append("data/processed/image_payload.zip (run build_image_index.py --pack)")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("REPRO_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for src, arc in members:
            z.write(src, arc)

    size_mb = out_path.stat().st_size / 1e6
    # Mirror the manifest next to the zip so it's readable without unzipping.
    (out_path.parent / "REPRO_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote {out_path}  ({size_mb:.1f} MB, {len(members) + 1} entries)")
    print(f"  models : {manifest['models']}")
    print(f"  dataset: {manifest['dataset']['decks']} decks / {manifest['dataset']['chunks']} chunks / "
          f"{manifest['dataset']['gold_qa_total']} gold QA "
          f"({manifest['dataset']['gold_qa_figure']} fig + {manifest['dataset']['gold_qa_text']} text)")
    print(f"  commit : {manifest['git_commit']}  python {manifest['env']['python']}")
    if missing:
        print("\n  WARNING — these expected files were absent and NOT packed:", file=sys.stderr)
        for m in missing:
            print(f"    - {m}", file=sys.stderr)
        if not args.with_payload:
            print("  (image_payload.zip is only needed for the GPU T6 rebuild; pass --with-payload to include it)")


if __name__ == "__main__":
    main()
