"""CLI: ingest the whole slide-deck corpus into one shared FAISS index.

Builds a single multi-deck index with the slide-render captioning path (config
``figures.mode: slide``). To control Gemini cost, only the **gold-QA eval decks**
are captioned by default; the rest are ingested text-only as distractors. Captions
are cached on disk, so a re-run after an interruption resumes without re-calling
Gemini.

    python scripts/ingest_corpus.py                 # 14 decks; caption only the 4 eval decks
    python scripts/ingest_corpus.py --caption-all   # caption every deck (full cost)
    python scripts/ingest_corpus.py --include-paper # also index attention_is_all_you_need.pdf
    python scripts/ingest_corpus.py a.pdf b.pdf     # explicit files (all captioned)

Chunks keep their ``source``, so downstream eval can stay source-aware on this
multi-deck index (page numbers alone collide across decks).
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path, forces UTF-8 stdout)
from mmrag.config import load_config
from mmrag.pipeline import MultimodalRAG

PAPER = "attention_is_all_you_need.pdf"  # text-rich contrast, has its own gold set

# The 4 decks the gold QA (T2) and retrieval eval (T3) are authored on — the only
# decks that need captions; everything else is a text-only distractor.
GOLD_DECKS = {
    "lecture3_1-MultimodalFusion.pdf",
    "lecture4.1-MultimodalAlignment.pdf",
    "lecture5_1-MultimodalTransformers-Part1.pdf",
    "Lecture9.1-Generation-Part1.pdf",
}


def _resolve_paths(args, cfg) -> list[Path]:
    if args.pdfs:
        return [Path(p) for p in args.pdfs]
    raw = cfg.data_raw
    decks = sorted(raw.glob("*.pdf"))
    if not args.include_paper:
        decks = [p for p in decks if p.name != PAPER]
    return decks


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest the slide-deck corpus into one index.")
    ap.add_argument("pdfs", nargs="*", help="Explicit PDF paths (default: all decks in data/raw)")
    ap.add_argument("--include-paper", action="store_true", help=f"Also index {PAPER}")
    ap.add_argument("--caption-all", action="store_true", help="Caption every deck (full Gemini cost)")
    args = ap.parse_args()

    cfg = load_config()
    paths = _resolve_paths(args, cfg)
    if not paths:
        raise SystemExit("No PDFs to ingest.")

    if args.pdfs or args.caption_all:
        caption_sources = None  # caption everything
    else:
        caption_sources = GOLD_DECKS

    print(f"Ingesting {len(paths)} document(s) into {cfg.index_dir} (mode=slide):")
    for p in paths:
        tag = "caption" if (caption_sources is None or p.name in caption_sources) else "text-only"
        print(f"  - [{tag:>9}] {p.name}")
    print()

    rag = MultimodalRAG(cfg)

    def _progress(stat: dict) -> None:
        tag = "caption" if stat.get("captioned") else "text-only"
        print(
            f"  [done] [{tag:>9}] {stat['source']:<52} "
            f"{stat['pages']:>4}p  {stat['text_chunks']:>4} text + "
            f"{stat['figure_chunks']:>4} slide chunks"
        )

    stats = rag.ingest_corpus(paths, caption_sources=caption_sources, progress=_progress)

    print(
        f"\nCorpus index built: {stats['docs']} docs, {stats['pages']} pages, "
        f"{stats['chunks']} chunks "
        f"({stats['text_chunks']} text + {stats['figure_chunks']} slide).\n"
        f"Index: {stats['index_dir']}"
    )

    stats_path = cfg.data_processed / "ingest_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stats logged to: {stats_path}")


if __name__ == "__main__":
    main()
