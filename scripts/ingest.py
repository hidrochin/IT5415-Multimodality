"""CLI: ingest a PDF into the multimodal RAG index.

    python scripts/ingest.py data/raw/paper.pdf
    python scripts/ingest.py data/raw/slides.pdf --append
"""
import argparse

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from mmrag.pipeline import MultimodalRAG


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a PDF into the multimodal RAG index.")
    ap.add_argument("pdf", help="Path to a PDF file")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Add to the existing index instead of rebuilding from scratch",
    )
    args = ap.parse_args()

    rag = MultimodalRAG()
    stats = rag.ingest(args.pdf, rebuild=not args.append)
    print(
        f"Ingested {stats['source']}: {stats['pages']} pages, "
        f"{stats['chunks']} chunks "
        f"({stats['text_chunks']} text + {stats['figure_chunks']} figure), "
        f"{stats['images']} images."
    )
    print(f"Index saved to: {stats['index_dir']}")


if __name__ == "__main__":
    main()
