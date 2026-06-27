"""CLI: probe the cross-modal image index (T6 DoD — "sane hits on a probe").

Encodes a text query with the SAME CLIP model used to build the image index and
returns the top visual chunks by cosine similarity in the shared space — a quick
sanity check that embed-the-image retrieval surfaces the right slides before the
full fusion eval (T7). The text query encoder is light (one string), so this runs
fine locally even though *building* the index wants a GPU.

    python scripts/probe_image_search.py "scaled dot-product attention diagram"
    python scripts/probe_image_search.py "the modality gap" --top-k 8

With no query argument, a few built-in probes are run.
"""
import argparse

import _bootstrap  # noqa: F401  (adds src/ to sys.path, forces UTF-8 stdout)
from mmrag.config import load_config
from mmrag.image_embeddings import ImageEmbedder, ImageIndex

DEFAULT_PROBES = [
    "scaled dot-product attention architecture diagram",
    "a scatter plot of data points",
    "table comparing model results",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe the cross-modal CLIP image index.")
    ap.add_argument("query", nargs="*", help="Text query (default: built-in probes).")
    ap.add_argument("--top-k", type=int, default=5, help="Hits to show per query.")
    args = ap.parse_args()

    cfg = load_config()
    icfg = cfg.get("image_embeddings", {})
    index_dir = cfg._resolve(icfg.get("index_dir", "data/processed/image_index"))

    index = ImageIndex.load(index_dir)
    print(f"Image index: {len(index.chunks)} vectors, dim={index.dim}, model={index.model_name}\n")

    embedder = ImageEmbedder(
        model_name=index.model_name or icfg.get("model", "clip-ViT-B-32"),
        device=cfg.resolve_device(),
        batch_size=int(icfg.get("batch_size", 32)),
    )

    queries = [" ".join(args.query)] if args.query else DEFAULT_PROBES
    for q in queries:
        q_vec = embedder.encode_queries([q])[0]
        hits = index.search(q_vec, top_k=args.top_k)
        print(f"Q: {q}")
        for rank, h in enumerate(hits, 1):
            cap = (h.get("figure_caption") or "").replace("\n", " ")
            cap = (cap[:90] + "…") if len(cap) > 90 else cap
            print(f"  {rank}. {h['score']:.3f}  {h['source']} p{h['page']}  {cap}")
        print()


if __name__ == "__main__":
    main()
