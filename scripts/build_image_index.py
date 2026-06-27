"""CLI: build the cross-modal image index (the *embed-the-image* arm, T6 / H3).

Encodes every retrievable **visual** chunk's image (slide/figure PNGs from the
FAISS chunk store) with a CLIP encoder into one shared text-image space, and
saves the vectors + chunk metadata to ``image_embeddings.index_dir``.

    python scripts/build_image_index.py            # encode all visual chunks (CLIP)
    python scripts/build_image_index.py --limit 8  # quick local smoke (few images)
    python scripts/build_image_index.py --pack      # zip a Colab payload, encode nothing

Encoding a vision transformer over hundreds of images is GPU-bound; on this
Windows box it is slow. The intended path is the Colab notebook
``notebooks/build_image_index.ipynb`` (GPU): run ``--pack`` here to produce
``data/processed/image_payload.zip`` (chunks.json + just the referenced PNGs),
upload it to the notebook, and drop the notebook's ``image_index/`` output back
into ``data/processed/``. ``--pack`` needs no torch, so it always runs locally.
"""
import argparse
import json
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path, forces UTF-8 stdout)
from mmrag.config import load_config
from mmrag.image_embeddings import ImageEmbedder, ImageIndex, visual_chunks
from mmrag.retrieval import FaissIndex


def _load_visual_chunks(cfg) -> list[dict]:
    index = FaissIndex.load(cfg.index_dir)
    chunks = visual_chunks(index.chunks)
    if not chunks:
        raise SystemExit(
            "No visual chunks (image_path) in the index. Ingest with figures.mode=slide first."
        )
    return chunks


def _resolve_device(cfg) -> str:
    dev = cfg.get("image_embeddings", {}).get("device", "auto")
    if dev and dev != "auto":
        return dev
    return cfg.resolve_device()


def _pack_payload(cfg, chunks: list[dict]) -> Path:
    """Zip chunks.json + only the referenced PNGs for upload to the GPU notebook."""
    out = cfg.data_processed / "image_payload.zip"
    seen: set[str] = set()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        # A flat manifest the notebook reads: arcname (in zip) -> chunk dict.
        manifest = []
        for c in chunks:
            src = Path(c["image_path"])
            arc = f"slides/{src.name}"
            if src.name not in seen:
                if not src.exists():
                    raise SystemExit(f"Missing image referenced by a chunk: {src}")
                z.write(src, arc)
                seen.add(src.name)
            entry = dict(c)
            entry["arcname"] = arc
            manifest.append(entry)
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the cross-modal CLIP image index.")
    ap.add_argument("--limit", type=int, default=0, help="Encode only the first N images (smoke).")
    ap.add_argument("--pack", action="store_true", help="Only zip a Colab payload; do not encode.")
    args = ap.parse_args()

    cfg = load_config()
    icfg = cfg.get("image_embeddings", {})
    index_dir = cfg._resolve(icfg.get("index_dir", "data/processed/image_index"))

    chunks = _load_visual_chunks(cfg)
    print(f"Visual chunks in index: {len(chunks)} (over {len({c['source'] for c in chunks})} decks)")

    if args.pack:
        out = _pack_payload(cfg, chunks)
        size_mb = out.stat().st_size / 1e6
        print(
            f"Packed Colab payload: {out} ({size_mb:.1f} MB, {len(chunks)} chunks).\n"
            f"Upload it to notebooks/build_image_index.ipynb, then place the notebook's\n"
            f"image_index/ output into {cfg.data_processed}."
        )
        return

    if args.limit > 0:
        chunks = chunks[: args.limit]
        print(f"--limit {args.limit}: encoding {len(chunks)} images only (smoke).")

    device = _resolve_device(cfg)
    model_name = icfg.get("model", "clip-ViT-B-32")
    print(f"Encoding {len(chunks)} images with {model_name} on {device} ...")

    embedder = ImageEmbedder(
        model_name=model_name,
        device=device,
        batch_size=int(icfg.get("batch_size", 32)),
    )
    vecs = embedder.encode_images([c["image_path"] for c in chunks])
    image_index = ImageIndex(model_name=model_name, vecs=vecs, chunks=chunks)
    image_index.save(index_dir)

    print(
        f"Image index built: {len(chunks)} vectors, dim={image_index.dim}.\n"
        f"Saved to {index_dir}.  Probe it with scripts/probe_image_search.py."
    )


if __name__ == "__main__":
    main()
