"""Self-check: generate a tiny PDF and run parse -> chunk -> embed -> retrieve.

No Gemini key or internet required (after the embedding model is cached once).

    python scripts/smoke_test.py
"""
import pathlib
import tempfile

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
import fitz

from mmrag.chunking import chunk_document
from mmrag.config import load_config
from mmrag.embeddings import TextEmbedder
from mmrag.parsing import parse_pdf
from mmrag.retrieval import FaissIndex

SAMPLE_PAGES = [
    "Self-attention is a mechanism that relates different positions of a single "
    "sequence in order to compute a representation of that sequence. The Transformer "
    "uses multi-head self-attention to model dependencies without recurrence.",
    "The encoder is composed of a stack of identical layers. Each layer has a "
    "multi-head self-attention sub-layer and a position-wise feed-forward network. "
    "Residual connections and layer normalization are applied around each sub-layer.",
]


def make_pdf(path: pathlib.Path) -> None:
    doc = fitz.open()
    for text in SAMPLE_PAGES:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=12)
    doc.save(str(path))
    doc.close()


def main() -> None:
    cfg = load_config()
    with tempfile.TemporaryDirectory() as tmp:
        pdf = pathlib.Path(tmp) / "sample.pdf"
        make_pdf(pdf)

        parsed = parse_pdf(pdf, image_out_dir=None, extract_images=False)
        print(f"[parse]    {len(parsed.pages)} pages")

        ccfg = cfg["chunking"]
        chunks = chunk_document(
            parsed, ccfg["chunk_size"], ccfg["chunk_overlap"], ccfg["min_chunk_chars"]
        )
        print(f"[chunk]    {len(chunks)} chunks")

        ecfg = cfg["embeddings"]
        embedder = TextEmbedder(
            ecfg["text_model"],
            device=cfg.resolve_device(),
            query_prefix=ecfg.get("query_prefix", ""),
            passage_prefix=ecfg.get("passage_prefix", ""),
        )
        chunk_dicts = [c.to_dict() for c in chunks]
        emb = embedder.encode_passages([c["text"] for c in chunk_dicts])
        print(f"[embed]    {emb.shape} via {embedder.model_name} (dim={embedder.dim})")

        index = FaissIndex(embedder.dim)
        index.add(emb, chunk_dicts)

        question = "What is self-attention?"
        hits = index.search(embedder.encode_queries([question]), top_k=3)
        print(f"[retrieve] query: {question!r}")
        for i, h in enumerate(hits, 1):
            snippet = " ".join(h["text"].split())[:64]
            print(f"   {i}. p{h['page']}  score={h['score']:.3f}  {snippet}...")

        assert hits, "retrieval returned no results"
        assert hits[0]["score"] > 0.2, "top score too low — retrieval looks broken"
        assert hits[0]["page"] == 1, "expected the self-attention passage (p1) to rank first"
        print("\nSMOKE TEST PASSED ✓")


if __name__ == "__main__":
    main()
