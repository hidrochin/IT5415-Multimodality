"""Cross-modal image embeddings (spec module 6.5b / PROPOSAL §3.2, §5.3) — the
*embed-the-image* arm that H3 puts on trial.

Where ``embeddings.TextEmbedder`` (BGE-M3) realises **describe-then-embed**
(figures are turned into captions and scored in the *text* space), this module
realises **embed-the-image**: a contrastive encoder ``ψ`` (CLIP / SigLIP) maps a
figure *image* and a text *query* into one shared space so retrieval is genuinely
cross-modal — ``s_V(q, c) = ⟨ψ(q), ψ(image_c)⟩``.

We use a CLIP model through ``sentence-transformers`` (no new dependency: the
same package already serves BGE-M3). One ``SentenceTransformer`` CLIP checkpoint
encodes **both** PIL images and text strings into the same normalized space, so
inner product == cosine, exactly as on the text side. SigLIP is a config swap
(open_clip) and left as future work.

Vectors are L2-normalized so a flat inner-product search == cosine similarity,
matching ``FaissIndex``. N is small (one vector per *visual* chunk, hundreds not
millions), so :class:`ImageIndex` keeps the vectors in a numpy matrix and scores
with a single dot product — no faiss needed, which also keeps the Colab build
notebook dependency-light.

The heavy lifting (encoding many images with a vision transformer) is GPU-bound
and is expected to run in ``notebooks/build_image_index.ipynb`` on Colab; this
module is the shared contract both that notebook and the local scripts import.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_VECS_FILE = "image_vecs.npy"
_META_FILE = "image_index.json"


class ImageEmbedder:
    """CLIP/SigLIP wrapper: encode figure images and text queries into one space.

    Mirrors :class:`mmrag.embeddings.TextEmbedder` (lazy torch import, normalized
    float32 output) so the two encoders are interchangeable at the call site.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        # Local import keeps `import mmrag` cheap and torch/PIL optional until used.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        fn = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(fn())

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        """Encode figure/slide PNGs into the contrastive space (normalized)."""
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in image_paths]
        try:
            return self._encode(images)
        finally:
            for im in images:
                im.close()

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode text queries into the *same* space as the images."""
        return self._encode(list(texts))

    def _encode(self, inputs: list) -> np.ndarray:
        emb = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(inputs) > 64,
        )
        return emb.astype("float32")


class ImageIndex:
    """Flat cosine index over figure-image vectors, paired with chunk metadata.

    The visual analogue of :class:`mmrag.retrieval.FaissIndex`, but tiny: one
    vector per visual chunk. ``search`` takes an already-encoded **query vector**
    (so the caller owns the embedder) and returns chunk dicts with a ``score``,
    matching ``FaissIndex.search``'s contract so T7 fusion can treat the two
    ranked lists uniformly.
    """

    def __init__(self, model_name: str, vecs: np.ndarray, chunks: list[dict]) -> None:
        if len(vecs) != len(chunks):
            raise ValueError("vecs and chunks must be the same length")
        self.model_name = model_name
        self.vecs = vecs.astype("float32")
        self.chunks = chunks

    @property
    def dim(self) -> int:
        return int(self.vecs.shape[1]) if self.vecs.size else 0

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        """Cosine search by inner product (both sides L2-normalized)."""
        if not len(self.chunks):
            return []
        q = np.asarray(query_vec, dtype="float32").reshape(-1)
        scores = self.vecs @ q
        top_k = min(top_k, len(self.chunks))
        top_idx = np.argsort(scores)[::-1][:top_k]
        results: list[dict] = []
        for idx in top_idx:
            hit = dict(self.chunks[int(idx)])
            hit["score"] = float(scores[int(idx)])
            results.append(hit)
        return results

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / _VECS_FILE, self.vecs)
        meta = {"model": self.model_name, "dim": self.dim, "chunks": self.chunks}
        with open(index_dir / _META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str | Path) -> "ImageIndex":
        index_dir = Path(index_dir)
        meta_path = index_dir / _META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No image index in {index_dir}. Build it with "
                "notebooks/build_image_index.ipynb (GPU) or scripts/build_image_index.py."
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        vecs = np.load(index_dir / _VECS_FILE)
        return cls(model_name=meta.get("model", ""), vecs=vecs, chunks=meta["chunks"])


def visual_chunks(chunks: list[dict]) -> list[dict]:
    """Select the retrievable *visual* chunks (those with a real ``image_path``).

    Slide/figure chunks are identified downstream by a non-empty ``image_path``
    (same convention as ``build_slide_chunk``), so the image index is built over
    exactly that subset — the chunks for which an image actually exists.
    """
    return [c for c in chunks if c.get("image_path")]
