"""End-to-end pipeline: ingest documents and answer questions.

Phase 1 vertical slice::

    rag = MultimodalRAG()
    rag.ingest("data/raw/paper.pdf")
    out = rag.ask("What is self-attention?")
    print(out["answer"])
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .config import Config, load_config
from .chunking import build_figure_chunk, chunk_document
from .parsing import ParsedDocument, parse_pdf


class MultimodalRAG:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or load_config()
        self._embedder = None  # lazily built (loads torch)
        self._reranker = None  # lazily built
        self._ocr = None       # lazily built (loads PaddleOCR)
        self._captioner = None  # lazily built (Gemini)

    @property
    def embedder(self):
        if self._embedder is None:
            from .embeddings import TextEmbedder

            ecfg = self.cfg["embeddings"]
            self._embedder = TextEmbedder(
                model_name=ecfg["text_model"],
                device=self.cfg.resolve_device(),
                batch_size=ecfg.get("batch_size", 32),
                query_prefix=ecfg.get("query_prefix", ""),
                passage_prefix=ecfg.get("passage_prefix", ""),
            )
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from .rerank import Reranker

            rcfg = self.cfg.get("rerank", {})
            self._reranker = Reranker(
                model_name=rcfg.get("model", "BAAI/bge-reranker-base"),
                device=self.cfg.resolve_device(),
                batch_size=rcfg.get("batch_size", 16),
            )
        return self._reranker

    @property
    def ocr(self):
        """PaddleOCR engine (spec module 6.2), or None if unavailable."""
        if self._ocr is None:
            from .ocr import OCREngine

            ocfg = self.cfg.get("ocr", {})
            self._ocr = OCREngine(
                lang=ocfg.get("lang", "en"),
                min_confidence=ocfg.get("min_confidence", 0.5),
                use_gpu=ocfg.get("use_gpu", False),
            )
        return self._ocr

    @property
    def captioner(self):
        """Gemini figure captioner (spec module 6.3), or None if no API key."""
        if self._captioner is None:
            api_key = self.cfg.gemini_api_key
            if not api_key:
                return None
            from .figures import FigureCaptioner

            fcfg = self.cfg.get("figures", {})
            self._captioner = FigureCaptioner(
                api_key=api_key,
                model=fcfg.get("model", "gemini-2.5-flash-lite"),
                prompt=fcfg.get("prompt", ""),
            )
        return self._captioner

    # ── Ingestion ────────────────────────────────────────────────────────────
    def ingest(self, pdf_path: str | Path, rebuild: bool = True) -> dict[str, Any]:
        """Parse -> chunk -> embed -> (re)build FAISS index on disk."""
        from .retrieval import FaissIndex

        pcfg = self.cfg["parsing"]
        ccfg = self.cfg["chunking"]

        parsed = parse_pdf(
            pdf_path,
            image_out_dir=self.cfg.data_processed / "images",
            extract_images=pcfg.get("extract_images", True),
            min_image_size=pcfg.get("min_image_size", 64),
        )
        chunks = chunk_document(
            parsed,
            chunk_size=ccfg["chunk_size"],
            chunk_overlap=ccfg["chunk_overlap"],
            min_chunk_chars=ccfg.get("min_chunk_chars", 50),
        )
        chunk_dicts = [c.to_dict() for c in chunks]

        # Phase 2: OCR + caption each extracted figure into its own chunk so
        # figures become retrievable alongside the body text (Experiment 2).
        figure_dicts = self._build_figure_chunks(parsed)
        chunk_dicts.extend(figure_dicts)

        if not chunk_dicts:
            raise ValueError(f"No usable text or figure chunks extracted from {pdf_path}")

        embeddings = self.embedder.encode_passages([c["text"] for c in chunk_dicts])

        index_path = self.cfg.index_dir
        if not rebuild and (index_path / "index.faiss").exists():
            index = FaissIndex.load(index_path)
        else:
            index = FaissIndex(self.embedder.dim)
        index.add(embeddings, chunk_dicts)
        index.save(index_path)

        return {
            "source": parsed.source,
            "pages": len(parsed.pages),
            "chunks": len(chunk_dicts),
            "text_chunks": len(chunks),
            "figure_chunks": len(figure_dicts),
            "images": parsed.num_images,
            "index_dir": str(index_path),
        }

    def _build_figure_chunks(self, parsed: ParsedDocument) -> list[dict]:
        """OCR + caption every extracted image into retrievable figure chunks."""
        ocr_on = self.cfg.get("ocr", {}).get("enabled", False)
        cap_on = self.cfg.get("figures", {}).get("enabled", False)
        if not (ocr_on or cap_on):
            return []

        max_figs = self.cfg.get("figures", {}).get("max_figures", 0) or 0
        figure_dicts: list[dict] = []
        captioned = 0

        for page in parsed.pages:
            for idx, img in enumerate(page.images):
                image_path = img["path"]

                ocr_text = ""
                if ocr_on:
                    try:
                        ocr_text = self.ocr.extract_text(image_path)
                    except Exception:
                        ocr_text = ""

                caption = ""
                want_caption = cap_on and (max_figs <= 0 or captioned < max_figs)
                if want_caption and self.captioner is not None:
                    caption = self.captioner.caption(image_path)
                    if caption:
                        captioned += 1

                chunk = build_figure_chunk(
                    source=parsed.source,
                    page=page.page_number,
                    index=idx,
                    image_path=image_path,
                    caption=caption,
                    ocr_text=ocr_text,
                )
                if chunk is not None:
                    figure_dicts.append(chunk.to_dict())

        return figure_dicts

    # ── Question answering ───────────────────────────────────────────────────
    def retrieve(self, question: str, top_k: Optional[int] = None) -> list[dict]:
        """Dense FAISS retrieval, optionally refined by the cross-encoder reranker."""
        from .retrieval import FaissIndex

        index = FaissIndex.load(self.cfg.index_dir)
        q_emb = self.embedder.encode_queries([question])

        rcfg = self.cfg["retrieval"]
        final_k = top_k or rcfg["top_k"]
        rerank_cfg = self.cfg.get("rerank", {})

        if rerank_cfg.get("enabled", False):
            candidate_k = max(rcfg.get("candidate_k", 20), final_k)
            candidates = index.search(q_emb, top_k=candidate_k)
            return self.reranker.rerank(question, candidates, top_k=final_k)

        return index.search(q_emb, top_k=final_k)

    def ask(self, question: str, top_k: Optional[int] = None) -> dict[str, Any]:
        """Retrieve evidence, then generate a grounded answer with Gemini."""
        hits = self.retrieve(question, top_k=top_k)

        api_key = self.cfg.gemini_api_key
        if not api_key:
            return {
                "answer": None,
                "error": "GEMINI_API_KEY not set. Copy .env.example to .env and add your key.",
                "sources": hits,
            }

        from .qa import GeminiQA

        qcfg = self.cfg["qa"]
        qa = GeminiQA(
            api_key=api_key,
            model=qcfg["model"],
            temperature=qcfg.get("temperature", 0.2),
            max_output_tokens=qcfg.get("max_output_tokens", 1024),
        )
        answer = qa.answer(question, hits)
        return {"answer": answer, "sources": hits}
