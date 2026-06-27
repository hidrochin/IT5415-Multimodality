"""End-to-end pipeline: ingest documents and answer questions.

Phase 1 vertical slice::

    rag = MultimodalRAG()
    rag.ingest("data/raw/paper.pdf")
    out = rag.ask("What is self-attention?")
    print(out["answer"])
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from .config import Config, load_config
from .chunking import build_figure_chunk, build_slide_chunk, chunk_document
from .parsing import ParsedDocument, parse_pdf


class MultimodalRAG:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or load_config()
        self._embedder = None  # lazily built (loads torch)
        self._reranker = None  # lazily built
        self._ocr = None       # lazily built (loads PaddleOCR)
        self._captioner = None  # lazily built (Gemini)
        self._caption_cache: Optional[dict[str, str]] = None  # resumable VLM captions
        self._caption_dirty = 0

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
    def _parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse one PDF, honouring the configured image/render paths."""
        pcfg = self.cfg["parsing"]
        slide_mode = self.cfg.get("figures", {}).get("mode", "figure") == "slide"
        return parse_pdf(
            pdf_path,
            image_out_dir=self.cfg.data_processed / "images",
            # In slide mode we caption the whole rendered page, so per-figure
            # extraction is unnecessary work — skip it unless explicitly on.
            extract_images=pcfg.get("extract_images", True) and not slide_mode,
            min_image_size=pcfg.get("min_image_size", 64),
            render_pages=pcfg.get("render_pages", False) or slide_mode,
            render_dpi=pcfg.get("render_dpi", 130),
            render_out_dir=self.cfg.data_processed / "slides",
        )

    def _chunk(
        self, parsed: ParsedDocument, caption: bool = True
    ) -> tuple[list[dict], list[dict]]:
        """Return (text-chunk dicts, figure/slide-chunk dicts) for one document.

        ``caption=False`` ingests the document text-only (no VLM calls) — used for
        distractor decks in a shared index, to keep API cost on the eval decks only.
        """
        ccfg = self.cfg["chunking"]
        text_chunks = chunk_document(
            parsed,
            chunk_size=ccfg["chunk_size"],
            chunk_overlap=ccfg["chunk_overlap"],
            min_chunk_chars=ccfg.get("min_chunk_chars", 50),
        )
        text_dicts = [c.to_dict() for c in text_chunks]
        figure_dicts = self._build_figure_chunks(parsed, caption=caption)
        return text_dicts, figure_dicts

    def ingest(self, pdf_path: str | Path, rebuild: bool = True) -> dict[str, Any]:
        """Parse -> chunk -> embed -> (re)build FAISS index on disk."""
        from .retrieval import FaissIndex

        parsed = self._parse(pdf_path)
        text_dicts, figure_dicts = self._chunk(parsed)
        chunk_dicts = text_dicts + figure_dicts
        self._flush_caption_cache()

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
            "text_chunks": len(text_dicts),
            "figure_chunks": len(figure_dicts),
            "images": parsed.num_images,
            "index_dir": str(index_path),
        }

    def ingest_corpus(
        self,
        pdf_paths: list[str | Path],
        caption_sources: Optional[set[str]] = None,
        progress: Optional[Callable[[dict], None]] = None,
    ) -> dict[str, Any]:
        """Ingest many PDFs into one shared index (single embed + save at the end).

        ``caption_sources`` (file names) limits which decks get VLM captions; the
        rest are ingested text-only as distractors, keeping API cost on the eval
        decks. ``None`` captions every deck. Captions are cached on disk
        (``caption_cache.json``), so a re-run after an interruption skips
        already-captioned slides instead of re-calling Gemini. Each chunk keeps its
        ``source``, so downstream eval can stay source-aware on this multi-deck
        index (page numbers alone collide across decks).
        """
        from .retrieval import FaissIndex

        all_chunks: list[dict] = []
        per_doc: list[dict] = []
        for path in pdf_paths:
            parsed = self._parse(path)
            do_caption = caption_sources is None or parsed.source in caption_sources
            text_dicts, figure_dicts = self._chunk(parsed, caption=do_caption)
            self._flush_caption_cache()
            all_chunks.extend(text_dicts)
            all_chunks.extend(figure_dicts)
            stat = {
                "source": parsed.source,
                "pages": len(parsed.pages),
                "text_chunks": len(text_dicts),
                "figure_chunks": len(figure_dicts),
                "captioned": do_caption,
            }
            per_doc.append(stat)
            if progress is not None:
                progress(stat)

        if not all_chunks:
            raise ValueError("No usable chunks extracted from the corpus.")

        embeddings = self.embedder.encode_passages([c["text"] for c in all_chunks])
        index = FaissIndex(self.embedder.dim)
        index.add(embeddings, all_chunks)
        index.save(self.cfg.index_dir)

        return {
            "docs": len(per_doc),
            "pages": sum(d["pages"] for d in per_doc),
            "chunks": len(all_chunks),
            "text_chunks": sum(d["text_chunks"] for d in per_doc),
            "figure_chunks": sum(d["figure_chunks"] for d in per_doc),
            "per_doc": per_doc,
            "index_dir": str(self.cfg.index_dir),
        }

    # ── Caption cache (resumable, concurrent VLM captioning) ──────────────────
    def _ensure_cache(self) -> dict[str, str]:
        if self._caption_cache is None:
            path = self.cfg.data_processed / "caption_cache.json"
            self._caption_cache = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            )
        return self._caption_cache

    def _caption_for(self, key: str, image_path: str) -> str:
        """Return a Gemini caption for one figure, using the disk cache."""
        cache = self._ensure_cache()
        if cache.get(key):  # only non-empty captions are cached
            return cache[key]
        caption = self.captioner.caption(image_path) if self.captioner is not None else ""
        if caption:  # don't cache transient failures — let a re-run retry them
            cache[key] = caption
            self._caption_dirty += 1
            if self._caption_dirty >= 10:
                self._flush_caption_cache()
        return caption

    def _caption_many(self, items: list[tuple[str, str]]) -> None:
        """Caption uncached (key, image_path) items concurrently into the cache.

        Each Gemini call blocks ~5–8 s and the shared flash-lite tier 503s under
        load, so a thread pool overlaps the waits/retries (≈ ``concurrency``×
        throughput). Worker threads only *read* shared state; results are consumed
        serially in this thread, so cache writes need no lock. Empties aren't
        cached, so a re-run retries anything that failed.
        """
        cache = self._ensure_cache()
        pending = [(k, p) for k, p in items if not cache.get(k)]
        if not pending or self.captioner is None:
            return
        workers = max(1, int(self.cfg.get("figures", {}).get("concurrency", 8)))

        def _one(kp: tuple[str, str]) -> tuple[str, str]:
            return kp[0], self.captioner.caption(kp[1])

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for key, caption in ex.map(_one, pending):
                if caption:
                    cache[key] = caption
                    self._caption_dirty += 1
                    if self._caption_dirty >= 20:
                        self._flush_caption_cache()
        self._flush_caption_cache()

    def _flush_caption_cache(self) -> None:
        if not self._caption_cache or not self._caption_dirty:
            return
        path = self.cfg.data_processed / "caption_cache.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._caption_cache, ensure_ascii=False), encoding="utf-8"
        )
        self._caption_dirty = 0

    def _build_figure_chunks(
        self, parsed: ParsedDocument, caption: bool = True
    ) -> list[dict]:
        """Caption (+OCR) figures into retrievable chunks.

        Slide mode: one chunk per *rendered page* (the slide image). Figure mode:
        one chunk per *extracted embedded image* (Phase-2 behaviour). Either way a
        figure with no caption and no OCR text is skipped, never indexed blank.
        ``caption=False`` disables the VLM call for this document (text-only).
        """
        ocr_on = self.cfg.get("ocr", {}).get("enabled", False)
        cap_on = caption and self.cfg.get("figures", {}).get("enabled", False)
        if not (ocr_on or cap_on):
            return []

        if self.cfg.get("figures", {}).get("mode", "figure") == "slide":
            return self._build_slide_chunks(parsed, ocr_on, cap_on)

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
                    caption = self._caption_for(f"{parsed.source}::p{page.page_number}::fig{idx}", image_path)
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

    def _build_slide_chunks(
        self, parsed: ParsedDocument, ocr_on: bool, cap_on: bool
    ) -> list[dict]:
        """One caption(+OCR) chunk per rendered slide page (captions concurrent)."""
        pages = [p for p in parsed.pages if p.render_path]

        def _key(p: "Any") -> str:
            return f"{parsed.source}::p{p.page_number}::slide"

        if cap_on:
            self._caption_many([(_key(p), p.render_path) for p in pages])
        cache = self._ensure_cache()

        slide_dicts: list[dict] = []
        for page in pages:
            caption = cache.get(_key(page), "") if cap_on else ""

            ocr_text = ""
            if ocr_on:
                try:
                    ocr_text = self.ocr.extract_text(page.render_path)
                except Exception:
                    ocr_text = ""

            chunk = build_slide_chunk(
                source=parsed.source,
                page=page.page_number,
                image_path=page.render_path,
                caption=caption,
                ocr_text=ocr_text,
            )
            if chunk is not None:
                slide_dicts.append(chunk.to_dict())

        return slide_dicts

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
