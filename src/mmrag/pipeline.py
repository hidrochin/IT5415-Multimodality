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
from .chunking import chunk_document
from .parsing import parse_pdf


class MultimodalRAG:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or load_config()
        self._embedder = None  # lazily built (loads torch)
        self._reranker = None  # lazily built

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
        if not chunk_dicts:
            raise ValueError(f"No usable text chunks extracted from {pdf_path}")

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
            "images": parsed.num_images,
            "index_dir": str(index_path),
        }

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
