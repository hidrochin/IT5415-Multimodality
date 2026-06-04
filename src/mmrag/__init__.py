"""mmrag — Research-oriented Multimodal RAG for academic documents.

Phase 1 (MVP) exposes a text-RAG vertical slice:
    parse PDF -> chunk -> embed -> FAISS retrieve -> Gemini grounded answer.

Heavy modules (embeddings, retrieval, qa) are imported lazily so that
`import mmrag` stays cheap and side-effect free.
"""

from .config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "0.1.0"
