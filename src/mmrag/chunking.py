"""Multimodal chunking (spec module 6.4).

Produces chunks with the structure from the spec::

    {
        "id": "...",
        "text": "...",
        "figure_caption": None,   # filled in Phase 2 by the figure module
        "page": 5,
        "image_path": None,       # filled in Phase 2
        "source": "lecture_slide.pdf",
    }
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

from .parsing import ParsedDocument


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    source: str
    figure_caption: Optional[str] = None
    image_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_document(
    parsed: ParsedDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 50,
) -> list[Chunk]:
    """Split each page's text into overlapping, metadata-tagged chunks."""
    chunks: list[Chunk] = []
    for page in parsed.pages:
        text = _clean(page.text)
        for j, piece in enumerate(_split_text(text, chunk_size, chunk_overlap)):
            piece = piece.strip()
            if len(piece) < min_chunk_chars:
                continue
            cid = f"{parsed.source}::p{page.page_number}::c{j}"
            chunks.append(
                Chunk(id=cid, text=piece, page=page.page_number, source=parsed.source)
            )
    return chunks


def _clean(text: str) -> str:
    text = text.replace("\r", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Word-aware sliding window: ~chunk_size chars per chunk with overlap."""
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        cur: list[str] = []
        length = 0
        j = start
        while j < len(words) and length + len(words[j]) + 1 <= chunk_size:
            cur.append(words[j])
            length += len(words[j]) + 1
            j += 1
        if not cur:  # a single word longer than chunk_size
            cur = [words[start]]
            j = start + 1
        chunks.append(" ".join(cur))
        if j >= len(words):
            break
        # Back up by ~overlap characters worth of words for the next window.
        back, k = 0, j
        while k > start and back < overlap:
            k -= 1
            back += len(words[k]) + 1
        start = max(k, start + 1)
    return chunks
