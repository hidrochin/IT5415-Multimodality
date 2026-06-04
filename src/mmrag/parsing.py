"""Document parsing (spec module 6.1): extract text + images with PyMuPDF."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


@dataclass
class PageContent:
    page_number: int  # 1-indexed
    text: str
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedDocument:
    source: str  # file name, e.g. "attention.pdf"
    path: str
    pages: list[PageContent]

    @property
    def num_images(self) -> int:
        return sum(len(p.images) for p in self.pages)


def parse_pdf(
    pdf_path: str | Path,
    image_out_dir: str | Path | None = None,
    extract_images: bool = True,
    min_image_size: int = 64,
) -> ParsedDocument:
    """Parse a PDF into per-page text and (optionally) extracted images.

    Images are saved as PNGs under ``image_out_dir`` and referenced by path so
    later modules (OCR, figure captioning, image embeddings) can pick them up.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages: list[PageContent] = []
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            images: list[dict[str, Any]] = []
            if extract_images and image_out_dir is not None:
                images = _extract_page_images(
                    doc, page, i + 1, pdf_path.stem, Path(image_out_dir), min_image_size
                )
            pages.append(PageContent(page_number=i + 1, text=text, images=images))
    finally:
        doc.close()

    return ParsedDocument(source=pdf_path.name, path=str(pdf_path), pages=pages)


def _extract_page_images(
    doc: "fitz.Document",
    page: "fitz.Page",
    page_number: int,
    stem: str,
    out_dir: Path,
    min_size: int,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.width < min_size or pix.height < min_size:
                pix = None
                continue
            # Convert CMYK / alpha-bearing images to RGB so PNG save is safe.
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            fname = f"{stem}_p{page_number}_{img_index}.png"
            fpath = out_dir / fname
            pix.save(fpath)
            results.append(
                {
                    "path": str(fpath),
                    "xref": xref,
                    "width": pix.width,
                    "height": pix.height,
                    "page": page_number,
                }
            )
            pix = None
        except Exception:
            # Skip images that PyMuPDF can't decode rather than failing the doc.
            continue
    return results
