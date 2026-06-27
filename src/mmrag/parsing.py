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
    render_path: str | None = None  # whole-page PNG (slide-render captioning path)


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
    render_pages: bool = False,
    render_dpi: int = 130,
    render_out_dir: str | Path | None = None,
) -> ParsedDocument:
    """Parse a PDF into per-page text and (optionally) extracted images.

    Two image paths, chosen by the caller:
    - ``extract_images``: pull each embedded raster (per-figure captioning path).
    - ``render_pages``: rasterize the *whole page* to one PNG (slide-render path).
      Slides are text-sparse and figure-heavy; captioning the rendered slide once
      (~1 Gemini call/page) is far cheaper than captioning every embedded fragment
      (~10×) and keeps the diagram in its on-slide context. See PROPOSAL.md §5.1.

    Saved PNGs are referenced by path so later modules (OCR, captioning, image
    embeddings) can pick them up.
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
            render_path = None
            if render_pages and render_out_dir is not None:
                render_path = _render_page(
                    page, i + 1, pdf_path.stem, Path(render_out_dir), render_dpi
                )
            pages.append(
                PageContent(
                    page_number=i + 1, text=text, images=images, render_path=render_path
                )
            )
    finally:
        doc.close()

    return ParsedDocument(source=pdf_path.name, path=str(pdf_path), pages=pages)


def _render_page(
    page: "fitz.Page", page_number: int, stem: str, out_dir: Path, dpi: int
) -> str | None:
    """Rasterize a whole page to a PNG and return its path (None on failure)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{stem}_p{page_number}.png"
    try:
        pix = page.get_pixmap(dpi=dpi)
        pix.save(fpath)
        return str(fpath)
    except Exception:
        return None


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
