"""OCR (spec module 6.2): read text baked into figures with PaddleOCR.

PaddleOCR is lightweight, multilingual, and Colab-friendly. Initialization is
expensive (loads detection + recognition models), so build one ``OCREngine`` and
reuse it across every image in a document.

The PaddleOCR return format changed between 2.x and 3.x; :meth:`extract_text`
parses both shapes and never raises on a single bad image — it returns "".
"""
from __future__ import annotations

from pathlib import Path


class OCREngine:
    def __init__(
        self,
        lang: str = "en",
        min_confidence: float = 0.5,
        use_gpu: bool = False,
    ) -> None:
        from paddleocr import PaddleOCR

        self.min_confidence = min_confidence
        # PaddleOCR renamed/added kwargs across versions; pass what each accepts.
        self._ocr = _build_paddleocr(PaddleOCR, lang=lang, use_gpu=use_gpu)

    def extract_text(self, image_path: str | Path) -> str:
        """Return confidence-filtered text from one image (newline-joined lines)."""
        image_path = str(image_path)
        try:
            raw = self._run(image_path)
        except Exception:
            return ""
        lines = [t for (t, conf) in raw if conf >= self.min_confidence]
        return "\n".join(lines).strip()

    def _run(self, image_path: str) -> list[tuple[str, float]]:
        # Prefer the 3.x predict() API; fall back to the classic ocr() call.
        predict = getattr(self._ocr, "predict", None)
        if callable(predict):
            try:
                return _parse_result(predict(image_path))
            except Exception:
                pass
        return _parse_result(self._ocr.ocr(image_path))


def _build_paddleocr(PaddleOCR, lang: str, use_gpu: bool):
    """Construct PaddleOCR, tolerating kwargs that differ across versions."""
    # Try richer kwargs first, then progressively drop unsupported ones.
    attempts = [
        dict(lang=lang, use_textline_orientation=True),  # 3.x
        dict(lang=lang, use_angle_cls=True, use_gpu=use_gpu, show_log=False),  # 2.x
        dict(lang=lang, use_angle_cls=True),
        dict(lang=lang),
    ]
    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError) as e:
            last_err = e
    raise RuntimeError(f"Could not initialize PaddleOCR: {last_err}")


def _parse_result(result) -> list[tuple[str, float]]:
    """Normalize PaddleOCR output into ``[(text, confidence), ...]``.

    Handles both shapes:
      * 3.x predict(): list of dict-like objects with ``rec_texts`` + ``rec_scores``
      * 2.x ocr():     ``[[ [box, (text, conf)], ... ]]`` per image
    """
    out: list[tuple[str, float]] = []
    if not result:
        return out

    for page in result:
        if page is None:
            continue
        # 3.x: dict-like with parallel rec_texts / rec_scores arrays.
        texts = _get(page, "rec_texts")
        scores = _get(page, "rec_scores")
        if texts is not None:
            scores = scores or [1.0] * len(texts)
            for t, s in zip(texts, scores):
                if t:
                    out.append((str(t), float(s)))
            continue
        # 2.x: iterable of [box, (text, conf)] lines.
        try:
            for line in page:
                info = line[1]
                text, conf = info[0], float(info[1])
                if text:
                    out.append((str(text), conf))
        except (TypeError, IndexError, ValueError):
            continue
    return out


def _get(obj, key):
    """Read ``key`` from a dict or an attribute-style result object."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
