"""Figure understanding (spec module 6.3): caption figures with the Gemini VLM.

Generated captions provide richer semantic information than OCR alone — they
describe pipeline stages, relationships, and the meaning of a diagram rather
than just the literal text baked into it. Each extracted figure gets one
caption, which becomes part of the figure's retrievable chunk.
"""
from __future__ import annotations

import mimetypes
import time
from pathlib import Path

DEFAULT_PROMPT = (
    "Describe this academic figure in detail. Focus on: pipeline stages, "
    "relationships between components, technical/mathematical content, and the "
    "figure's overall semantic meaning. Be specific and concise (no preamble)."
)

# Substrings that mark a *transient* server condition worth retrying (vs. a
# permanent failure like a safety block, which we return "" for immediately).
_TRANSIENT = ("503", "502", "500", "429", "unavailable", "overloaded", "resource_exhausted")


class FigureCaptioner:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        prompt: str = DEFAULT_PROMPT,
        max_output_tokens: int = 512,
        max_retries: int = 5,
    ) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.prompt = prompt or DEFAULT_PROMPT
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries

    def caption(self, image_path: str | Path) -> str:
        """Return a semantic caption for one figure, or "" on permanent failure.

        Transient server errors (503/429/overloaded — common on the shared
        flash-lite tier) are retried with exponential backoff so a demand spike
        mid-ingest doesn't silently drop captions. The caller does not cache
        empties, so anything still empty after retries is retried on the next run.
        """
        image_path = Path(image_path)
        try:
            data = image_path.read_bytes()
        except OSError:
            return ""
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"

        part = self._types.Part.from_bytes(data=data, mime_type=mime)
        config = self._types.GenerateContentConfig(
            temperature=0.2, max_output_tokens=self.max_output_tokens
        )
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=[part, self.prompt], config=config
                )
                return (resp.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                transient = any(t in str(exc).lower() for t in _TRANSIENT)
                if not transient or attempt == self.max_retries:
                    return ""
                time.sleep(min(2 ** attempt, 30))  # 1, 2, 4, 8, 16, capped 30s
        return ""
