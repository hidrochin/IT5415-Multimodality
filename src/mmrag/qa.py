"""Grounded QA (spec module 6.8) using the Gemini API.

Uses the current `google-genai` SDK::

    from google import genai
    client = genai.Client(api_key=...)
    client.models.generate_content(model=..., contents=..., config=...)
"""
from __future__ import annotations

GROUNDED_SYSTEM = (
    "You are a precise academic assistant. Answer the user's question using ONLY the "
    "provided evidence passages. Do not use any outside knowledge. If the evidence is "
    "insufficient to answer, say so explicitly rather than guessing. "
    "When you cite, cite ONLY page numbers that appear in the evidence headers "
    "(e.g. [p7]); never cite a page that is not shown in the evidence."
)


def format_evidence(chunks: list[dict]) -> str:
    """Render retrieved chunks into a numbered, citable evidence block."""
    blocks: list[str] = []
    for i, c in enumerate(chunks, 1):
        header = f"[Evidence {i} | source={c.get('source')} | p{c.get('page')}]"
        body = c.get("text", "")
        caption = c.get("figure_caption")
        if caption:
            body = f"{body}\n(Figure caption: {caption})"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


class GeminiQA:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def answer(self, question: str, chunks: list[dict]) -> str:
        evidence = format_evidence(chunks)
        prompt = (
            f"Evidence:\n{evidence}\n\n"
            f"Question: {question}\n\n"
            "Answer (grounded strictly in the evidence above, with [pN] citations):"
        )
        config = self._types.GenerateContentConfig(
            system_instruction=GROUNDED_SYSTEM,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        return (resp.text or "").strip()
