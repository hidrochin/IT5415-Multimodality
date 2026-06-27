"""Gradio UI for the Multimodal RAG (task T9 — the demo deliverable).

A query box drives the *production* retrieve+answer path (`MultimodalRAG.ask`):
dense FAISS -> cross-encoder rerank -> top_k -> grounded Gemini answer. The page
then shows, side by side:

    * the grounded answer with its [pN] citations,
    * a thumbnail gallery of the *visual* evidence (rendered slides), and
    * a ranked breakdown of every retrieved chunk (source, page, score, snippet),
      flagging which chunks the answer actually cited.

Run (needs an index from scripts/ingest_corpus.py; GEMINI_API_KEY in .env for the
answer — retrieval/evidence still work without a key):

    ./.venv/Scripts/python.exe scripts/app.py
    ./.venv/Scripts/python.exe scripts/app.py --share   # public Gradio link
"""
import argparse
import os
import re

# Same Windows guard the eval scripts use: faiss + torch both vendor OpenMP and
# segfault when two copies load. Set before either is imported (via pipeline).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import _bootstrap  # noqa: F401  (adds src/ to sys.path, forces UTF-8 stdout)

import gradio as gr

from mmrag.config import load_config
from mmrag.pipeline import MultimodalRAG

_CITE_RE = re.compile(r"\[p(\d+)\]", re.IGNORECASE)

# Built once, lazily, on the first query — loads torch + the embedder/reranker.
_RAG: MultimodalRAG | None = None


def _get_rag() -> MultimodalRAG:
    global _RAG
    if _RAG is None:
        _RAG = MultimodalRAG()
    return _RAG


def _cited_pages(answer: str) -> set[int]:
    """The set of page numbers the answer cited via [pN]."""
    return {int(m) for m in _CITE_RE.findall(answer or "")}


def _score_str(hit: dict) -> str:
    if "rerank_score" in hit:
        return f"rerank {hit['rerank_score']:.2f} · faiss {hit['score']:.3f}"
    return f"score {hit['score']:.3f}"


def _evidence_markdown(hits: list[dict], cited: set[int]) -> str:
    """Ranked, human-readable breakdown of every retrieved chunk."""
    if not hits:
        return "_No evidence retrieved._"
    lines = ["### Retrieved evidence"]
    for i, h in enumerate(hits, 1):
        snippet = " ".join((h.get("text") or "").split())[:320]
        kind = "🖼️ slide" if h.get("image_path") else "📄 text"
        flag = " ✅ cited" if h.get("page") in cited else ""
        lines.append(
            f"**{i}. {kind} · p{h.get('page')} · `{h.get('source')}`**  "
            f"<sub>{_score_str(h)}{flag}</sub>\n\n{snippet}…"
        )
    return "\n\n".join(lines)


def _gallery(hits: list[dict], cited: set[int]) -> list[tuple[str, str]]:
    """(thumbnail, caption) pairs for the visual (slide) chunks among the hits."""
    items: list[tuple[str, str]] = []
    for i, h in enumerate(hits, 1):
        path = h.get("image_path")
        if not path or not os.path.exists(path):
            continue
        flag = " ✅" if h.get("page") in cited else ""
        items.append((path, f"#{i} · p{h.get('page')} · {h.get('source')}{flag}"))
    return items


def answer_question(question: str, top_k: int):
    """Gradio callback: retrieve + answer, returning (answer_md, gallery, evidence_md)."""
    question = (question or "").strip()
    if not question:
        return "_Enter a question above._", [], ""

    out = _get_rag().ask(question, top_k=int(top_k))
    hits = out.get("sources", [])

    if out.get("answer"):
        cited = _cited_pages(out["answer"])
        answer_md = f"### Answer\n\n{out['answer']}"
    else:
        cited = set()
        err = out.get("error", "No answer produced.")
        answer_md = (
            f"### Answer\n\n_⚠️ {err}_\n\n"
            "Retrieval still ran — see the evidence below."
        )

    return answer_md, _gallery(hits, cited), _evidence_markdown(hits, cited)


def build_demo(cfg) -> gr.Blocks:
    default_k = cfg["retrieval"]["top_k"]
    has_key = bool(cfg.gemini_api_key)

    key_note = (
        "" if has_key else
        "\n\n> ⚠️ `GEMINI_API_KEY` not set — retrieval/evidence work, "
        "but the grounded answer is disabled."
    )

    with gr.Blocks(title="Multimodal RAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Multimodal RAG over lecture slides\n"
            "Ask a question; the system retrieves from slide **text and figures** "
            "(Gemini-captioned, BGE-M3 + reranked) and answers **grounded** in that "
            "evidence with `[pN]` citations." + key_note
        )

        with gr.Row():
            question = gr.Textbox(
                label="Question",
                placeholder="e.g. What is early vs late fusion?",
                scale=5,
                autofocus=True,
            )
            top_k = gr.Slider(
                1, 10, value=default_k, step=1, label="Top-k evidence", scale=1
            )
        ask_btn = gr.Button("Ask", variant="primary")

        gr.Examples(
            examples=[
                ["What is the difference between early and late fusion?"],
                ["Explain the scaled dot-product attention diagram."],
                ["What are the main approaches to multimodal alignment?"],
                ["How do multimodal transformers handle cross-modal attention?"],
            ],
            inputs=question,
        )

        answer_md = gr.Markdown(label="Answer")
        gallery = gr.Gallery(
            label="Visual evidence (retrieved slides)",
            columns=4,
            height="auto",
            object_fit="contain",
        )
        evidence_md = gr.Markdown(label="Evidence")

        outputs = [answer_md, gallery, evidence_md]
        ask_btn.click(answer_question, [question, top_k], outputs)
        question.submit(answer_question, [question, top_k], outputs)

    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Gradio UI for the Multimodal RAG.")
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link")
    ap.add_argument("--port", type=int, default=7860, help="Server port")
    args = ap.parse_args()

    cfg = load_config()
    index_file = cfg.index_dir / "index.faiss"
    if not index_file.exists():
        raise SystemExit(
            f"No index at {cfg.index_dir}. Build it first:\n"
            "  ./.venv/Scripts/python.exe scripts/ingest_corpus.py"
        )

    demo = build_demo(cfg)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
