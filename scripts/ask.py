"""CLI: ask a question against the ingested documents.

    python scripts/ask.py "What is self-attention?"
    python scripts/ask.py "Explain the architecture diagram" -k 8
"""
import argparse

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from mmrag.pipeline import MultimodalRAG


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask a question against the ingested documents.")
    ap.add_argument("question", help="Your question (wrap in quotes)")
    ap.add_argument("-k", "--top-k", type=int, default=None, help="Number of passages to retrieve")
    args = ap.parse_args()

    rag = MultimodalRAG()
    out = rag.ask(args.question, top_k=args.top_k)

    print("\n=== ANSWER ===")
    print(out.get("answer") or f"[no answer] {out.get('error')}")

    print("\n=== SOURCES ===")
    for i, s in enumerate(out["sources"], 1):
        snippet = " ".join(s["text"].split())[:160]
        if "rerank_score" in s:
            score = f"rerank={s['rerank_score']:.2f} (faiss={s['score']:.3f})"
        else:
            score = f"score={s['score']:.3f}"
        print(f"{i}. p{s['page']}  {score}  {s['source']}")
        print(f"   {snippet}...")


if __name__ == "__main__":
    main()
