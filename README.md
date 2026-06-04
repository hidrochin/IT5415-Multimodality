# Research-Oriented Multimodal RAG

Multimodal Retrieval-Augmented Generation over academic documents and lecture
slides. See [EXPECTATION.md](EXPECTATION.md) for the full research proposal.

This repo is built **incrementally**. Phase 1 (this MVP) is a working
text-RAG vertical slice; later phases add OCR, figure captioning, image
embeddings, reranking, evaluation, and a Gradio UI.

```
parse (PyMuPDF) -> chunk -> embed (sentence-transformers) -> FAISS -> Gemini grounded answer
```

## Quickstart

```powershell
# 1. (recommended) virtual env
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install
pip install -r requirements.txt

# 3. add your Gemini key
copy .env.example .env        # then edit .env and paste your key

# 4. get a sample PDF (or drop your own PDFs into data/raw/)
python scripts/fetch_sample.py

# 5. ingest + ask
python scripts/ingest.py "data/raw/attention_is_all_you_need.pdf"
python scripts/ask.py "What is self-attention?"
```

Get a free Gemini API key at https://aistudio.google.com/app/apikey.

## Project layout

```
config.yaml            # all knobs: models, chunk sizes, top_k, paths
src/mmrag/
  config.py            # load config.yaml + .env
  parsing.py           # 6.1 PyMuPDF text + image extraction
  chunking.py          # 6.4 multimodal chunk structure
  embeddings.py        # 6.5 text embeddings (normalized for cosine)
  retrieval.py         # 6.6 FAISS dense retrieval
  qa.py                # 6.8 Gemini grounded QA
  pipeline.py          # ingest() + ask() orchestration
scripts/
  fetch_sample.py      # download a sample academic PDF
  ingest.py            # build the index from a PDF
  ask.py               # query the index
data/raw/              # input PDFs / slides
data/processed/        # extracted images + FAISS index
```

## Swapping in the spec's models

Edit `config.yaml`:

| Role            | Default                            | Alternatives (set in config.yaml)                          |
| --------------- | ---------------------------------- | ---------------------------------------------------------- |
| Text embedding  | `BAAI/bge-m3` (1024-d, strong)     | `all-MiniLM-L6-v2` (fast/light), `intfloat/multilingual-e5-base` |
| Reranker        | `BAAI/bge-reranker-base`           | turn off with `rerank.enabled: false`                      |
| VLM / QA        | `gemini-2.5-flash-lite` (cheapest) | `gemini-2.5-flash`, `gemini-2.5-pro`                       |

For E5 models, also set `query_prefix: "query: "` and
`passage_prefix: "passage: "`.

## Roadmap (maps to EXPECTATION.md)

- [x] **Phase 1 — text RAG MVP**: parsing, chunking, embeddings, FAISS, grounded QA
- [x] **Phase 1.5 — quality pass**: bge-m3 embeddings, bge-reranker-base (module 6.7), tightened grounding
- [ ] **Phase 2 — OCR** (PaddleOCR) over extracted figures/screenshots
- [ ] **Phase 2 — figure captioning** (Gemini) for diagrams/charts
- [ ] **Phase 3 — image embeddings** (CLIP/SigLIP) + hybrid score fusion
- [ ] **Phase 4 — evaluation** (Recall@K, MRR, QA accuracy) for RQ1–RQ3
- [ ] **Phase 4 — Gradio UI** and Colab notebook
```
