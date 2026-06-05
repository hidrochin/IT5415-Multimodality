# Research-Oriented Multimodal RAG

Multimodal Retrieval-Augmented Generation over academic documents and lecture
slides. See [EXPECTATION.md](EXPECTATION.md) for the full research proposal.

This repo is built **incrementally**. Phase 1 (this MVP) is a working
text-RAG vertical slice; later phases add OCR, figure captioning, image
embeddings, reranking, evaluation, and a Gradio UI.

```
parse (PyMuPDF) ─┬─ text chunks ───────────────────────────────┐
                 └─ figures ─> OCR (PaddleOCR) + caption (Gemini) ┤
                                                                  ├─> embed -> FAISS
query -> embed -> FAISS -> rerank (bge) -> Gemini grounded answer ┘
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

> **PaddleOCR note (Windows):** `requirements.txt` pins `paddlepaddle==3.0.0`.
> The 3.3.x CPU build has a PIR/oneDNN executor regression that crashes OCR on
> Windows. If you don't need figure OCR, set `ocr.enabled: false` in
> `config.yaml` and the rest of the pipeline runs without PaddlePaddle.
> Captioning needs only a Gemini key; turn it off with `figures.enabled: false`.

## Project layout

```
config.yaml            # all knobs: models, chunk sizes, top_k, paths
src/mmrag/
  config.py            # load config.yaml + .env
  parsing.py           # 6.1 PyMuPDF text + image extraction
  ocr.py               # 6.2 PaddleOCR over extracted figures
  figures.py           # 6.3 Gemini figure captioning (VLM)
  chunking.py          # 6.4 multimodal chunk structure (text + figure chunks)
  embeddings.py        # 6.5 text embeddings (normalized for cosine)
  retrieval.py         # 6.6 FAISS dense retrieval
  rerank.py            # 6.7 bge-reranker cross-encoder
  qa.py                # 6.8 Gemini grounded QA
  evaluation.py        # 9   Recall@K, MRR, in-memory index builder
  pipeline.py          # ingest() + ask() orchestration
scripts/
  fetch_sample.py      # download a sample academic PDF
  ingest.py            # build the index from a PDF
  ask.py               # query the index
  evaluate.py          # text-only vs text+caption retrieval metrics
data/raw/              # input PDFs / slides
data/processed/        # extracted images + FAISS index
data/eval/             # hand-authored gold QA sets
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

## Evaluation (RQ1 / RQ2)

Build the index, then score retrieval text-only vs text+figure-caption against a
hand-authored gold set (`data/eval/attention_qa.json`, 13 QA pairs):

```powershell
python scripts/ingest.py "data/raw/attention_is_all_you_need.pdf"
python scripts/evaluate.py            # add --no-rerank for first-stage dense only
```

**Results on _Attention Is All You Need_** (13 questions, top_k=5):

| Config       | R@1   | R@3   | R@5   | MRR   |
| ------------ | ----- | ----- | ----- | ----- |
| text-only    | 0.692 | 0.692 | 0.923 | 0.746 |
| text+caption | 0.692 | 0.692 | 0.923 | 0.746 |

**Interpretation.** On this paper, captions don't move *page-level* recall:
figure pages (p3/p4) are text-rich, so text-only already retrieves them. But the
caption chunks **are** surfaced — a figure chunk appears in the top-5 for 2/2
figure-grounded questions — they're simply redundant when the page also has
descriptive prose. The payoff for captions should show on **text-sparse
documents (lecture slides)**, which is the next dataset to add (EXPECTATION.md §7).
Sample size is small; treat as a sanity baseline, not a conclusion.

## Roadmap (maps to EXPECTATION.md)

- [x] **Phase 1 — text RAG MVP**: parsing, chunking, embeddings, FAISS, grounded QA
- [x] **Phase 1.5 — quality pass**: bge-m3 embeddings, bge-reranker-base (module 6.7), tightened grounding
- [x] **Phase 2 — OCR** (PaddleOCR, module 6.2) over extracted figures/screenshots
- [x] **Phase 2 — figure captioning** (Gemini VLM, module 6.3); figures become searchable chunks (enables Experiment 2)
- [x] **Phase 4 — retrieval evaluation** (Recall@K, MRR; module 9) for RQ1/RQ2 — see [Evaluation](#evaluation-rq1--rq2)
- [ ] **Phase 3 — image embeddings** (CLIP/SigLIP) + hybrid score fusion (RQ3)
- [ ] **Phase 4 — QA accuracy** (Gemini-judged) + lecture-slide gold set
- [ ] **Phase 4 — Gradio UI** and Colab notebook
```
