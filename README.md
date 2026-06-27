# Research-Oriented Multimodal RAG

Multimodal Retrieval-Augmented Generation over academic documents and lecture
slides. See [PROPOSAL.md](PROPOSAL.md) for the full research proposal (theory,
formal problem statement, hypotheses, and evaluation protocol).

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
descriptive prose. This null is exactly what hypothesis **H2** predicts for a
text-rich document (caption benefit is *moderated by page text-sparsity*); the
decisive test is **text-sparse lecture slides** (below).

### Decisive test — text-sparse lecture slides (H1 / H2)

The slide gold set (`data/eval/slides_qa.json`, **33 QA pairs**, 18 figure / 15
text) is scored against the multi-deck index. Because page numbers collide across
the 14 decks, scoring is **source-aware** — a hit counts only when both its deck
(`source`) and page match the gold:

```powershell
python scripts/ingest_corpus.py                            # build the multi-deck slide index
python scripts/evaluate.py --gold data/eval/slides_qa.json # add --no-rerank for dense-only
```

**Results** (33 questions, top_k=5):

| Subset       | Config       | rerank | R@1   | R@3   | R@5   | MRR   |
| ------------ | ------------ | ------ | ----- | ----- | ----- | ----- |
| all (33)     | text-only    | on     | 0.576 | 0.788 | 0.818 | 0.683 |
| all (33)     | text+caption | on     | **0.727** | **0.909** | **0.909** | **0.813** |
| all (33)     | text-only    | off    | 0.500 | 0.778 | 0.889 | 0.655 |
| all (33)     | text+caption | off    | **0.833** | **0.944** | **0.944** | **0.880** |
| figure (18)  | text-only    | on     | 0.556 | 0.778 | 0.833 | 0.669 |
| figure (18)  | text+caption | on     | **0.778** | **0.944** | **0.944** | **0.852** |

**Interpretation.** Flipping the regime flips the result. On text-sparse slides,
making figures retrievable via captions **lifts** retrieval — overall R@1
0.576→0.727 (MRR 0.683→0.813), with the gain concentrated in the figure-grounded
subset (R@1 0.556→0.778). The relevant slide caption chunk reaches top-5 for
14/18 figure questions. This is the mirror image of the text-rich null above, so
both **H1** (multimodal beats text-only, concentrated in figure queries +
text-sparse docs) and **H2** (caption benefit is moderated by page text-density)
are **supported**. One incidental finding: the *text* cross-encoder reranker
slightly demotes correct caption chunks (text+caption R@1 is higher *without* it,
0.833 vs 0.727) — a cross-modal-aware reranker is future work. Caveat: n=33, one
course corpus, page-level relevance — **indicative, not significant** until BM25
baselines and bootstrap CIs land ([PROPOSAL.md](PROPOSAL.md) §6.4, §7.2).

## Roadmap (maps to [PROPOSAL.md](PROPOSAL.md) §10)

- [x] **Phase 1 — text RAG MVP**: parsing, chunking, embeddings, FAISS, grounded QA
- [x] **Phase 1.5 — quality pass**: bge-m3 embeddings, bge-reranker-base, tightened grounding
- [x] **Phase 2 — OCR** (PaddleOCR) over extracted figures/screenshots
- [x] **Phase 2 — figure captioning** (Gemini VLM): describe-then-embed figure chunks (condition E2)
- [x] **Phase 4a — retrieval evaluation** (Recall@K, MRR) for RQ1/RQ2 — see [Evaluation](#evaluation-rq1--rq2)
- [x] **Dataset — text-sparse lecture slides ingested**: 14 decks, whole-slide render + VLM caption
      (one shared index, 1,226 chunks; 4 eval decks captioned, rest text-only distractors)
- [x] **Gold QA over the slide decks** (33 pairs) + **source-aware slide retrieval eval** — H1/H2
      both supported (captions lift R@1 0.576→0.727; figure subset 0.556→0.778), see [Evaluation](#evaluation-rq1--rq2)
- [ ] **Phase 4b — eval hardening**: BM25 baseline, nDCG, bootstrap CIs, faithfulness / citation precision-recall
- [ ] **Phase 3 — image embeddings** (CLIP/SigLIP) + **RRF / distribution-aware fusion** (RQ3 / H3),
      *not* a fixed-weight score blend
- [ ] **Phase 4c — QA accuracy** (LLM-judge + human-κ validation) + Gradio UI + Colab notebook
```
