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
  image_embeddings.py  # 6.5b CLIP cross-modal image/query embeddings (embed-the-image, H3)
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
  build_image_index.py # build the CLIP image index (--pack zips a Colab payload)
  probe_image_search.py# cross-modal probe: text query -> top slide images
notebooks/
  build_image_index.ipynb  # GPU (Colab) encode of slide images -> image index
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

**Results** (33 questions, top_k=5). Four first-stage conditions — a **BM25
lexical baseline** (B0) and the dense BGE-M3 retriever, each over text-only vs
text+caption chunks. `nDCG@5` carries a 95% bootstrap CI (1,000 resamples over
questions); **bold** marks the best per block:

_All questions (33):_

| Retriever      | Chunks       | rerank | R@1       | R@5       | MRR       | nDCG@5 (95% CI)            |
| -------------- | ------------ | ------ | --------- | --------- | --------- | ------------------------- |
| BM25 (B0)      | text-only    | —      | 0.545     | 0.788     | 0.643     | 0.637 [0.500, 0.767]      |
| BM25 (B0)      | text+caption | —      | 0.606     | 0.879     | 0.712     | 0.701 [0.583, 0.813]      |
| dense BGE-M3   | text-only    | on     | 0.576     | 0.818     | 0.683     | 0.680 [0.554, 0.800]      |
| dense BGE-M3   | text+caption | on     | **0.727** | **0.909** | **0.813** | **0.775 [0.660, 0.871]**  |
| dense BGE-M3   | text-only    | off    | 0.455     | 0.818     | 0.605     | 0.632 [0.498, 0.746]      |
| dense BGE-M3   | text+caption | off    | **0.788** | 0.909     | **0.838** | **0.780 [0.679, 0.875]**  |

_Figure-grounded subset (18):_

| Retriever      | Chunks       | rerank | R@1       | R@5       | MRR       | nDCG@5 (95% CI)            |
| -------------- | ------------ | ------ | --------- | --------- | --------- | ------------------------- |
| BM25 (B0)      | text-only    | —      | 0.611     | 0.833     | 0.696     | 0.713 [0.531, 0.878]      |
| BM25 (B0)      | text+caption | —      | 0.611     | 0.889     | 0.717     | 0.726 [0.566, 0.886]      |
| dense BGE-M3   | text-only    | on     | 0.556     | 0.833     | 0.669     | 0.689 [0.516, 0.844]      |
| dense BGE-M3   | text+caption | on     | **0.778** | **0.944** | **0.852** | **0.827 [0.700, 0.934]**  |

**Interpretation.** Flipping the regime flips the result. On text-sparse slides,
making figures retrievable via captions **lifts** retrieval *under both
retrievers* — BM25 nDCG@5 0.637→0.701, dense (rerank) 0.680→0.775 — with the gain
concentrated in the figure-grounded subset (dense R@1 0.556→0.778). The relevant
slide caption chunk reaches top-5 for 14/18 (rerank) — 16/18 (no rerank) figure
questions. This is the mirror image of the text-rich null above, so both **H1**
(multimodal beats text-only, concentrated in figure queries + text-sparse docs)
and **H2** (caption benefit is moderated by page text-density) are **supported**.
Two further reads: (i) the **caption signal matters more than the encoder** — even
lexical BM25 over caption chunks (nDCG@5 0.701) edges dense text-only (0.680/0.632),
so the win is the captions, not just the embeddings; (ii) the *text* cross-encoder
reranker **demotes** correct caption chunks (dense text+caption is best *without*
it: R@1 0.788 vs 0.727, 16/18 vs 14/18 chunk hits) — a cross-modal-aware reranker
is future work. **Significance caveat:** at n=33 the 95% CIs are wide and overlap
across conditions (e.g. dense text+caption nDCG@5 0.775 [0.660, 0.871] vs dense
text-only 0.680 [0.554, 0.800]) — the lift is **consistent across metrics and
retrievers but not statistically separated**; a larger gold set is needed to
tighten it ([PROPOSAL.md](PROPOSAL.md) §6.4, §7.2).

## Faithfulness — does the answer stay inside the evidence?

Retrieval recall asks whether the evidence is *found*; faithfulness asks whether
the answer stays *inside* it. `scripts/faithfulness.py` generates a grounded
answer for every gold question, then scores its `[pN]` citations against the
evidence the model was actually shown — **citation precision**, **leakage rate**
(answers citing a page never shown), and a page-level **citation recall** proxy:

```powershell
python scripts/faithfulness.py --gold data/eval/slides_qa.json
```

The metric immediately caught a **prompt bug, not a model failure**: the evidence
header used to be numbered `[Evidence 1 | … | p7]`, giving the model two integers
per passage, and the cheap QA model often cited the *evidence index* as if it were
a page (evidence 1–5 → `[p2, p3, p4, p5]`). Making the page the only citable number
in the header (`[source=… | p7]`) eliminates it:

| Evidence header              | Citation precision   | Leakage rate         | Leaked citations |
| ---------------------------- | -------------------- | -------------------- | ---------------- |
| `[Evidence i | … | pN]`       | 0.849 [0.735, 0.947] | 0.250 [0.100, 0.419] | 32 / 143         |
| `[source=… | pN]` (current)  | **1.000 [1.00, 1.00]** | **0.000 [0.00, 0.00]** | **0 / 138**    |

After the fix, **0 of 138 citations** leak across the gold set and citation recall
rises 0.78→0.90 (95% bootstrap CIs; 2/33 answers abstain when evidence is thin).
The takeaway: a faithfulness metric probes the *prompt* as much as the model
([PROPOSAL.md](PROPOSAL.md) §7.3).

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
- [x] **Phase 4b — eval hardening**: BM25 lexical baseline, nDCG@K, bootstrap CIs (added) — see [Evaluation](#evaluation-rq1--rq2)
- [x] **Faithfulness / citation precision-recall** over grounded answers — zero leakage after a prompt
      fix the metric exposed, see [Faithfulness](#faithfulness--does-the-answer-stay-inside-the-evidence)
- [x] **Phase 3a — cross-modal image embeddings** (CLIP): 252 slide images encoded into a shared
      text-image space; a text-query probe returns sane slide hits (the *embed-the-image* arm of H3).
      Build on GPU via [`notebooks/build_image_index.ipynb`](notebooks/build_image_index.ipynb) or
      locally with `scripts/build_image_index.py`; probe with `scripts/probe_image_search.py`
- [ ] **Phase 3b — RRF / distribution-aware fusion** of text + image rankings (RQ3 / H3), *not* a
      fixed-weight score blend, with full E3 eval + α sensitivity curve
- [ ] **Phase 4c — QA accuracy** (LLM-judge + human-κ validation) + Gradio UI + Colab notebook
```
