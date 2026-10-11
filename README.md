# VoiceRAG — Voice AI Assistant

A voice assistant that answers spoken or typed questions grounded in your own documents
(PDF, TXT, Markdown) using retrieval-augmented generation (RAG). It cites page numbers,
combines semantic and keyword retrieval, refuses out-of-scope questions, remembers
follow-ups, streams answers sentence by sentence, and ships with a measured evaluation.

**Pipeline:** mic → faster-whisper (STT) → *(rewrite follow-up)* → hybrid retrieval
(FAISS dense + BM25, fused with RRF) → relevance gate → Gemini (streamed, grounded) →
sentence-by-sentence TTS.

## Demo
[![Watch the demo](https://img.shields.io/badge/▶-Watch%20Demo-red)](https://youtu.be/elWw2ebRFqg)

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt  # add requirements-ui.txt for the web UI
```

Linux only — TTS needs a system engine: `sudo apt install espeak-ng`

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (free key at
[aistudio.google.com](https://aistudio.google.com)). Put documents in `documents/`, then:

```bash
python ingest.py                 # build the index (rerun after changing documents)
python main.py --text            # type questions
python main.py                   # voice mode: Enter to record, Enter to stop
streamlit run app.py             # web UI (pip install -r requirements-ui.txt)
```

> If you upgraded from an earlier version, delete old files in `data/` and re-run
> `python ingest.py` (the index format changed).

CLI flags: `--mode dense|bm25|hybrid`, `--top-k N`, `--no-stream`, `--no-memory`,
`--no-tts`, `--timings` (latency per stage), `--debug` (retrieval scores).
Say or type `reset` to clear conversation memory.

### Web UI
Upload documents and rebuild the index from the sidebar, ask by typing or recording,
watch the answer stream in, and open **Retrieved passages** under any answer to see
exactly which chunks were used and their similarity / BM25 scores.

## How it works

| Stage | Choice | Why |
|---|---|---|
| Cleaning | strip control chars, drop repeated headers/footers | The sample PDF has garbled glyphs and a watermark on 38 of 41 pages |
| Chunking | 200 words, 40 overlap, page-aware | Precise retrieval; overlap protects answers that straddle a boundary; page metadata powers citations |
| Embeddings | `all-MiniLM-L6-v2`, normalised | Small, fast on CPU, local |
| Index | FAISS `IndexFlatIP` on unit vectors | Exact search; inner product = cosine. At this scale approximate indexes would only lose accuracy |
| Keyword search | own BM25 (`bm25.py`) | Exact technical terms ("TLB", "semaphore") that embeddings can blur |
| Fusion | Reciprocal Rank Fusion (k=60) | Combines ranks, so BM25 and cosine scores need no shared scale |
| Gate | best dense cosine ≥ `MIN_SIMILARITY` | Off-topic questions get a refusal; the LLM is never called |
| Memory | LLM rewrites follow-ups into standalone questions | "How do I prevent it?" becomes searchable; one extra call only on follow-ups |
| Generation | Gemini via `google-genai`, grounded system prompt | Context-only, 2–4 spoken-style sentences; excerpts treated as untrusted data (prompt-injection guard) |
| Output | streamed, spoken sentence by sentence | First audio starts before the full answer exists |
| STT | faster-whisper `base.en`, int8, VAD filter | Fast on CPU; audio read with `soundfile` so no PyAV decoder dependency |
| Robustness | index/model consistency checks, atomic file writes, friendly LLM errors | A stale or half-written index fails loudly with the fix, not silently wrong |

### Gemini notes
- Default model: `gemini-3.6-flash` (generally available when this was written; the newer
  `gemini-3.8-flash` also works — set `GEMINI_MODEL`). Use any model your key can access.
- Sampling parameters (`temperature`, `top_p`, `top_k`) are intentionally **not** sent:
  Google's Gemini 3 migration notes list them as removed.
- `GEMINI_THINKING_LEVEL=low` by default for lower latency; if a model rejects it the
  call is retried without it. Set it empty to use the model default.

## Evaluation

`eval.py` scores retrieval on 46 in-scope questions (12 deliberately paraphrased to avoid
the notes' keywords) and 8 out-of-scope questions. A retrieved chunk is a *hit* if it
contains the question's evidence phrase (a unit test validates every phrase against the PDF).

```bash
python eval.py --modes bm25 dense hybrid --chunk-sizes 100 200 400 --show-misses
python eval.py --modes dense --calibrate     # choose MIN_SIMILARITY from data
```

**Keyword-only (BM25) baseline — measured on `OS_Full_Notes.pdf`:**

| chunk size | hit@1 | hit@3 | hit@5 | MRR | paraphrased hit@3 |
|---|---|---|---|---|---|
| 100 words | 50.0% | 76.1% | 80.4% | 0.637 | 66.7% |
| 200 words | 69.6% | 84.8% | 89.1% | 0.780 | 66.7% |
| 400 words | 67.4% | 89.1% | 95.7% | 0.794 | 77.8% |

Dense and hybrid results are produced by the command above once the embedding model is
available — paste them here after you run it. Findings so far: 100-word chunks lose
context; BM25 alone cannot match "microkernel" to the notes' "Micro Kernel"; and its
out-of-scope refusal rate is only 25% (any shared word counts), which is why the gate uses
embedding similarity. These are retrieval metrics, not end-to-end answer accuracy — review
generated answers manually too. The questions were written against one document; add your own.

## Project structure

```
├── documents/        # source documents
├── data/             # generated by ingest.py (faiss.index, chunks.json, index_meta.json)
├── eval/questions.json
├── config.py         # every tunable setting, env-overridable
├── documents.py      # load PDFs/text, clean, page-aware chunking
├── text_utils.py     # control-char/footer cleaning, match normalisation
├── ingest.py         # embed chunks, build FAISS index (atomic writes)
├── bm25.py           # keyword ranker
├── retrieve.py       # dense / bm25 / hybrid retrieval, RRF, relevance gate, index checks
├── llm.py            # Gemini answers (streaming) + follow-up rewriting
├── sentences.py      # stream -> sentences for incremental TTS
├── tts.py            # pyttsx3 on a worker thread
├── stt.py / audio_utils.py
├── pipeline.py       # the RAG loop, I/O-free and unit-tested
├── main.py           # CLI
├── app.py            # Streamlit UI
├── api.py            # optional FastAPI server
├── eval.py           # retrieval evaluation + threshold calibration
└── tests/            # pytest suite (88 tests)
```

## Optional HTTP API

```bash
pip install -r requirements-api.txt
uvicorn api:app --reload
# POST /ask {"question": "..."}   POST /ask-audio (16 kHz mono WAV)   GET /health
```
Stateless: no conversation memory; answers returned as JSON.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
Covers chunking and cleaning, BM25, RRF, dense retrieval and the gate (real FAISS with a
deterministic fake embedder), sentence streaming, the pipeline, the Gemini wrapper (fake
client), audio loading, index checks, atomic writes, evaluation metrics, the API, the
Streamlit app, and an end-to-end run over the real PDF with only the Gemini call faked.
The embedding model, real Gemini calls, microphone and speaker are not exercised by tests.

## Privacy and security

Text extraction, embeddings, retrieval and speech-to-text run locally. For answer
generation, the question and retrieved passages are sent to the Gemini API — do not index
confidential documents unless that use is approved. Keep `.env` out of version control.
Index files are JSON + FAISS (no pickle), but only load index files you created yourself.
Retrieved text is treated as untrusted data in the prompt, which reduces (but does not
eliminate) prompt-injection risk.

## Known limitations / next steps

- `MIN_SIMILARITY=0.25` is a starting guess — calibrate with `eval.py --calibrate`.
- Scanned PDFs need OCR (not included); they are skipped.
- No cross-encoder reranker yet (the natural next step after hybrid search).
- Push-to-talk only; the browser UI cannot play `pyttsx3` audio (it plays on the server machine).
- `pyttsx3` is offline but robotic; neural TTS would sound better.
- Single-process, single-user design; BM25 is plain Python (fine for thousands of chunks, not millions).
- Word-based chunking ignores headings/sentences; structure-aware chunking is a possible upgrade.
