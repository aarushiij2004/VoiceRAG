# VoiceRAG — Voice AI Assistant

A voice assistant that answers spoken or typed questions grounded in your own documents
(PDF, TXT, Markdown) using retrieval-augmented generation (RAG). It cites page numbers,
combines semantic and keyword retrieval, refuses out-of-scope questions, remembers
follow-ups, streams answers sentence by sentence, and ships with a measured evaluation.

**Pipeline:** mic → faster-whisper (STT) → *(rewrite follow-up)* → hybrid retrieval
(FAISS dense + BM25, fused with RRF) → relevance gate → Gemini (streamed, grounded) →
sentence-by-sentence TTS.

## Demo
[![Watch the demo]](https://youtu.be/elWw2ebRFqg)

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
```

