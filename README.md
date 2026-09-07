# Voice RAG Assistant

A voice assistant that answers spoken questions grounded in your own documents,
using retrieval-augmented generation (RAG) instead of just calling an LLM
directly.

**Pipeline:** mic input -> Whisper transcription -> FAISS retrieval over your
documents -> Gemini answers using only the retrieved context -> text-to-speech
output.

Uses Google's **Gemini API free tier** for the LLM call -- no credit card
required.

## Why RAG and not just STT -> LLM -> TTS?

A voice wrapper around a raw LLM call demonstrates API-chaining, not AI
engineering. The retrieval step -- chunking documents, embedding them, indexing
them, and grounding the LLM's answer in retrieved content -- is what actually
shows you understand how RAG systems work. Don't skip it to save time.

## Setup

1. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Linux only:** `pyttsx3` needs a system TTS engine.

   ```bash
   sudo apt install espeak-ng
   ```

3. **Get a free Gemini API key**

   - Go to https://aistudio.google.com and sign in with any Google account
     (no card, no phone verification).
   - Click **Get API Key** in the left nav, then **Create API key**.
   - Copy the key.

4. **Set your API key**

   ```bash
   cp .env.example .env        # Windows (Git Bash/PowerShell): copy .env.example .env
   ```

   Open `.env` and paste your key:

   ```
   GEMINI_API_KEY=your-actual-key-here
   ```

5. **Add your documents**

   Drop `.txt`, `.md`, or `.pdf` files into `documents/`. A sample file is
   included so the pipeline works out of the box -- replace it with your
   resume, project write-ups, class notes, or a product manual for a real demo.

6. **Build the index**

   ```bash
   python ingest.py
   ```

   Re-run this any time you add or change documents.

7. **Run it**

   ```bash
   python main.py            # voice mode -- needs a microphone
   python main.py --text     # type questions instead, no mic needed
   ```

   In voice mode: press Enter to start recording, ask your question out loud,
   press Enter again to stop. The transcription, retrieved sources, and
   spoken answer all print to the terminal.

## About the free tier

Google's Gemini API free tier (Flash models) is a genuinely standing free
allowance, not a one-time trial credit -- no card needed at all. Two things
worth knowing:

- **Rate limits**: expect something in the range of ~15 requests/minute and
  ~1,000-1,500 requests/day on the free Flash tier. Way more than enough for
  building and demoing this project.
- **Data usage**: Google's free tier terms allow prompts/outputs to be used to
  improve their products (this is different from the paid tier). Fine for a
  portfolio project with non-sensitive sample documents; worth knowing if you
  ever point this at real private data.
- If `gemini-2.5-flash` in `llm.py` throws a "model not found" error, Google
  renames/rotates free-tier model names periodically -- open
  https://aistudio.google.com, check which Flash model is currently listed,
  and update the `MODEL` constant in `llm.py` to match.

## Project structure

```
voice-rag-assistant/
├── documents/       # your source documents go here
├── data/            # generated: faiss.index + chunks.pkl (from ingest.py)
├── ingest.py        # chunk documents, embed, build FAISS index
├── retrieve.py      # load index, retrieve top-k chunks for a query
├── stt.py           # faster-whisper transcription
├── llm.py           # Gemini call, grounded in retrieved chunks
├── tts.py           # pyttsx3 text-to-speech
├── audio_utils.py   # microphone recording
├── main.py          # CLI loop tying it all together
└── requirements.txt
```

## Notes on design choices

- **CLI, not a web UI.** The point of this project is the retrieval + grounding
  pipeline, not frontend work. A UI adds build time without adding AI-engineering
  signal.
- **Manual start/stop recording, no VAD.** Auto-detecting when you've stopped
  talking is a real feature but is fiddly to get right fast. A reliable
  press-Enter baseline demos better than a flaky auto-stop.
- **`faster-whisper` on CPU with `int8` compute** so it runs without a GPU.
- **`IndexFlatL2`** (exact search) rather than an approximate index -- for a
  small personal document set, exact search is fast enough and one less thing
  to tune.

## Extending it (stretch / reach goals, in rough order of effort)

- **Streaming responses**: stream the answer token-by-token instead of
  waiting for the full response.
- **Voice activity detection (VAD)**: auto-detect when the user stops talking
  instead of requiring a keypress (e.g. `webrtcvad` or `silero-vad`).
- **Interrupt handling**: let the user talk over the assistant's TTS output.
- **Fully offline mode**: swap `faster-whisper` for `whisper.cpp`, Gemini for a
  local model via Ollama, and `pyttsx3` for Piper -- lets you demo it running
  with zero API calls and zero internet.
- **Better chunking**: sentence- or paragraph-aware chunking instead of a
  fixed word-count sliding window, especially if your documents have structure
  (headings, sections) worth preserving.

## Troubleshooting

- **`RuntimeError: No audio captured`**: check your OS's microphone
  permissions and that `sounddevice` sees an input device
  (`python -c "import sounddevice; print(sounddevice.query_devices())"`).
- **No sound from TTS on Linux**: confirm `espeak-ng` is installed and that
  `aplay`/`pulseaudio` is working.
- **`404` / model not found from Gemini**: see "About the free tier" above --
  update the model name in `llm.py`.
- **`429` rate limit error from Gemini**: you've hit the free-tier request
  limit; wait a minute (per-minute limit) or a day (daily limit) and retry.
- **First run is slow**: `faster-whisper` and `sentence-transformers` download
  model weights the first time they're used -- this needs an internet
  connection once, then works offline (except for the Gemini API call itself).
