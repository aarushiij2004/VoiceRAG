# VoiceRAG — Voice AI Assistant

A voice assistant that answers spoken questions grounded in your own documents
using retrieval-augmented generation (RAG).

**Pipeline:** mic input → Whisper transcription → FAISS retrieval over your
documents → Gemini answers using only the retrieved context → text-to-speech
output.

## Demo
[![Watch the demo](https://img.shields.io/badge/▶-Watch%20Demo-red)](https://youtu.be/elWw2ebRFqg)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Linux only — TTS needs a system engine: `sudo apt install espeak-ng`

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com)
(no card required), then:

```bash
cp .env.example .env
# paste your key into .env as GEMINI_API_KEY=...
```

Add your documents (`.txt`, `.md`, `.pdf`) to `documents/`, then:

```bash
python ingest.py          # builds the FAISS index — rerun after changing documents
python main.py --text     # type questions, no mic needed
python main.py            # voice mode — press Enter to record, Enter again to stop
```

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
ini API call itself).
