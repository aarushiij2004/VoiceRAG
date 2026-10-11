"""
config.py -- Single place for every tunable setting.

Everything can be overridden with an environment variable (or a line in .env),
so experiments don't require editing code.
"""

import os
from pathlib import Path

DOCS_DIR = Path(os.getenv("DOCS_DIR", "documents"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.json"   # JSON, not pickle: safe to load, human-readable
META_PATH = DATA_DIR / "index_meta.json"  # records how the index was built

# --- Indexing -------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))      # words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))  # words shared by neighbours

# --- Retrieval ------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # dense | bm25 | hybrid
# Minimum cosine similarity of the best dense match for a question to count as
# "in the documents". NOT yet calibrated for your corpus -- run
# `python eval.py --calibrate` and set this from the printed sweep.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))
RRF_K = 60  # standard constant for Reciprocal Rank Fusion

# --- Generation -----------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Reasoning effort for the answer call. "low" cuts latency, which matters for voice.
# Set to an empty string to use the model's default.
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low")
# Generous on purpose: reasoning-capable models can spend part of this budget on thinking.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "3"))  # past Q&A pairs used for follow-ups

# --- Speech ---------------------------------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
TTS_RATE = int(os.getenv("TTS_RATE", "175"))

NO_ANSWER_MESSAGE = "I couldn't find anything about that in the documents."
