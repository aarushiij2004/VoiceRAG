"""
retrieve.py -- Load the FAISS index built by ingest.py and retrieve
the top-k most relevant chunks for a query.
"""

import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "No index found. Run `python ingest.py` first to build one."
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def query(self, question: str, top_k: int = 4):
        query_vec = self.model.encode([question], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "distance": float(dist)})
        return results
