"""
ingest.py -- Build a FAISS index from documents in the documents/ folder.

Run this once before main.py, and again any time you add or change documents.

    python ingest.py
"""

import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

DOCS_DIR = Path("documents")
DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 200       # words per chunk
CHUNK_OVERLAP = 40     # words of overlap between consecutive chunks


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents():
    """Returns a list of (source_filename, full_text) tuples."""
    docs = []
    for path in sorted(DOCS_DIR.glob("**/*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            text = read_txt(path)
        elif suffix == ".pdf":
            text = read_pdf(path)
        else:
            continue
        if text.strip():
            docs.append((path.name, text))
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple word-based sliding-window chunking with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_index():
    DATA_DIR.mkdir(exist_ok=True)
    docs = load_documents()
    if not docs:
        raise SystemExit(
            f"No documents found in {DOCS_DIR}/. Add .txt, .md, or .pdf files and re-run."
        )

    print(f"Loaded {len(docs)} document(s): {[d[0] for d in docs]}")

    all_chunks = []  # list of {"text": ..., "source": ...}
    for filename, text in docs:
        for chunk in chunk_text(text):
            all_chunks.append({"text": chunk, "source": filename})

    print(f"Split into {len(all_chunks)} chunk(s). Embedding with {EMBEDDING_MODEL}...")

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Saved index -> {INDEX_PATH}")
    print(f"Saved {len(all_chunks)} chunk records -> {CHUNKS_PATH}")


if __name__ == "__main__":
    build_index()
