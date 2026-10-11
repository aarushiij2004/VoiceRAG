"""
ingest.py -- Build the FAISS index from documents in documents/.

    python ingest.py
    python ingest.py --chunk-size 150 --overlap 30

Run once before main.py, and again whenever documents or chunking settings change.
"""

import argparse
import json
import os
import tempfile

from config import (CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, DATA_DIR, DOCS_DIR,
                    EMBEDDING_MODEL, INDEX_PATH, META_PATH)
from documents import build_chunks, load_documents


def atomic_write_text(path, text: str) -> None:
    """Write to a temp file in the same folder, then rename: readers never see a half-written file."""
    path = os.fspath(path)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def embed_texts(model, texts: list):
    """
    Embed texts into L2-normalised float32 vectors.

    Normalising to unit length means inner product == cosine similarity, which
    is what the FAISS IndexFlatIP below ranks by. We normalise explicitly instead
    of relying on a model's default, so swapping embedding models stays safe.
    """
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=len(texts) > 50)
    return vectors.astype("float32")


def build_index(chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> int:
    import faiss
    from sentence_transformers import SentenceTransformer

    DATA_DIR.mkdir(exist_ok=True)
    docs = load_documents(DOCS_DIR)
    if not docs:
        raise SystemExit(
            f"No documents found in {DOCS_DIR}/. Add .txt, .md, or .pdf files and re-run."
        )
    print(f"Loaded {len(docs)} document(s): {[d[0] for d in docs]}")

    chunks = build_chunks(docs, chunk_size, overlap)
    print(f"Split into {len(chunks)} chunk(s) "
          f"({chunk_size} words, {overlap} overlap). Embedding with {EMBEDDING_MODEL}...")

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = embed_texts(model, [c["text"] for c in chunks])

    index = faiss.IndexFlatIP(vectors.shape[1])  # exact search, cosine via normalised vectors
    index.add(vectors)
    tmp_index = str(INDEX_PATH) + ".tmp"
    faiss.write_index(index, tmp_index)
    os.replace(tmp_index, INDEX_PATH)

    atomic_write_text(CHUNKS_PATH, json.dumps(chunks, ensure_ascii=False))
    atomic_write_text(META_PATH, json.dumps({
        "embedding_model": EMBEDDING_MODEL,
        "dim": int(vectors.shape[1]),
        "metric": "cosine (inner product on normalised vectors)",
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "num_chunks": len(chunks),
    }, indent=2))

    print(f"Saved index -> {INDEX_PATH}")
    print(f"Saved {len(chunks)} chunk records -> {CHUNKS_PATH}")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()
    build_index(args.chunk_size, args.overlap)
