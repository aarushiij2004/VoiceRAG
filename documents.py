"""
documents.py -- Load documents and split them into overlapping, page-aware chunks.

Pure Python (no FAISS / embedding imports) so it is cheap to import and easy to test.
"""

from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR
from text_utils import clean_page_text, remove_repeated_lines


def read_txt(path: Path) -> list:
    """A text file is treated as a single 'page' (page number None)."""
    return [(None, path.read_text(encoding="utf-8", errors="ignore"))]


def read_pdf(path: Path) -> list:
    """Returns [(page_number, text), ...] with 1-based page numbers."""
    import logging

    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed-object warnings are noise
    reader = PdfReader(str(path))
    pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    pages = remove_repeated_lines(pages)  # drop per-page headers/footers/watermarks
    return [(n, clean_page_text(t)) for n, t in pages]


def load_documents(docs_dir: Path = DOCS_DIR) -> list:
    """Returns [(source_filename, [(page_number, text), ...]), ...]."""
    docs = []
    for path in sorted(Path(docs_dir).glob("**/*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            pages = read_txt(path)
        elif suffix == ".pdf":
            pages = read_pdf(path)
        else:
            continue
        if any(text.strip() for _, text in pages):
            docs.append((path.name, pages))
    return docs


def chunk_pages(pages: list, source: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list:
    """
    Sliding-window word chunking over a whole document, remembering which page
    each word came from so every chunk carries page_start / page_end.

    Chunking across page boundaries (instead of per page) keeps sentences that
    straddle a page break intact; the page metadata keeps citations accurate.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words, word_pages = [], []
    for page_no, text in pages:
        for w in text.split():
            words.append(w)
            word_pages.append(page_no)
    if not words:
        return []

    chunks, start, stride = [], 0, chunk_size - overlap
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_pages_ = [p for p in word_pages[start:end] if p is not None]
        chunks.append({
            "text": " ".join(words[start:end]),
            "source": source,
            "page_start": min(chunk_pages_) if chunk_pages_ else None,
            "page_end": max(chunk_pages_) if chunk_pages_ else None,
        })
        if end >= len(words):
            break
        start += stride

    return chunks


def build_chunks(docs: list, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Chunk every document and assign a global chunk_id (== row in the vector index)."""
    all_chunks = []
    for source, pages in docs:
        all_chunks.extend(chunk_pages(pages, source, chunk_size, overlap))
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i
    return all_chunks


def format_citation(chunk: dict) -> str:
    """'OS_Full_Notes.pdf p.36' or 'OS_Full_Notes.pdf pp.36-37' (or just the filename)."""
    start, end = chunk.get("page_start"), chunk.get("page_end")
    if start is None:
        return chunk["source"]
    if start == end:
        return f"{chunk['source']} p.{start}"
    return f"{chunk['source']} pp.{start}-{end}"
