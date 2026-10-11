"""
retrieve.py -- Retrieval: dense (FAISS), keyword (BM25), or hybrid (both, fused with RRF).

Also decides whether the question is answerable from the documents at all
(`relevant`), using the best dense cosine similarity as a gate.
"""

import json
from dataclasses import dataclass, field

import numpy as np

from bm25 import BM25
from config import (CHUNKS_PATH, EMBEDDING_MODEL, INDEX_PATH, META_PATH, MIN_SIMILARITY,
                    RETRIEVAL_MODE, RRF_K, TOP_K)

MODES = ("dense", "bm25", "hybrid")


def reciprocal_rank_fusion(rankings: list, k: int = RRF_K) -> list:
    """
    Fuse several ranked lists of document ids into one.

        RRF score(d) = sum over lists of 1 / (k + rank(d))      (rank starts at 1)

    Uses only *ranks*, never raw scores, so it needs no normalisation between
    BM25 scores and cosine similarities (which live on totally different scales).
    Returns [(doc_id, fused_score), ...] best first.
    """
    fused = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def check_index_consistency(n_vectors: int, chunks: list, meta: dict = None,
                            embedding_model: str = EMBEDDING_MODEL) -> None:
    """
    Fail loudly (with the fix) if the saved index can't be trusted:
      * vector count != chunk count -> a row would map to the wrong text
      * index built with a different embedding model -> query vectors aren't comparable
    """
    if n_vectors != len(chunks):
        raise RuntimeError(
            f"Index has {n_vectors} vectors but chunks file has {len(chunks)} records. "
            "They are out of sync -- re-run `python ingest.py`.")
    built_with = (meta or {}).get("embedding_model")
    if built_with and built_with != embedding_model:
        raise RuntimeError(
            f"Index was built with '{built_with}' but EMBEDDING_MODEL is '{embedding_model}'. "
            "Use the same model or re-run `python ingest.py`.")


@dataclass
class RetrievalResult:
    chunks: list = field(default_factory=list)  # chunk dicts + 'similarity'/'bm25' where known
    top_similarity: float = None                # best dense cosine similarity (None for bm25 mode)
    relevant: bool = True                       # False -> question looks out-of-scope


class Retriever:
    def __init__(self, chunks: list, mode: str = RETRIEVAL_MODE, index=None, embedder=None,
                 min_similarity: float = MIN_SIMILARITY, pool_size: int = 20):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode in ("dense", "hybrid") and (index is None or embedder is None):
            raise ValueError(f"mode '{mode}' needs both an index and an embedder")
        self.chunks = chunks
        self.mode = mode
        self.index = index
        self.embedder = embedder
        self.min_similarity = min_similarity
        self.pool_size = pool_size
        self.bm25 = BM25([c["text"] for c in chunks]) if mode in ("bm25", "hybrid") else None

    # ---- constructors ----------------------------------------------------
    @classmethod
    def from_disk(cls, mode: str = RETRIEVAL_MODE, **kwargs):
        """Load the index built by ingest.py. Loads the embedding model once."""
        needs_vectors = mode in ("dense", "hybrid")
        if not CHUNKS_PATH.exists() or (needs_vectors and not INDEX_PATH.exists()):
            raise FileNotFoundError("Index not found. Add documents and run `python ingest.py` first.")
        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        index = embedder = None
        if needs_vectors:
            import faiss
            from sentence_transformers import SentenceTransformer

            index = faiss.read_index(str(INDEX_PATH))
            meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else None
            check_index_consistency(index.ntotal, chunks, meta)
            embedder = SentenceTransformer(EMBEDDING_MODEL)
        return cls(chunks, mode=mode, index=index, embedder=embedder, **kwargs)

    @classmethod
    def build_in_memory(cls, chunks: list, embedder, mode: str = "hybrid", **kwargs):
        """Build an exact cosine index in RAM -- used by eval.py to try chunk sizes quickly."""
        index = None
        if mode in ("dense", "hybrid"):
            import faiss

            vectors = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True,
                                      normalize_embeddings=True).astype("float32")
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
        return cls(chunks, mode=mode, index=index, embedder=embedder, **kwargs)

    # ---- search ----------------------------------------------------------
    def _dense(self, question: str, n: int) -> list:
        q = self.embedder.encode([question], convert_to_numpy=True,
                                 normalize_embeddings=True).astype("float32")
        sims, ids = self.index.search(q, min(n, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i != -1]

    def query(self, question: str, top_k: int = TOP_K) -> RetrievalResult:
        dense_hits = self._dense(question, self.pool_size) if self.mode != "bm25" else []
        bm25_hits = self.bm25.top_k(question, self.pool_size) if self.mode != "dense" else []
        dense_sim = dict(dense_hits)
        bm25_score = dict(bm25_hits)

        if self.mode == "dense":
            order = [i for i, _ in dense_hits]
        elif self.mode == "bm25":
            order = [i for i, _ in bm25_hits]
        else:
            fused = reciprocal_rank_fusion([[i for i, _ in dense_hits], [i for i, _ in bm25_hits]])
            order = [i for i, _ in fused]

        results = []
        for rank, i in enumerate(order[:top_k], start=1):
            chunk = dict(self.chunks[i])
            chunk["rank"] = rank
            if i in dense_sim:
                chunk["similarity"] = dense_sim[i]
            if i in bm25_score:
                chunk["bm25"] = bm25_score[i]
            results.append(chunk)

        top_similarity = dense_hits[0][1] if dense_hits else None
        if top_similarity is not None:
            relevant = top_similarity >= self.min_similarity
        else:  # bm25-only: relevant if at least one query term matched anything
            relevant = bool(bm25_hits)
        return RetrievalResult(chunks=results, top_similarity=top_similarity, relevant=relevant)
