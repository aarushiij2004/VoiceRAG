import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: shared words => higher cosine similarity."""
    DIM = 128

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **_):
        out = np.zeros((len(texts), self.DIM), dtype="float32")
        for i, text in enumerate(texts):
            for word in text.lower().split():
                word = word.strip(".,?!")
                out[i, zlib.crc32(word.encode()) % self.DIM] += 1.0  # stable across runs (unlike hash())
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms == 0, 1, norms)
        return out


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def sample_chunks():
    texts = [
        "A semaphore is a synchronization primitive used to control access by threads",
        "Paging divides memory into fixed size pages and avoids external fragmentation",
        "A zombie process has finished execution but still has an entry in the process table",
        "Deadlock occurs when processes wait forever for resources held by each other",
    ]
    return [{"text": t, "source": "notes.txt", "page_start": i + 1, "page_end": i + 1, "chunk_id": i}
            for i, t in enumerate(texts)]
