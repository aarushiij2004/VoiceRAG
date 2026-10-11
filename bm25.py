"""
bm25.py -- A small, dependency-free BM25 (Okapi) keyword ranker.

Why have it next to dense embeddings? Embeddings capture *meaning* but can blur
exact technical terms ("TLB", "semaphore", "Belady's anomaly"). BM25 matches those
exact words. Combining the two (hybrid search) covers both failure modes.

    score(q, d) = sum over query terms t of
        IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
"""

import math
import re
from collections import Counter

# Question words and glue words carry no topical signal; they only add noise to BM25.
STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has have how
i if in into is it its me my of on or our so than that the their them then there these
they this those to was we were what when where which who whom why will with would you your
explain tell describe define mean means about between difference different
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Tiny plural/suffix normaliser so 'semaphores' matches 'semaphore'. Deliberately minimal."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, remove_stopwords: bool = True) -> list:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_stem(t) for t in tokens]


class BM25:
    def __init__(self, documents: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.n_docs = len(documents)
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0
        self.term_freqs = [Counter(t) for t in self.doc_tokens]

        df = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        # "+1" inside the log keeps IDF non-negative even for terms in most documents.
        self.idf = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }

    def scores(self, query: str) -> list:
        """BM25 score of the query against every document (0.0 when nothing matches)."""
        q_terms = tokenize(query)
        out = [0.0] * self.n_docs
        if not q_terms or not self.n_docs:
            return out
        for i, tf in enumerate(self.term_freqs):
            norm = self.k1 * (1 - self.b + self.b * self.doc_lens[i] / (self.avgdl or 1.0))
            s = 0.0
            for term in q_terms:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out

    def top_k(self, query: str, k: int) -> list:
        """[(doc_index, score), ...] best first, only documents with score > 0."""
        scored = sorted(enumerate(self.scores(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 0]
