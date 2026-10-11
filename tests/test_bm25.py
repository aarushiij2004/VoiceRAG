from bm25 import BM25, tokenize

DOCS = [
    "a semaphore controls access to a shared resource",
    "paging avoids external fragmentation in memory",
    "a zombie process remains in the process table after it exits",
    "the process table stores process attributes",
]


def test_exact_term_ranks_first():
    bm = BM25(DOCS)
    assert bm.top_k("zombie", 3)[0][0] == 2


def test_stopwords_and_question_words_ignored():
    assert tokenize("What is the difference between a thread and a process?") == ["thread", "process"]


def test_plural_stemming_matches():
    bm = BM25(DOCS)
    assert bm.top_k("semaphores", 1)[0][0] == 0


def test_rare_terms_outweigh_common_terms():
    bm = BM25(DOCS)
    # 'process' is in 2 docs, 'zombie' in 1 -> zombie doc should win for "zombie process"
    assert bm.top_k("zombie process", 1)[0][0] == 2


def test_no_match_returns_nothing():
    bm = BM25(DOCS)
    assert bm.top_k("capital of France", 3) == []
    assert bm.top_k("", 3) == []


def test_idf_is_never_negative():
    bm = BM25(["the cat", "the cat", "the cat sat"])
    assert all(v >= 0 for v in bm.idf.values())


def test_empty_corpus():
    assert BM25([]).scores("anything") == []
