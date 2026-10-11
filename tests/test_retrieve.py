import faiss
import pytest

from retrieve import Retriever, reciprocal_rank_fusion


def make_retriever(chunks, embedder, mode, **kw):
    return Retriever.build_in_memory(chunks, embedder, mode=mode, **kw)


def test_rrf_rewards_agreement_between_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    ids = [d for d, _ in fused]
    assert ids[0] == 1               # ranked 1st and 2nd -> best combined
    assert set(ids) == {1, 2, 3, 4}  # union of both lists


def test_rrf_formula():
    fused = dict(reciprocal_rank_fusion([[7]], k=60))
    assert fused[7] == pytest.approx(1 / 61)


def test_dense_retrieval_finds_the_right_chunk(sample_chunks, embedder):
    r = make_retriever(sample_chunks, embedder, "dense", min_similarity=0.1)
    res = r.query("zombie process still has an entry in the process table", top_k=2)
    assert res.chunks[0]["chunk_id"] == 2
    assert res.chunks[0]["similarity"] > 0.5
    assert res.relevant


def test_similarity_is_cosine_in_unit_range(sample_chunks, embedder):
    r = make_retriever(sample_chunks, embedder, "dense")
    res = r.query("deadlock processes wait", top_k=4)
    assert all(-1.0001 <= c["similarity"] <= 1.0001 for c in res.chunks)
    sims = [c["similarity"] for c in res.chunks]
    assert sims == sorted(sims, reverse=True)


def test_gate_refuses_out_of_scope(sample_chunks, embedder):
    r = make_retriever(sample_chunks, embedder, "dense", min_similarity=0.3)
    res = r.query("recipe for spicy tomato pasta", top_k=3)
    assert res.relevant is False
    assert res.top_similarity < 0.3


def test_gate_threshold_is_respected(sample_chunks, embedder):
    q = "semaphore threads access"
    strict = make_retriever(sample_chunks, embedder, "dense", min_similarity=0.99).query(q)
    loose = make_retriever(sample_chunks, embedder, "dense", min_similarity=0.0).query(q)
    assert strict.relevant is False and loose.relevant is True


def test_bm25_mode_needs_no_embedder(sample_chunks):
    r = Retriever(sample_chunks, mode="bm25")
    assert r.query("paging fragmentation").chunks[0]["chunk_id"] == 1
    assert r.query("quantum chromodynamics").relevant is False


def test_hybrid_returns_both_signals(sample_chunks, embedder):
    r = make_retriever(sample_chunks, embedder, "hybrid", min_similarity=0.1)
    res = r.query("semaphore", top_k=2)
    top = res.chunks[0]
    assert top["chunk_id"] == 0 and "similarity" in top and "bm25" in top


def test_hybrid_rescues_exact_term_that_dense_misses(embedder):
    chunks = [{"text": t, "source": "n", "page_start": 1, "page_end": 1, "chunk_id": i}
              for i, t in enumerate(["alpha beta gamma delta epsilon zeta eta theta",
                                     "alpha beta gamma delta epsilon zeta eta rareword"])]
    r = make_retriever(chunks, embedder, "hybrid", min_similarity=0.0)
    assert r.query("rareword", top_k=1).chunks[0]["chunk_id"] == 1


def test_mode_validation(sample_chunks):
    with pytest.raises(ValueError):
        Retriever(sample_chunks, mode="nonsense")
    with pytest.raises(ValueError):
        Retriever(sample_chunks, mode="dense")  # dense needs index + embedder


def test_top_k_larger_than_corpus(sample_chunks, embedder):
    r = make_retriever(sample_chunks, embedder, "dense")
    assert len(r.query("process", top_k=50).chunks) == len(sample_chunks)


def test_index_uses_inner_product(sample_chunks, embedder):
    r = make_retriever(sample_chunks, embedder, "dense")
    assert isinstance(r.index, faiss.IndexFlatIP)
