from pathlib import Path

import pytest

from documents import build_chunks, load_documents
from eval import compute_metrics, evaluate, first_hit_rank, load_questions, threshold_sweep
from retrieve import Retriever
from text_utils import normalize_for_match

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.json"
PDF = ROOT / "documents" / "OS_Full_Notes.pdf"


def test_first_hit_rank():
    chunks = [{"text": "nothing"}, {"text": "The CPU\u2019s cache state is preserved."}]
    assert first_hit_rank(chunks, "cache state is preserved") == 2
    assert first_hit_rank(chunks, "absent phrase") is None


def test_compute_metrics():
    m = compute_metrics([1, 2, None, 5])
    assert m["hit@1"] == 0.25 and m["hit@3"] == 0.5 and m["hit@5"] == 0.75
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.2) / 4)


def test_compute_metrics_empty():
    assert compute_metrics([]) == {"n": 0}


def test_mrr_ignores_hits_beyond_cutoff():
    assert compute_metrics([11])["mrr"] == 0.0


def test_threshold_sweep():
    rows = threshold_sweep([0.6, 0.5, 0.2], [0.1, 0.15, 0.4], [0.3])
    assert rows[0]["oos_refused"] == pytest.approx(2 / 3)
    assert rows[0]["false_refusals"] == pytest.approx(1 / 3)


@pytest.mark.skipif(not PDF.exists(), reason="sample PDF not present")
def test_every_evidence_phrase_exists_in_the_document():
    """Guards the ground truth: a typo in questions.json would silently cap every metric."""
    docs = load_documents(ROOT / "documents")
    corpus = normalize_for_match(" ".join(t for _, pages in docs for _, t in pages))
    for q in load_questions(QUESTIONS):
        if q.get("in_scope", True):
            assert normalize_for_match(q["evidence"]) in corpus, q["question"]


@pytest.mark.skipif(not PDF.exists(), reason="sample PDF not present")
def test_bm25_end_to_end_on_real_notes():
    docs = load_documents(ROOT / "documents")
    retriever = Retriever(build_chunks(docs, 200, 40), mode="bm25")
    res = retriever.query("What is a zombie process?", top_k=3)
    assert any(c["page_start"] == 16 or c["page_end"] == 16 for c in res.chunks)
    report = evaluate(retriever, load_questions(QUESTIONS))
    assert report["overall"]["hit@5"] > 0.7  # sanity floor, not a benchmark
