import pytest

from documents import build_chunks, chunk_pages, format_citation
from text_utils import clean_page_text, normalize_for_match, remove_repeated_lines


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_chunk_size_and_overlap():
    chunks = chunk_pages([(1, words(500))], "a.txt", chunk_size=200, overlap=40)
    assert all(len(c["text"].split()) <= 200 for c in chunks)
    first, second = chunks[0]["text"].split(), chunks[1]["text"].split()
    assert first[-40:] == second[:40]            # exactly 40 shared words
    assert chunks[-1]["text"].split()[-1] == "w499"  # nothing dropped at the end


def test_every_word_is_covered():
    text = words(333)
    chunks = chunk_pages([(1, text)], "a.txt", 100, 20)
    covered = set()
    for c in chunks:
        covered.update(c["text"].split())
    assert covered == set(text.split())


def test_page_metadata_spans_pages():
    pages = [(1, words(120, "a")), (2, words(120, "b"))]
    chunks = chunk_pages(pages, "a.pdf", 100, 0)
    assert chunks[0]["page_start"] == 1 and chunks[0]["page_end"] == 1
    assert chunks[1]["page_start"] == 1 and chunks[1]["page_end"] == 2  # straddles the page break
    assert chunks[2]["page_start"] == 2


def test_plain_text_has_no_pages():
    chunks = chunk_pages([(None, words(50))], "a.txt", 100, 10)
    assert chunks[0]["page_start"] is None
    assert format_citation(chunks[0]) == "a.txt"


def test_invalid_parameters():
    with pytest.raises(ValueError):
        chunk_pages([(1, "x y z")], "a", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_pages([(1, "x y z")], "a", chunk_size=0, overlap=0)


def test_empty_document_gives_no_chunks():
    assert chunk_pages([(1, "   ")], "a.txt") == []


def test_global_chunk_ids_are_sequential():
    docs = [("a.txt", [(None, words(300))]), ("b.txt", [(None, words(300))])]
    chunks = build_chunks(docs, 100, 20)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_format_citation():
    assert format_citation({"source": "x.pdf", "page_start": 4, "page_end": 4}) == "x.pdf p.4"
    assert format_citation({"source": "x.pdf", "page_start": 4, "page_end": 5}) == "x.pdf pp.4-5"


def test_clean_page_text_removes_control_chars_and_stray_lines():
    cleaned = clean_page_text("Hello\x12\x0f world\nY\n-\nReal line here")
    assert "\x12" not in cleaned
    assert "Real line here" in cleaned
    assert "\nY\n" not in "\n" + cleaned + "\n"


def test_remove_repeated_lines_drops_watermark_only():
    pages = [(i, f"unique content {i}\nCodeHelp") for i in range(1, 9)]
    cleaned = remove_repeated_lines(pages)
    assert all("CodeHelp" not in t for _, t in cleaned)
    assert all(f"unique content {n}" in t for n, t in cleaned)


def test_remove_repeated_lines_leaves_tiny_docs_alone():
    pages = [(1, "a\nfooter"), (2, "b\nfooter")]
    assert remove_repeated_lines(pages) == pages


def test_normalize_for_match():
    assert normalize_for_match("CPU\u2019s  Cache\n State") == "cpu's cache state"
