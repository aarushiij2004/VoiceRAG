"""
text_utils.py -- Cleaning and normalisation helpers shared by ingestion,
retrieval and evaluation.
"""

import re
from collections import Counter

# C0 control characters (except tab/newline). Some PDF fonts decode bullets and
# digits into these, so they appear as invisible junk inside extracted text.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def strip_control_chars(text: str) -> str:
    return _CONTROL_CHARS.sub(" ", text)


def remove_repeated_lines(pages: list, min_fraction: float = 0.25, min_pages: int = 4) -> list:
    """
    Drop header/footer boilerplate: any line that appears on at least
    `min_fraction` of pages (and on at least `min_pages` pages). This is
    corpus-agnostic -- it removes e.g. a watermark printed on every page without
    hard-coding its text. `pages` is a list of (page_number, text).
    """
    n_pages = len(pages)
    if n_pages < min_pages:
        return pages

    counts = Counter()
    for _, text in pages:
        lines = {ln.strip() for ln in text.splitlines() if ln.strip()}
        counts.update(lines)

    threshold = max(min_pages, int(min_fraction * n_pages))
    boilerplate = {ln for ln, c in counts.items() if c >= threshold}

    cleaned = []
    for page_no, text in pages:
        kept = [ln for ln in text.splitlines() if ln.strip() not in boilerplate]
        cleaned.append((page_no, "\n".join(kept)))
    return cleaned


def clean_page_text(text: str) -> str:
    """Strip control chars and drop lines that are a single stray character."""
    text = strip_control_chars(text)
    lines = [ln for ln in text.splitlines() if len(ln.strip()) > 1]
    return "\n".join(lines)


def normalize_for_match(text: str) -> str:
    """Lower-case, unify quotes/dashes, collapse whitespace. Used for evidence matching."""
    text = text.lower()
    text = (
        text.replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2013", "-").replace("\u2014", "-")
    )
    return " ".join(text.split())
