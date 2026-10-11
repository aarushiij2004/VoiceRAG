"""
sentences.py -- Turn a stream of text fragments into complete sentences, so TTS can
start speaking sentence 1 while the LLM is still writing sentence 2.
"""

import re

# A sentence ends at . ! or ? (optionally followed by a closing quote/bracket), then
# whitespace, then something that looks like the start of a new sentence.
_BOUNDARY = re.compile(r"""([.!?]+["')\]]*)\s+(?=["'(\[A-Z0-9*])""")

# Abbreviations whose trailing period does NOT end a sentence.
_ABBREVIATIONS = {"e.g.", "i.e.", "vs.", "etc.", "dr.", "mr.", "mrs.", "ms.", "fig.", "no.", "approx."}


def clean_for_speech(text: str) -> str:
    """Strip markdown/symbols a TTS engine would read out loud ('asterisk asterisk')."""
    text = re.sub(r"[`*_#>]+", "", text)
    text = re.sub(r"^\s*[-\u2022]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_first_sentence(buffer: str):
    """Return (sentence, remainder) if the buffer holds a complete sentence, else (None, buffer)."""
    for match in _BOUNDARY.finditer(buffer):
        candidate = buffer[: match.end(1)]
        last_word = candidate.split()[-1].lower() if candidate.split() else ""
        if last_word in _ABBREVIATIONS:
            continue
        return candidate, buffer[match.end():]
    return None, buffer


def stream_sentences(fragments):
    """
    Consume an iterator of text fragments; yield cleaned sentences as soon as each is
    complete. Whatever is left when the stream ends is yielded as the final sentence.
    """
    buffer = ""
    for fragment in fragments:
        buffer += fragment
        while True:
            sentence, buffer = _split_first_sentence(buffer)
            if sentence is None:
                break
            cleaned = clean_for_speech(sentence)
            if cleaned:
                yield cleaned
    tail = clean_for_speech(buffer)
    if tail:
        yield tail
