from sentences import clean_for_speech, stream_sentences


def test_splits_into_sentences():
    assert list(stream_sentences(["Paging is useful. It avoids fragmentation. Done."])) == [
        "Paging is useful.", "It avoids fragmentation.", "Done."]


def test_works_across_token_boundaries():
    tokens = ["Pag", "ing is use", "ful. It av", "oids fragm", "entation."]
    assert list(stream_sentences(tokens)) == ["Paging is useful.", "It avoids fragmentation."]


def test_first_sentence_is_yielded_before_stream_ends():
    consumed = []

    def gen():
        for t in ["First sentence here. ", "Second one ", "continues."]:
            consumed.append(t)
            yield t

    it = stream_sentences(gen())
    assert next(it) == "First sentence here."
    assert len(consumed) < 3  # did not wait for the whole stream


def test_abbreviations_and_decimals_do_not_split():
    out = list(stream_sentences(["Use units, e.g. KB or MB. Version 3.5 is fine."]))
    assert out == ["Use units, e.g. KB or MB.", "Version 3.5 is fine."]


def test_markdown_is_stripped_for_speech():
    assert clean_for_speech("**Paging** is `fast`.\n- item one") == "Paging is fast. item one"


def test_trailing_text_without_punctuation_is_flushed():
    assert list(stream_sentences(["No ending punctuation"])) == ["No ending punctuation"]


def test_empty_stream():
    assert list(stream_sentences([])) == []
