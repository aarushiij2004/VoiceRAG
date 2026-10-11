from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from config import NO_ANSWER_MESSAGE  # noqa: E402
from retrieve import RetrievalResult  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
CHUNK = {"text": "Thrashing is excessive paging.", "source": "n.pdf", "page_start": 41,
         "page_end": 41, "rank": 1, "similarity": 0.71, "bm25": 4.2}


class FakeRetriever:
    relevant = True

    def query(self, q, top_k):
        return RetrievalResult(chunks=[CHUNK], top_similarity=0.71 if self.relevant else 0.05,
                               relevant=self.relevant)


class FakeLLM:
    fail = False

    def stream_answer(self, q, chunks):
        if self.fail:
            raise RuntimeError("gemini is down")
        yield "Thrashing is excessive paging. "
        yield "It slows the system."

    def rewrite_query(self, history, q):
        return q


@pytest.fixture
def env(monkeypatch, tmp_path):
    import config
    import llm
    import retrieve
    import streamlit as st

    st.cache_resource.clear()
    (tmp_path / "faiss.index").write_text("x")
    (tmp_path / "chunks.json").write_text("[]")
    monkeypatch.setattr(config, "INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "chunks.json")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    retriever, fake_llm = FakeRetriever(), FakeLLM()
    monkeypatch.setattr(retrieve.Retriever, "from_disk",
                        classmethod(lambda cls, mode="hybrid", **kw: retriever))
    monkeypatch.setattr(llm, "GeminiLLM", lambda *a, **k: fake_llm)
    yield {"retriever": retriever, "llm": fake_llm, "tmp": tmp_path}
    st.cache_resource.clear()


def run():
    return AppTest.from_file(APP, default_timeout=30).run()


def test_asks_and_shows_answer_sources_and_passages(env):
    at = run()
    assert not at.exception
    at.chat_input[0].set_value("what is thrashing").run()
    assert not at.exception
    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "Thrashing is excessive paging. It slows the system."
    assert messages[1]["sources"] == ["n.pdf p.41"]
    assert messages[1]["passages"][0]["similarity"] == 0.71
    assert any("Retrieved passages" in e.label for e in at.expander)


def test_out_of_scope_question_gets_refusal(env):
    env["retriever"].relevant = False
    at = run()
    at.chat_input[0].set_value("capital of France").run()
    assert at.session_state["messages"][1]["content"] == NO_ANSWER_MESSAGE


def test_llm_failure_shows_error_and_keeps_app_alive(env):
    env["llm"].fail = True
    at = run()
    at.chat_input[0].set_value("what is thrashing").run()
    assert not at.exception
    assert any("Something went wrong" in e.value for e in at.error)
    assert [m["role"] for m in at.session_state["messages"]] == ["user"]


def test_no_index_shows_setup_message(env, monkeypatch):
    import config

    monkeypatch.setattr(config, "INDEX_PATH", env["tmp"] / "missing.index")
    at = run()
    assert any("No index yet" in i.value for i in at.info)
    assert len(at.chat_input) == 0


def test_missing_api_key_shows_error(env, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = run()
    assert any("GEMINI_API_KEY" in e.value for e in at.error)
