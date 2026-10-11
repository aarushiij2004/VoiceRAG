from config import NO_ANSWER_MESSAGE
from pipeline import RAGPipeline
from retrieve import RetrievalResult

CHUNK = {"text": "ctx", "source": "n.pdf", "page_start": 3, "page_end": 3}


class FakeRetriever:
    def __init__(self, relevant=True):
        self.relevant, self.queries = relevant, []

    def query(self, q, top_k):
        self.queries.append(q)
        return RetrievalResult(chunks=[CHUNK], top_similarity=0.8 if self.relevant else 0.05,
                               relevant=self.relevant)


class FakeLLM:
    def __init__(self):
        self.answers, self.rewrites = 0, []

    def stream_answer(self, q, chunks):
        self.answers += 1
        yield from ["Thrashing is excessive paging. ", "It hurts performance."]

    def answer(self, q, chunks):
        self.answers += 1
        return "Thrashing is excessive paging. It hurts performance."

    def rewrite_query(self, history, q):
        self.rewrites.append((list(history), q))
        return f"STANDALONE: {q}"


class RecordingSpeaker:
    def __init__(self):
        self.said, self.waited = [], 0

    def say(self, t):
        self.said.append(t)

    def wait(self):
        self.waited += 1


def test_streaming_speaks_sentence_by_sentence():
    spk, seen = RecordingSpeaker(), []
    p = RAGPipeline(FakeRetriever(), FakeLLM(), spk)
    out = p.ask("what is thrashing", on_sentence=seen.append)
    assert spk.said == ["Thrashing is excessive paging.", "It hurts performance."]
    assert seen == spk.said
    assert out["answer"] == "Thrashing is excessive paging. It hurts performance."
    assert out["sources"] == ["n.pdf p.3"]
    assert "first_sentence_at" in out["timings"] and "retrieve" in out["timings"]
    assert out["passages"] == [CHUNK]
    assert spk.waited == 1


def test_non_streaming_path():
    spk = RecordingSpeaker()
    out = RAGPipeline(FakeRetriever(), FakeLLM(), spk, stream=False).ask("q")
    assert spk.said == ["Thrashing is excessive paging.", "It hurts performance."]
    assert out["relevant"] is True


def test_out_of_scope_never_calls_llm():
    llm, spk = FakeLLM(), RecordingSpeaker()
    out = RAGPipeline(FakeRetriever(relevant=False), llm, spk).ask("capital of France?")
    assert llm.answers == 0
    assert out["answer"] == NO_ANSWER_MESSAGE and out["relevant"] is False
    assert spk.said == [NO_ANSWER_MESSAGE]
    assert out["sources"] == []
    assert out["passages"] == [CHUNK]  # still shown, so the user can see why it was refused


def test_follow_ups_are_rewritten_but_first_question_is_not():
    retr, llm = FakeRetriever(), FakeLLM()
    p = RAGPipeline(retr, llm, RecordingSpeaker())
    p.ask("what is thrashing")
    assert llm.rewrites == []                        # no history yet -> no extra LLM call
    out = p.ask("how do I prevent it")
    assert len(llm.rewrites) == 1
    assert retr.queries[-1] == "STANDALONE: how do I prevent it"
    assert out["search_query"].startswith("STANDALONE")
    assert llm.rewrites[0][0][0][0] == "what is thrashing"


def test_memory_can_be_disabled_and_reset():
    llm = FakeLLM()
    p = RAGPipeline(FakeRetriever(), llm, RecordingSpeaker(), use_memory=False)
    p.ask("a"); p.ask("b")
    assert llm.rewrites == []

    llm2 = FakeLLM()
    p2 = RAGPipeline(FakeRetriever(), llm2, RecordingSpeaker())
    p2.ask("a"); p2.reset(); p2.ask("b")
    assert llm2.rewrites == []


def test_history_is_capped():
    p = RAGPipeline(FakeRetriever(), FakeLLM(), RecordingSpeaker(), history_turns=2)
    for q in ["a", "b", "c", "d"]:
        p.ask(q)
    assert [h[0] for h in p.history] == ["c", "d"]
