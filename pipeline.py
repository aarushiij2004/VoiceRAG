"""
pipeline.py -- The RAG loop for one question:

    (rewrite follow-up) -> retrieve -> relevance gate -> generate (streamed) -> speak

Kept free of I/O details: the retriever, LLM and speaker are injected, so the same
pipeline serves the CLI, the web API and the unit tests (with fakes).
"""

from collections import deque

from config import HISTORY_TURNS, NO_ANSWER_MESSAGE, TOP_K
from documents import format_citation
from sentences import stream_sentences
from timing import Timings
from tts import NullSpeaker


class RAGPipeline:
    def __init__(self, retriever, llm, speaker=None, top_k: int = TOP_K, stream: bool = True,
                 use_memory: bool = True, history_turns: int = HISTORY_TURNS):
        self.retriever = retriever
        self.llm = llm
        self.speaker = speaker or NullSpeaker()
        self.top_k = top_k
        self.stream = stream
        self.use_memory = use_memory
        self.history = deque(maxlen=history_turns)  # (question, answer) pairs

    def reset(self):
        self.history.clear()

    def ask(self, question: str, timings: Timings = None, on_sentence=None) -> dict:
        """
        Answer one question. `on_sentence(text)` is called as each sentence is ready
        (used by the CLI to print progressively). Returns a dict with the answer,
        sources, retrieval info and per-stage timings.
        """
        timings = timings or Timings()

        # 1. Make follow-ups self-contained ("what about its drawbacks?" -> full question).
        search_query = question
        if self.use_memory and self.history:
            with timings.stage("rewrite"):
                search_query = self.llm.rewrite_query(list(self.history), question)

        # 2. Retrieve.
        with timings.stage("retrieve"):
            result = self.retriever.query(search_query, self.top_k)

        # 3. Relevance gate: don't let the LLM improvise when nothing relevant was found.
        if not result.relevant:
            self.speaker.say(NO_ANSWER_MESSAGE)
            if on_sentence:
                on_sentence(NO_ANSWER_MESSAGE)
            self.speaker.wait()
            self.history.append((question, NO_ANSWER_MESSAGE))
            return {"answer": NO_ANSWER_MESSAGE, "sources": [], "relevant": False,
                    "search_query": search_query, "top_similarity": result.top_similarity,
                    "passages": result.chunks, "timings": timings.stages}

        # 4. Generate (streamed sentence by sentence, or all at once).
        sentences = []
        with timings.stage("generate+speak"):
            if self.stream:
                for sentence in stream_sentences(self.llm.stream_answer(search_query, result.chunks)):
                    if not sentences:
                        timings.mark("first_sentence_at")  # time until the user can hear something
                    sentences.append(sentence)
                    self.speaker.say(sentence)
                    if on_sentence:
                        on_sentence(sentence)
            else:
                text = self.llm.answer(search_query, result.chunks)
                timings.mark("first_sentence_at")
                for sentence in stream_sentences([text]):
                    sentences.append(sentence)
                    self.speaker.say(sentence)
                    if on_sentence:
                        on_sentence(sentence)
            self.speaker.wait()

        answer = " ".join(sentences)
        self.history.append((question, answer))
        sources = list(dict.fromkeys(format_citation(c) for c in result.chunks))
        return {"answer": answer, "sources": sources, "relevant": True,
                "search_query": search_query, "top_similarity": result.top_similarity,
                "passages": result.chunks, "timings": timings.stages}
