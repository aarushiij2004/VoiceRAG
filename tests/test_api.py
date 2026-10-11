import io
import wave

import pytest
from fastapi.testclient import TestClient

from api import create_app


class FakePipeline:
    def ask(self, question, timings=None, on_sentence=None):
        return {"answer": f"echo: {question}", "sources": ["n.pdf p.1"], "relevant": True,
                "search_query": question, "top_similarity": 0.9,
                "timings": timings.stages if timings else {}}


def make_wav(rate=16000, channels=1, frames=1600):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames * channels)
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(create_app(FakePipeline(), transcriber=lambda path: "what is thrashing"))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask(client):
    r = client.post("/ask", json={"question": "what is paging"})
    assert r.status_code == 200 and r.json()["answer"] == "echo: what is paging"


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_ask_audio_transcribes_then_answers(client):
    r = client.post("/ask-audio", files={"file": ("q.wav", make_wav(), "audio/wav")})
    body = r.json()
    assert r.status_code == 200
    assert body["transcript"] == "what is thrashing" and body["answer"].startswith("echo:")
    assert "stt" in body["timings"]


def test_ask_audio_rejects_wrong_format(client):
    bad_rate = client.post("/ask-audio", files={"file": ("q.wav", make_wav(rate=44100), "audio/wav")})
    not_wav = client.post("/ask-audio", files={"file": ("q.wav", b"not audio", "audio/wav")})
    assert bad_rate.status_code == 422 and not_wav.status_code == 422
