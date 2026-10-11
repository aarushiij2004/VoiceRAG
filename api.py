"""
api.py -- Optional HTTP API around the same pipeline.

    pip install -r requirements-api.txt
    uvicorn api:app --reload

    POST /ask        {"question": "What is thrashing?"}
    POST /ask-audio  multipart upload of a 16 kHz mono WAV file
    GET  /health

The API is stateless: each request is independent (no conversation memory), and
answers are returned as text/JSON rather than spoken on the server.
"""

import io
import os
import tempfile
import wave

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from timing import Timings


class AskRequest(BaseModel):
    question: str


def _default_pipeline():
    from config import RETRIEVAL_MODE
    from llm import GeminiLLM
    from pipeline import RAGPipeline
    from retrieve import Retriever

    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")
    return RAGPipeline(Retriever.from_disk(mode=RETRIEVAL_MODE), GeminiLLM(),
                       stream=False, use_memory=False)


def create_app(pipeline=None, transcriber=None) -> FastAPI:
    """Factory so tests can inject a fake pipeline and transcriber."""
    app = FastAPI(title="VoiceRAG")
    state = {"pipeline": pipeline, "transcriber": transcriber}

    def get_pipeline():
        if state["pipeline"] is None:  # lazy: load heavy models on first request
            state["pipeline"] = _default_pipeline()
        return state["pipeline"]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask")
    def ask(req: AskRequest):
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be empty")
        return get_pipeline().ask(question)

    @app.post("/ask-audio")
    async def ask_audio(file: UploadFile = File(...)):
        data = await file.read()
        try:
            with wave.open(io.BytesIO(data)) as wav:
                if wav.getframerate() != 16000 or wav.getnchannels() != 1:
                    raise HTTPException(status_code=422, detail="WAV must be 16 kHz mono")
        except wave.Error:
            raise HTTPException(status_code=422, detail="could not read WAV file")

        if state["transcriber"] is None:
            from stt import transcribe
            state["transcriber"] = transcribe

        timings = Timings()
        with timings.stage("stt"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
            try:
                question = state["transcriber"](tmp.name)  # transcribe() takes a file path
            finally:
                os.unlink(tmp.name)
        if not question.strip():
            raise HTTPException(status_code=422, detail="no speech detected")
        result = get_pipeline().ask(question, timings=timings)
        result["transcript"] = question
        return result

    return app


app = create_app()
