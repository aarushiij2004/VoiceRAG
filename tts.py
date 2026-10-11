"""
tts.py -- Text to speech (pyttsx3, fully offline).

SpeechWorker runs the TTS engine on its own thread with a queue, so the pipeline can
hand over sentence 1 and keep working while it is being spoken.
pyttsx3 engines must be created and used on a single thread, hence the worker.
"""

import queue
import threading

from config import TTS_RATE


class NullSpeaker:
    """Silent stand-in (tests, --no-tts, API)."""

    def say(self, text: str):
        pass

    def wait(self):
        pass

    def close(self):
        pass


class SpeechWorker:
    def __init__(self, rate: int = TTS_RATE):
        self._rate = rate
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                engine.say(text)
                engine.runAndWait()
            finally:
                self._queue.task_done()

    def say(self, text: str):
        """Queue a sentence to be spoken; returns immediately."""
        if text and text.strip():
            self._queue.put(text)

    def wait(self):
        """Block until everything queued so far has been spoken."""
        self._queue.join()

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=5)


def speak(text: str, rate: int = TTS_RATE):
    """One-shot blocking speech (kept for backwards compatibility)."""
    worker = SpeechWorker(rate)
    worker.say(text)
    worker.wait()
    worker.close()
