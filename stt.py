"""
stt.py -- Speech-to-text using faster-whisper.
"""

from faster_whisper import WhisperModel

_model = None


def get_model(model_size: str = "base.en"):
    global _model
    if _model is None:
        # int8 compute keeps this fast enough to run on a laptop CPU.
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str, model_size: str = "base.en") -> str:
    model = get_model(model_size)
    segments, _ = model.transcribe(audio_path, beam_size=5)
    return " ".join(segment.text.strip() for segment in segments)
