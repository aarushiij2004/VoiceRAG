"""
stt.py -- Speech-to-text using faster-whisper.

faster-whisper is a CTranslate2 re-implementation of OpenAI Whisper: same accuracy,
several times faster, lower memory. The model is loaded lazily and cached.

The WAV is read with soundfile and handed to Whisper as a float array, so we don't
depend on faster-whisper's built-in decoder (PyAV), whose version mismatches are a
common source of install-specific crashes. `vad_filter` trims silence before decoding.
"""

import numpy as np
import soundfile as sf

from config import WHISPER_MODEL

TARGET_RATE = 16000
_model = None


def get_model(model_size: str = WHISPER_MODEL):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        # int8 quantisation keeps this fast enough to run on a laptop CPU.
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def load_audio_16k(path: str) -> np.ndarray:
    """Read an audio file as mono float32 at 16 kHz (what Whisper expects)."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != TARGET_RATE and len(mono) > 1:
        # Linear-interpolation resample: dependency-free and adequate for speech.
        duration = len(mono) / rate
        n_out = max(1, int(round(duration * TARGET_RATE)))
        mono = np.interp(np.linspace(0, len(mono) - 1, n_out), np.arange(len(mono)), mono)
    return mono.astype("float32")


def transcribe(audio_path: str, model_size: str = WHISPER_MODEL) -> str:
    model = get_model(model_size)
    audio = load_audio_16k(audio_path)
    segments, _ = model.transcribe(audio, beam_size=5, vad_filter=True)  # beam search > greedy
    return " ".join(segment.text.strip() for segment in segments).strip()
