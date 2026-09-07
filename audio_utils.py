"""
audio_utils.py -- Record microphone audio to a WAV file.

Press Enter to start recording, press Enter again to stop.
No VAD/auto-stop -- keeping this manual is what makes the baseline reliable
to demo. Auto-stop-on-silence is a good stretch goal once the baseline works.
"""

import threading
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1


def record_until_enter(output_path: str = "input.wav") -> str:
    input("Press Enter to start recording...")
    print("Recording... press Enter again to stop.")

    frames = []
    stop_flag = threading.Event()

    def callback(indata, frame_count, time_info, status):
        if not stop_flag.is_set():
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", callback=callback
    )
    with stream:
        input()  # blocks until Enter is pressed again
        stop_flag.set()

    if not frames:
        raise RuntimeError("No audio captured -- check your microphone.")

    audio = np.concatenate(frames, axis=0)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    return output_path
