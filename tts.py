"""
tts.py -- Text-to-speech.
"""

import pyttsx3


def speak(text: str):
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.say(text)
    engine.runAndWait()
    engine.stop()