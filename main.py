"""
main.py -- Voice RAG Assistant.

Loop: record mic input -> transcribe -> retrieve relevant chunks ->
ask Gemini a grounded question -> speak the answer.

Usage:
    python main.py            # voice mode (needs a microphone)
    python main.py --text     # type questions instead of speaking (no mic needed)
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from retrieve import Retriever
from llm import get_answer
from tts import speak

load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    sys.exit("Set GEMINI_API_KEY in your environment or a .env file first.")


def answer_and_speak(retriever: Retriever, question: str):
    chunks = retriever.query(question, top_k=4)
    if not chunks:
        answer = "I couldn't find anything relevant in the documents for that."
    else:
        answer = get_answer(question, chunks)

    print(f"Assistant: {answer}")
    speak(answer)


def run_text_mode(retriever: Retriever):
    print("Text mode. Type a question (or 'quit' to exit).")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if question:
            answer_and_speak(retriever, question)


def run_voice_mode(retriever: Retriever):
    from audio_utils import record_until_enter
    from stt import transcribe

    print("Voice mode. Ctrl+C to exit.")
    while True:
        try:
            audio_path = record_until_enter()
            question = transcribe(audio_path)
            print(f"You said: {question}")
            if not question.strip():
                print("Didn't catch that -- try again.")
                continue
            answer_and_speak(retriever, question)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text", action="store_true", help="Type questions instead of speaking them"
    )
    args = parser.parse_args()

    retriever = Retriever()

    if args.text:
        run_text_mode(retriever)
    else:
        run_voice_mode(retriever)


if __name__ == "__main__":
    main()
