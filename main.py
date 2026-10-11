"""
main.py -- VoiceRAG command-line app.

    python main.py                 # voice mode: press Enter, speak, press Enter
    python main.py --text          # type questions instead (no microphone needed)
    python main.py --mode bm25     # dense | bm25 | hybrid (default from config)
    python main.py --timings       # print a per-stage latency breakdown
    python main.py --debug         # print retrieval scores for each passage
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from config import RETRIEVAL_MODE, TOP_K
from retrieve import MODES, Retriever
from timing import Timings


def parse_args():
    p = argparse.ArgumentParser(description="VoiceRAG: ask spoken questions about your documents.")
    p.add_argument("--text", action="store_true", help="type questions instead of speaking")
    p.add_argument("--mode", choices=MODES, default=RETRIEVAL_MODE, help="retrieval strategy")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--no-stream", action="store_true", help="wait for the full answer before speaking")
    p.add_argument("--no-memory", action="store_true", help="treat every question independently")
    p.add_argument("--no-tts", action="store_true", help="print answers without speaking them")
    p.add_argument("--timings", action="store_true", help="print per-stage latency")
    p.add_argument("--debug", action="store_true", help="print retrieval scores per passage")
    args = p.parse_args()
    if args.top_k < 1:
        p.error("--top-k must be at least 1")
    return args


def print_debug(result: dict):
    from documents import format_citation

    print(f"  [search query: {result['search_query']!r}  top similarity: {result['top_similarity']}]")
    for c in result["passages"]:
        scores = ", ".join(f"{k}={c[k]:.3f}" for k in ("similarity", "bm25") if k in c)
        print(f"   #{c['rank']} {format_citation(c)} ({scores})")


def read_question(args, timings: Timings) -> str:
    """Get one question from the keyboard or the microphone."""
    if args.text:
        return input("You: ").strip()

    from audio_utils import record_until_enter
    from stt import transcribe

    wav_path = None
    try:
        wav_path = record_until_enter()  # prompts for Enter itself, returns a temp WAV path
        with timings.stage("stt"):
            question = transcribe(wav_path)
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
    print(f"You: {question or '[no speech detected]'}")
    return question


def main() -> int:
    load_dotenv()
    args = parse_args()
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        return 1

    from llm import GeminiLLM
    from pipeline import RAGPipeline
    from tts import NullSpeaker, SpeechWorker

    print("Loading models...")
    try:
        retriever = Retriever.from_disk(mode=args.mode)
        llm = GeminiLLM()
    except Exception as exc:
        print(f"Could not start: {exc}", file=sys.stderr)
        return 1
    speaker = NullSpeaker() if args.no_tts else SpeechWorker()
    pipeline = RAGPipeline(retriever, llm, speaker, top_k=args.top_k,
                           stream=not args.no_stream, use_memory=not args.no_memory)

    print(f"Ready (retrieval: {args.mode}). Say or type 'quit' to exit, 'reset' to clear memory.\n")
    try:
        while True:
            timings = Timings()
            try:
                question = read_question(args, timings)
                if not question:
                    continue
                command = question.lower().strip(" .!")
                if command in ("quit", "exit"):
                    break
                if command == "reset":
                    pipeline.reset()
                    print("Conversation memory cleared.\n")
                    continue

                print("Assistant: ", end="", flush=True)
                result = pipeline.ask(question, timings=timings,
                                      on_sentence=lambda s: print(s + " ", end="", flush=True))
                print()
                if result["sources"]:
                    print(f"  Sources: {'; '.join(result['sources'])}")
                if args.debug:
                    print_debug(result)
                if args.timings:
                    print(f"  [{timings.summary()}]")
                print()
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception as exc:  # one failed request must not end the session
                print(f"\nRequest failed: {exc}\n", file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        speaker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
