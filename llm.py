import os

import google.generativeai as genai

MODEL ="gemini-3.6-flash"

SYSTEM_PROMPT = (
    "You are a voice assistant that answers questions using ONLY the provided "
    "context. If the context doesn't contain the answer, say you don't know -- "
    "don't make things up. Keep answers short (2-4 sentences) and conversational, "
    "since they will be read aloud."
)

_model = None


def get_model():
    global _model
    if _model is None:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT)
    return _model


def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )


def get_answer(question: str, chunks: list) -> str:
    model = get_model()
    prompt = build_prompt(question, chunks)
    response = model.generate_content(prompt)
    return response.text

def get_answer_streaming(question: str, chunks: list) -> str:
    model = get_model()
    prompt = build_prompt(question, chunks)
    response = model.generate_content(
        prompt, stream=True, request_options={"timeout": 60}
    )
    full_text = ""
    print("Assistant: ", end="", flush=True)
    for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_text += chunk.text
    print()
    return full_text