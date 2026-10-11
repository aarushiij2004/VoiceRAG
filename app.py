"""
app.py -- Streamlit UI for VoiceRAG.

    pip install -r requirements-ui.txt
    streamlit run app.py

Upload documents, ask by typing or recording, watch the answer stream in, and inspect
exactly which passages were retrieved (with their similarity / BM25 scores).
"""

import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import CHUNKS_PATH, DOCS_DIR, INDEX_PATH, RETRIEVAL_MODE, TOP_K  # noqa: E402
from documents import format_citation  # noqa: E402
from retrieve import MODES  # noqa: E402

st.set_page_config(page_title="VoiceRAG", page_icon="🎙️", layout="wide")
st.title("🎙️ VoiceRAG")
st.caption("Ask questions about your documents. Retrieval and speech-to-text run locally; "
           "the question and retrieved passages are sent to Gemini to write the answer.")


@st.cache_resource(show_spinner="Loading embedding model, index and LLM client...")
def load_components(mode: str):
    from llm import GeminiLLM
    from retrieve import Retriever

    return Retriever.from_disk(mode=mode), GeminiLLM()


@st.cache_resource(show_spinner=False)
def get_speaker():
    from tts import SpeechWorker

    return SpeechWorker()


def index_ready() -> bool:
    return CHUNKS_PATH.exists() and INDEX_PATH.exists()


def transcribe_bytes(data: bytes) -> str:
    from stt import transcribe

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        handle.write(data)
        path = handle.name
    try:
        return transcribe(path)
    finally:
        Path(path).unlink(missing_ok=True)


def render_extras(message: dict):
    """Sources + inspectable passages under an assistant message."""
    if message.get("search_query") and message["search_query"] != message.get("question"):
        st.caption(f"Searched for: {message['search_query']}")
    if message.get("sources"):
        st.caption("Sources: " + "; ".join(message["sources"]))
    passages = message.get("passages") or []
    if passages:
        with st.expander(f"Retrieved passages ({len(passages)})"):
            for c in passages:
                scores = " · ".join(f"{k} {c[k]:.3f}" for k in ("similarity", "bm25") if k in c)
                st.markdown(f"**#{c['rank']} {format_citation(c)}**" + (f"  \n`{scores}`" if scores else ""))
                st.write(c["text"])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Knowledge base")
    uploads = st.file_uploader("Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Save uploads and rebuild index", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Choose at least one file first.")
        else:
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            for uploaded in uploads:
                # Only the basename is used, which prevents path traversal.
                (DOCS_DIR / Path(uploaded.name).name).write_bytes(uploaded.getvalue())
            with st.spinner("Extracting text, embedding and building the index..."):
                try:
                    from ingest import build_index

                    count = build_index()
                    st.cache_resource.clear()
                    st.session_state.pop("pipeline", None)
                    st.success(f"Index rebuilt with {count} chunks.")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    st.divider()
    st.header("Settings")
    mode = st.selectbox("Retrieval mode", MODES, index=MODES.index(RETRIEVAL_MODE))
    top_k = st.slider("Passages to retrieve", 1, 10, TOP_K)
    use_memory = st.toggle("Remember conversation (follow-up questions)", value=True)
    speak_output = st.toggle("Speak answers on this computer", value=False,
                             help="Audio plays on the machine running Streamlit, not in the browser.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        if "pipeline" in st.session_state:
            st.session_state.pipeline.reset()

# ---------------------------------------------------------------- guards
if not index_ready():
    st.info("No index yet. Upload documents in the sidebar and click **Save uploads and rebuild index**, "
            "or run `python ingest.py`.")
    st.stop()
if not os.getenv("GEMINI_API_KEY"):
    st.error("Set GEMINI_API_KEY in your .env file, then restart the app.")
    st.stop()

try:
    retriever, llm = load_components(mode)
except Exception as exc:
    st.error(f"Could not load the index/models: {exc}")
    st.stop()

# ---------------------------------------------------------------- session state
from pipeline import RAGPipeline  # noqa: E402
from tts import NullSpeaker  # noqa: E402

if st.session_state.get("pipeline_mode") != mode or "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(retriever, llm)
    st.session_state.pipeline_mode = mode
pipeline = st.session_state.pipeline
pipeline.top_k, pipeline.use_memory, pipeline.stream = top_k, use_memory, True
pipeline.speaker = get_speaker() if speak_output else NullSpeaker()
st.session_state.setdefault("messages", [])

# ---------------------------------------------------------------- conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_extras(message)

audio = st.audio_input("Or record your question") if hasattr(st, "audio_input") else None
typed = st.chat_input("Ask about your documents...")

question = typed
if not question and audio is not None:
    data = audio.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if digest != st.session_state.get("last_audio"):  # don't re-ask the same recording on reruns
        st.session_state.last_audio = digest
        with st.spinner("Transcribing..."):
            try:
                question = transcribe_bytes(data)
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
        if not question:
            st.warning("No speech detected. Try again.")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        shown = []

        def on_sentence(sentence: str):
            shown.append(sentence)
            placeholder.write(" ".join(shown))

        try:
            result = pipeline.ask(question, on_sentence=on_sentence)
        except Exception as exc:
            placeholder.empty()
            st.error(f"Something went wrong: {exc}")
        else:
            message = {"role": "assistant", "content": result["answer"], "question": question,
                       "sources": result["sources"], "passages": result["passages"],
                       "search_query": result["search_query"]}
            st.session_state.messages.append(message)
            render_extras(message)
