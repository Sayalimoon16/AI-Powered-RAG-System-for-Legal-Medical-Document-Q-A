import sys
import os
from pathlib import Path
import streamlit as st
import tempfile
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
import whisper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import extract_pdf_text
from src.chunker_data import chunk_documents
from src.embedder import build_domain_indexes
from src.retriever import retrieve, load_model, load_index
from src.LLM_chain import generate_answer
from src.utils import check_empty_pdf, detect_language, setup_logger, validate_env


# ------------------------
# INITIAL SETUP
# ------------------------
setup_logger()
validate_env()

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_PATH = BASE_DIR / "output" / "faiss_indexes"
FAISS_PATH.mkdir(parents=True, exist_ok=True)

speech_model = whisper.load_model("base")


# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI RAG Document Assistant")


# ------------------------
# SESSION STATE
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.metadata = None
    st.session_state.chunk_ids = None
    st.session_state.score = None
    st.session_state.filename = None
    st.session_state.doc_loaded = False


# ------------------------
# SIDEBAR
# ------------------------
with st.sidebar:

    st.header("⚙️ System")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    domain_mode = st.selectbox(
        "Select Domain",
        ["Auto-Detect", "Legal", "Medical"]
    )

    st.divider()

    st.write("Embedding Model")
    st.success("MiniLM-L6-v2")

    st.write("Vector DB")
    st.success("FAISS")

    st.write("LLM")
    st.success("Groq Mixtral")

    if st.session_state.score:
        st.metric("Confidence", f"{st.session_state.score:.2f}")


model = load_model()


# ------------------------
# PROCESS PDF
# ------------------------
if uploaded_file:

    if uploaded_file.name != st.session_state.filename:

        st.session_state.filename = uploaded_file.name
        st.session_state.doc_loaded = False
        st.session_state.messages = []
        st.session_state.index = None

    if not st.session_state.doc_loaded:

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Processing document..."):

            pages = extract_pdf_text(pdf_path, uploaded_file.name)

            check_empty_pdf(pages)

            chunks = chunk_documents(pages)

            build_domain_indexes(chunks)

            index, metadata, chunk_ids = load_index(
                str(FAISS_PATH),
                "combined_index"
            )

            st.session_state.index = index
            st.session_state.metadata = metadata
            st.session_state.chunk_ids = chunk_ids
            st.session_state.doc_loaded = True

        st.success("Document Ready")


# ------------------------
# DISPLAY CHAT HISTORY
# ------------------------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ------------------------
# VOICE INPUT
# ------------------------
st.subheader("🎤 Voice Question")

audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop")

voice_question = None

if audio:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:

        temp_audio.write(audio["bytes"])
        audio_path = temp_audio.name

    result = speech_model.transcribe(audio_path)

    voice_question = result["text"]

    st.success(f"You said: {voice_question}")


# ------------------------
# TEXT INPUT
# ------------------------
text_question = st.chat_input("Ask something about your document...")

question = voice_question if voice_question else text_question


# ------------------------
# QUESTION PROCESS
# ------------------------
if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    results = retrieve(
        question,
        st.session_state.index,
        st.session_state.metadata,
        st.session_state.chunk_ids,
        model,
        k=5,
        doc_type=domain_mode.lower() if domain_mode != "Auto-Detect" else None
    )

    answer = generate_answer(question, results, domain_mode)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)

    # voice output
    tts = gTTS(answer)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
        tts.save(audio_file.name)
        st.audio(audio_file.name)

    # confidence score
    if results:
        score = float(results[0]["score"])
        st.session_state.score = score

    # sources
    st.subheader("📚 Sources")

    for r in results[:3]:

        with st.expander(f"{r['source']} — Page {r['page']}"):
            st.write(r["text"])