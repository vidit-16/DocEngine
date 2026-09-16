import logging

import streamlit as st

from docengine.chunker import chunk_pages
from docengine.config import RETRIEVAL_MODES, ConfigError, Settings
from docengine.embedder import load_model
from docengine.llm import LLMError, format_context, generate_answer
from docengine.loader import DocumentError, load_pdf_pages
from docengine.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Document Q&A Engine", layout="wide")
st.title("📄 Document Q&A Engine")

try:
    settings = Settings.from_env()
except ConfigError as e:
    st.error(f"Invalid configuration: {e}")
    st.stop()


@st.cache_resource
def get_model(name: str):
    return load_model(name)


@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes: bytes, chunk_size: int, overlap: int, min_len: int):
    pages = load_pdf_pages(file_bytes)
    chunks = chunk_pages(pages, chunk_size, overlap, min_len)
    if not chunks:
        return None
    return Retriever(chunks, get_model(settings.embedding_model))


with st.sidebar:
    st.header("Settings")
    mode = st.selectbox(
        "Retrieval", RETRIEVAL_MODES, index=RETRIEVAL_MODES.index(settings.retrieval_mode)
    )
    top_k = st.slider("Chunks to retrieve", 1, 10, settings.top_k)
    model_name = st.text_input("OpenAI model", settings.llm_model)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    with st.spinner("Processing PDF..."):
        try:
            retriever = process_pdf(
                uploaded.getvalue(),
                settings.chunk_size,
                settings.chunk_overlap,
                settings.min_chunk_len,
            )
        except DocumentError as e:
            st.error(str(e))
            st.stop()

    if retriever is None:
        st.error("No text found in this PDF (it may be scanned or empty).")
        st.stop()

    st.success(f"PDF ready! ({len(retriever.chunks)} chunks)")

    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Thinking..."):
            results = retriever.search(query, k=top_k, mode=mode)
            try:
                answer = generate_answer(
                    query,
                    format_context(results),
                    model=model_name or None,
                    temperature=settings.temperature,
                )
            except LLMError as e:
                st.error(f"Failed to generate answer: {e}")
                st.stop()

        st.write("### Answer")
        st.write(answer)

        with st.expander("📚 Sources"):
            for i, chunk in enumerate(results, start=1):
                st.markdown(f"**{i}. Page {chunk.page}** - {chunk.text[:250]}...")
