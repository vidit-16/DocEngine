import streamlit as st

from src.chunker import chunk_pages
from src.embedder import embed
from src.llm import AnswerError, generate_answer
from src.loader import EmptyDocumentError, load_pdf
from src.retriever import search
from src.vector_store import create_index

st.set_page_config(page_title="Document Q&A Engine", layout="wide")
st.title("📄 Document Q&A Engine")


@st.cache_resource(show_spinner=False)
def build_pipeline(file_bytes: bytes):
    """Parse, chunk and index a PDF. Cached so re-asking does not re-embed."""
    pages = load_pdf(file_bytes)
    chunks = chunk_pages(pages)
    index = create_index(embed([chunk.text for chunk in chunks]))
    return pages, chunks, index


uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    file_bytes = uploaded.read()

    try:
        with st.spinner("Reading and indexing the document..."):
            pages, chunks, index = build_pipeline(file_bytes)
    except EmptyDocumentError as exc:
        st.error(str(exc))
        st.stop()

    st.success(f"Indexed {len(chunks)} passages across {len(pages)} pages.")

    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Retrieving..."):
            results = search(query, chunks, index=index, k=5)

        if not results:
            st.warning("Nothing in the document matched that question.")
            st.stop()

        try:
            with st.spinner("Answering..."):
                answer = generate_answer(query, results[:3])
        except AnswerError as exc:
            st.error(str(exc))
            st.stop()

        st.write("### Answer")
        st.write(answer)

        with st.expander(f"📚 Passages used ({len(results)} retrieved)"):
            for position, item in enumerate(results, start=1):
                matched_by = []
                if item.keyword_rank:
                    matched_by.append(f"keyword #{item.keyword_rank}")
                if item.semantic_rank:
                    matched_by.append(f"semantic #{item.semantic_rank}")

                st.markdown(
                    f"**{position}. Page {item.chunk.page}** "
                    f"· {', '.join(matched_by)} · score {item.score:.4f}"
                )
                st.caption(item.chunk.text[:400] + ("..." if len(item.chunk.text) > 400 else ""))
