import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:  # optional: plain environment variables work without it
    pass
else:
    load_dotenv()

from src.chunker import SENTENCE_CHUNK_SIZE, chunk_sentences
from src.embedder import embed
from src.llm import ABSTAIN, AnswerError, expand_query, generate_answer
from src.loader import DocumentError, load_pdf
from src.retriever import DEFAULT_K, search
from src.ui import answer_card, inject, passage, status
from src.vector_store import create_index

st.set_page_config(
    page_title="DocEngine | Document Q&A", page_icon="assets/favicon.png", layout="wide"
)
inject()
st.title("Document Q&A")
st.caption(
    "Upload a PDF and ask questions about it. Answers are drawn only from the "
    "document and cite the pages they come from."
)


@st.cache_resource(show_spinner=False)
def build_pipeline(file_bytes: bytes):
    """Parse, chunk and index a PDF. Cached so re-asking does not re-embed."""
    pages = load_pdf(file_bytes)
    chunks = chunk_sentences(pages, chunk_size=SENTENCE_CHUNK_SIZE)
    index = create_index(embed([chunk.text for chunk in chunks]))
    return pages, chunks, index


uploaded = st.file_uploader("PDF document", type=["pdf"])

if uploaded:
    file_bytes = uploaded.read()

    try:
        with st.spinner("Reading and indexing the document"):
            pages, chunks, index = build_pipeline(file_bytes)
    except DocumentError as exc:
        st.error(str(exc))
        st.stop()

    st.markdown(status(len(pages), len(chunks)), unsafe_allow_html=True)

    query = st.text_input("Question", placeholder="What does the document say about ...?")

    if query:
        with st.spinner("Searching the document"):
            opening = " ".join(page.text for page in pages[:2])
            rewrites = expand_query(query, opening)
            results = search(
                query, chunks, index=index, k=DEFAULT_K, rerank=True, extra_queries=rewrites
            )

        if not results:
            st.warning("No passages in the document match this question. Try different wording.")
            st.stop()

        try:
            with st.spinner("Writing the answer"):
                answer = generate_answer(query, results)
        except AnswerError as exc:
            st.error(str(exc))
            st.stop()

        if answer.strip().rstrip(".").lower() == ABSTAIN.lower():
            st.info(
                "The document does not clearly answer this question. "
                "The closest passages are listed below."
            )
        else:
            st.markdown(answer_card(answer), unsafe_allow_html=True)

        pages_cited = sorted({item.chunk.page for item in results})
        label = ", ".join(str(page) for page in pages_cited)
        with st.expander(f"Sources: {len(results)} passages from pages {label}"):
            for position, item in enumerate(results, start=1):
                matched_by = []
                if item.keyword_rank:
                    matched_by.append(f"keyword match (rank {item.keyword_rank})")
                if item.semantic_rank:
                    matched_by.append(f"meaning match (rank {item.semantic_rank})")

                text = item.chunk.text[:400] + ("..." if len(item.chunk.text) > 400 else "")
                why = (
                    " and ".join(matched_by)
                    if matched_by
                    else "found through a rephrased version of the question"
                )
                st.markdown(
                    passage(position, item.chunk.page, why, text), unsafe_allow_html=True
                )
