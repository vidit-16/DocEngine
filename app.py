import streamlit as st
from src.loader import load_pdf
from src.chunker import chunk_text
from src.embedder import get_embeddings, model
from src.vector_store import create_index
from src.retriever import search
from src.llm import generate_answer

st.set_page_config(page_title="Document Q&A Engine", layout="wide")
st.title("📄 Document Q&A Engine")

# ------------------------
# Session state
# ------------------------
if "last_results" not in st.session_state:
    st.session_state.last_results = []

# ------------------------
# Upload PDF
# ------------------------
uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    file_bytes = uploaded.read()

    # Process PDF
    with st.spinner("Processing PDF..."):
        text = load_pdf(file_bytes)
        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)
        index = create_index(embeddings)

    st.success("PDF ready!")

    # ------------------------
    # Query input
    # ------------------------
    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Thinking..."):
            results = search(query, model, index, chunks)

            # Use top chunks as context
            context = "\n\n".join(results[:3])

            answer = generate_answer(query, context)

        # Save results for sources
        st.session_state.last_results = results

        # ------------------------
        # Show Answer
        # ------------------------
        st.write("### Answer")
        st.write(answer)

        # ------------------------
        # Show Sources (clean + optional)
        # ------------------------
        if st.checkbox("📚 Show Sources"):
            for i, chunk in enumerate(st.session_state.last_results[:2]):
                clean = chunk.replace("\n", " ").strip()
                st.markdown(f"**{i+1}.** {clean[:250]}...")