# Document Q&A Engine

**Upload a PDF, ask a question, and get an answer grounded in the document.**

DocEngine is a small Retrieval-Augmented Generation system built around a simple idea: retrieve relevant parts of a document first, then give those passages to an LLM as the only context it can use to answer.

There is a live Streamlit demo:

https://doc--engine.streamlit.app/

---

## What it actually does

A PDF moves through five stages:

```text
PDF
 ↓
pdfplumber text extraction
 ↓
400-character chunks with 50-character overlap
 ↓
SentenceTransformer embeddings
 ↓
FAISS vector index
 ↓
retrieval + LLM answer generation
```

The current retriever is deliberately hybrid rather than purely semantic. It first checks for chunks containing words from the query. When at least three keyword matches are available, those are returned. Otherwise it falls back to vector search with FAISS.

The answer stage then joins the top three retrieved chunks and sends them to `gpt-4o-mini` with an explicit instruction to answer **only from the supplied context**. When the context is not enough, the prompt asks the model to say that the answer was not clearly found in the document.

## Why the retrieval layer matters

The LLM is not given the whole PDF. The system first narrows the document down to a small context window, which makes the answer generation stage simpler and keeps the source material visible in the pipeline.

That also makes the main failure mode explicit: **bad retrieval leads to bad answers**. The quality of chunking and matching matters just as much as the final generation step.

## Sources

The app can optionally show the retrieved source snippets after answering a question. It displays the top matching chunks used by the pipeline, although the current implementation does not preserve exact page or line references.

That means the sources are useful for seeing the supporting text, but they are not yet a page-level citation system.

## Limitations

The current implementation intentionally stays small, and there are a few known boundaries:

- retrieval uses keyword matching first and semantic search as a fallback
- chunks are character-based rather than structure-aware
- source snippets are not tied to exact PDF page or line numbers
- one uploaded PDF is queried at a time
- larger PDFs increase extraction, embedding, and indexing time
- an OpenAI API key is required for answer generation

These are also the natural next places to improve the system: structure-aware chunking, reranking, page-level provenance, and multi-document retrieval.

## Architecture

```text
app.py
  |
  +--> loader.py        PDF → text
  +--> chunker.py       text → overlapping chunks
  +--> embedder.py      chunks → vectors
  +--> vector_store.py  vectors → FAISS index
  +--> retriever.py     query → relevant chunks
  +--> llm.py           query + context → answer
```

The components are kept separate so that the retrieval and generation stages can be changed independently.

## Stack

Python · Streamlit · pdfplumber · Sentence Transformers · `all-MiniLM-L6-v2` · FAISS · OpenAI API

## Run locally

```bash
git clone https://github.com/vidit-16/doc-engine.git
cd doc-engine
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
setx OPENAI_API_KEY "your_api_key"
```

Then run:

```bash
streamlit run app.py
```

The app provides a PDF uploader, question input, generated answer, and optional retrieved source snippets.

## Project structure

```text
.
├── app.py
├── src/
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── llm.py
└── requirements.txt
```

## What I was interested in

The interesting part of this project is the boundary between retrieval and generation.

The retriever decides **what the model gets to see**.

The LLM decides **how that evidence is turned into an answer**.

Keeping those two jobs separate makes the system easier to inspect, and it gives a much clearer place to work when an answer is wrong: was the right passage never retrieved, or was the passage retrieved and then interpreted badly?