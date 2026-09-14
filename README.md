# Document Q&A Engine

**Upload a PDF, ask a question, get an answer grounded in the document — with the page it came from.**

DocEngine is a small Retrieval-Augmented Generation system built around one idea:
retrieve the relevant parts of a document first, then let the model answer using
only those passages.

Live demo: https://doc--engine.streamlit.app/

## The pipeline

```text
PDF
 ↓  pdfplumber, page by page
pages
 ↓  400-char chunks, 50-char overlap, never crossing a page boundary
chunks (each tagged with its page)
 ↓  MiniLM embeddings, L2-normalised
FAISS inner-product index
 ↓
hybrid retrieval: keyword ranking + semantic ranking, fused
 ↓
top passages → gpt-4o-mini, answering only from them, citing pages
```

## Retrieval

This is the part of the system that decides what the model is allowed to see, so
it is the part worth explaining.

Two strategies run on every query.

**Keyword.** The query is tokenised: lowercased, punctuation stripped, stopwords
removed, numbers kept. Chunks are scored by how much of the query's content
vocabulary they contain, weighted so that a rare term counts for more than a
common one, and scaled by what fraction of the query the chunk covers.

**Semantic.** The query is embedded and matched against the FAISS index by
cosine similarity.

The two rankings are then combined with Reciprocal Rank Fusion: each strategy
contributes `1 / (60 + rank)` to the chunks it ranks highly. A chunk both agree
on rises to the top, while a chunk only one of them is confident about can still
surface. RRF fuses *rankings* rather than scores, which avoids having to invent
a common scale for a keyword weight and a cosine similarity.

### The bug this replaced

An earlier version was keyword-first with a shortcut:

```python
if any(word in chunk_lower for word in query.lower().split()):
    keyword_hits.append(chunk)
if len(keyword_hits) >= 3:
    return keyword_hits[:k]
```

`query.lower().split()` keeps stopwords, and almost every chunk of English prose
contains "the". So `keyword_hits` filled from the start of the document, the
`>= 3` condition passed on essentially every query, and the function returned
**the first five chunks of the document regardless of the question**. The
semantic branch below it was unreachable in practice: the FAISS index was built
on every upload and never consulted.

The same `split()` also left punctuation attached, so a query ending in
"revenue?" would not match the word "revenue" in the text — the one term that
mattered was the one that failed.

The demo still produced plausible answers, because `gpt-4o-mini` is good at
working with whatever context it is handed. That is exactly why it went
unnoticed, and it is the argument for testing retrieval separately from
generation: a bad retriever behind a good model looks fine until you check.

`tests/test_retriever.py` pins the fixed behaviour, including a case for each
symptom above.

## Citations

Chunks carry the page they came from, passages reach the model labelled
`[page 4]`, and the prompt requires page numbers in the answer. Without that, a
grounded answer and a confident guess look identical.

## Running it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

The model name can be overridden with `DOCENGINE_MODEL`.

## Tests

```bash
pip install -r requirements-test.txt
pytest
```

59 tests, no API key and no network. The embedding model is replaced with a
deterministic bag-of-words vectoriser and the OpenAI client with a stub, so the
suite installs about 50MB rather than the roughly 2GB a torch stack needs, and
runs in well under a second.

That split is deliberate: `requirements.txt` is what the app needs,
`requirements-test.txt` is what the tests need. If a test ever starts needing
torch, that is a sign it has grown a dependency on something it should be
stubbing.

## Structure

```text
.
├── app.py                 # Streamlit UI
├── src/
│   ├── loader.py          # PDF → pages
│   ├── chunker.py         # pages → page-tagged chunks
│   ├── embedder.py        # chunks → normalised vectors (model loaded lazily)
│   ├── vector_store.py    # FAISS inner-product index
│   ├── retriever.py       # keyword + semantic, fused
│   └── llm.py             # answer generation with page citations
├── tests/
└── .github/workflows/ci.yml
```

## Known limits

- Chunking is character-based, so it can split mid-sentence. Sentence-aware
  chunking would retrieve better and is the obvious next change.
- Scanned PDFs with no text layer are rejected rather than OCR'd.
- The index is rebuilt per upload and held in memory; there is no persistence
  across sessions.
- Retrieval quality is pinned by unit tests on a synthetic corpus, not measured
  against a labelled question set. A small evaluation set would turn "the tests
  pass" into a number.

## License

MIT.
