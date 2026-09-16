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

### Measured effect

Two question sets over the same 15-page paper. *Lexical overlap* is the fraction
of a question's content words that also appear in the passage answering it —
how much of the question can be solved by string matching alone.

**Lexical questions** (overlap 0.69, 20 questions) — built from the document's
own distinctive terms:

| Retriever | Recall@1 | Recall@5 | Recall@8 | MRR | Page@1 |
| --- | --- | --- | --- | --- | --- |
| legacy | 0.00 | 0.20 | 0.25 | 0.073 | 0.05 |
| keyword only | 0.75 | 1.00 | 1.00 | 0.852 | 0.40 |
| hybrid | **0.80** | **1.00** | **1.00** | **0.892** | **0.55** |

**Paraphrase questions** (overlap 0.14, 22 questions) — the same material asked
in a reader's words, deliberately avoiding the document's vocabulary:

| Retriever | Recall@1 | Recall@5 | Recall@8 | MRR | Page@1 |
| --- | --- | --- | --- | --- | --- |
| legacy | 0.00 | 0.18 | 0.23 | 0.078 | 0.09 |
| keyword only | 0.18 | 0.36 | 0.45 | 0.242 | 0.23 |
| hybrid | **0.27** | **0.36** | **0.50** | **0.328** | **0.36** |

Three things worth reading off these tables.

The old retriever never ranked the right passage first, on either set. That is
the bug, and it is not a close call.

The lexical set alone would have been misleading: keyword matching already
scores 1.00 there, so it cannot show whether embeddings contribute anything. The
paraphrase set is what separates them — hybrid beats keyword-only on every
metric once the question stops sharing words with the answer. That is the
argument for keeping the semantic half.

Paraphrase recall of 0.50 is not good. It is reported because it is true: this
retriever handles vocabulary it has seen far better than vocabulary it has not.
Sentence-aware chunking is the obvious next lever.

Method, caveats and the command to reproduce are in
[`evaluation/RESULTS.md`](evaluation/RESULTS.md). The old retriever is kept
verbatim in `evaluation/legacy.py` so the comparison can be re-run rather than
taken on trust.

### Why eight passages

`DEFAULT_K` is 8, chosen by measurement. Paraphrase recall runs 0.32, 0.36,
0.50, 0.55 at k = 3, 5, 8, 10 and then flattens, while the lexical set sits at
1.00 throughout. Eight costs roughly 800 extra tokens per query and buys 18
points of recall on the hard questions.

A chunk-size sweep from 400 to 1600 characters — measured under a fixed
2000-character context budget, so larger chunks got no free advantage — came out
non-monotonic: 0.36, 0.45, 0.27, 0.45, 0.27, 0.18. That is noise at 22
questions, not signal, so chunk size was left alone. Picking 600 because it
scored well would have been fitting the benchmark.

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

## Text extraction

pdfplumber inserts a space when the gap between two characters exceeds
`x_tolerance`, which defaults to 3 points. That is too wide for the Type 1 fonts
most academic PDFs use: narrow spaces fall below the threshold and words arrive
glued together, as `theencoderiscomposedof`.

Measured across five arXiv papers, dropping to 1.5 recovers **30% to 182% more
word tokens** and removes glued tokens entirely. On the Transformer paper the
old setting was surfacing about a third of the document's words, which a keyword
retriever cannot recover from. Override with `DOCENGINE_X_TOLERANCE`.

## Citations

Chunks carry the page they came from, passages reach the model labelled
`[page 4]`, and the prompt requires page numbers in the answer. Without that, a
grounded answer and a confident guess look identical.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your key in OPENAI_API_KEY
streamlit run app.py
```

The key can also come from an ordinary environment variable. The model name can be
overridden with `DOCENGINE_MODEL`.

With Docker (the embedding model is baked into the image):

```bash
docker build -t docengine .
docker run -p 8501:8501 --env-file .env docengine     # http://localhost:8501
```

Unreadable uploads (empty files, non-PDFs, damaged PDFs, scanned PDFs with no text
layer) are reported in the UI rather than raised as a traceback.

## Evaluation

```bash
python evaluation/evaluate.py
```

Downloads the reference paper, runs all three retrievers over the labelled
question set, and rewrites `evaluation/RESULTS.md`. Labels are self-checking:
the harness verifies every evidence string is present in the extracted text and
refuses to run if the gold set and the document have drifted apart.

`--skip-semantic` measures legacy and keyword only, without loading a model.

### Answer models

```bash
python evaluation/answer_ab.py
```

Holds retrieval fixed and compares answer models (`gpt-4o-mini`, `gpt-4.1-mini`,
`gpt-4.1-nano` by default) on the same passages. Each answer is scored on whether it
cites a page the evidence is actually on, and on whether it abstains for three
off-document control questions, where any other reply is an answer from outside
knowledge. Writes `evaluation/ANSWER_AB.md`. This one makes real API calls, a few
cents with the default models.

| Model | Cites a gold page (42 questions) | Abstained | Off-document abstained | Median latency |
| --- | --- | --- | --- | --- |
| gpt-4o-mini (default) | 62% | 7% | 100% | 1.15s |
| gpt-4.1-mini | 64% | 0% | 100% | 1.18s |
| gpt-4.1-nano | 57% | 7% | 100% | 0.92s |

The spread is three questions out of 42, which is inside the noise for a set this
size, so the default stays `gpt-4o-mini`. All three refused every off-document
question rather than answering from outside knowledge.

## Tests

```bash
pip install -r requirements-test.txt
pytest
```

89 tests, no API key and no network. The embedding model is replaced with a
deterministic bag-of-words vectoriser and the OpenAI client with a stub, so the
suite installs about 50MB rather than the roughly 2GB a torch stack needs, and
runs in well under a second.

**Mutation testing.** `python scripts/mutation_test.py` changes the chunker, retriever,
loader and answer module one operator or constant at a time and re-runs the suite
for each change. **55 of 56 mutants are killed (98.2%)**, and CI runs it on every push.
The first run killed 50%: the overlap test used a repeating string, so it passed
with overlap switched off, and nothing pinned the fusion formula, the keyword score
or the short-tail threshold. `tests/test_boundaries.py` closes those. The one
survivor is equivalent: removing a `not` in an early return for "no matches" leaves
fusion to return the same empty list.

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
├── evaluation/            # retrieval eval, answer-model A/B, gold question sets
├── scripts/mutation_test.py
├── tests/
├── Dockerfile
├── .env.example
└── .github/workflows/ci.yml
```

## Known limits

- Chunking is character-based, so it can split mid-sentence. Sentence-aware
  chunking would retrieve better and is the obvious next change.
- Scanned PDFs with no text layer are rejected rather than OCR'd.
- The index is rebuilt per upload and held in memory; there is no persistence
  across sessions.
- The evaluation is one document and 20 questions. Enough to catch a retriever
  that ignores the query, not enough to separate two good ones with confidence.
- Relevance is approximated by an evidence substring appearing in a retrieved
  chunk, so recall is an upper bound on usefulness.
- Answer quality is measured only by citation grounding and abstention
  (`evaluation/answer_ab.py`), not by judging whether the answer is correct.

## License

MIT.
