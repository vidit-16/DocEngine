"""Fixtures.

Nothing here downloads a model or calls an API. The embedding model is replaced
with a deterministic bag-of-words vectoriser: it is not good at semantics, but
it is a real vector space, which is all the fusion logic needs to be exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.chunker import Chunk
from src.loader import Page

# A small corpus where the right answer is never the first chunk. The old
# retriever returned the opening chunks for every query, so any document whose
# answer lives near the front would have hidden the bug.
CORPUS = [
    "This annual report covers the financial year and was prepared by the board "
    "of directors for the information of the shareholders of the company.",
    "The introduction describes the structure of this document and the "
    "conventions used throughout it for the presentation of the information.",
    "Employee headcount grew over the period, with the engineering organisation "
    "accounting for most of the increase across the regional offices.",
    "Total revenue for the year was 42.7 crore, up from 31.2 crore in the prior "
    "year, driven primarily by subscription growth.",
    "The board recommends a final dividend and thanks the staff for their work "
    "during a demanding year for the organisation as a whole.",
]


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(text=text, page=(position // 2) + 1, index=position)
        for position, text in enumerate(CORPUS)
    ]


@pytest.fixture
def pages() -> list[Page]:
    return [
        Page(number=1, text="First page text.\nSecond line of the first page."),
        Page(number=2, text="Second page begins here with different content."),
    ]


class BagOfWordsModel:
    """A tiny deterministic stand-in for SentenceTransformer.

    Vectors are term-count vectors over a fixed vocabulary built from the
    corpus, so similar texts really do land near each other and the cosine
    ordering is meaningful without downloading anything.
    """

    def __init__(self, vocabulary: list[str]):
        self.vocabulary = vocabulary

    def encode(self, texts):
        from src.retriever import tokenize

        vectors = np.zeros((len(texts), len(self.vocabulary)), dtype="float32")
        for row, text in enumerate(texts):
            counts = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            for column, term in enumerate(self.vocabulary):
                vectors[row, column] = counts.get(term, 0.0)
        return vectors


@pytest.fixture
def fake_model(chunks):
    from src.retriever import tokenize

    vocabulary = sorted({
        token for chunk in chunks for token in tokenize(chunk.text)
    })
    return BagOfWordsModel(vocabulary)


@pytest.fixture(autouse=True)
def installed_model(fake_model):
    from src import embedder

    embedder.set_model(fake_model)
    yield
    embedder._model = None


@pytest.fixture
def index(chunks):
    from src.embedder import embed
    from src.vector_store import create_index

    return create_index(embed([chunk.text for chunk in chunks]))
