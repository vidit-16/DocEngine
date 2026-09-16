"""Sentence embeddings.

The model is loaded on first use, not at import. Loading at import meant that
importing anything downstream — including in a test run or a lint pass —
downloaded and initialised a 90MB model before a single line of logic ran.
"""

from __future__ import annotations

import numpy as np

# bge-small-en-v1.5 is trained with an instruction prefix on queries (not on
# passages). On the four-document gold set it matched MiniLM on recall@8 and
# ranked the right passage first more often (see evaluation/ACCURACY.md).
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model():
    """Return the process-wide SentenceTransformer, loading it on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def set_model(model) -> None:
    """Install a model explicitly. Used by tests to avoid the real download."""
    global _model
    _model = model


def embed(texts: list[str], query: bool = False) -> np.ndarray:
    """Embed texts as L2-normalised float32 vectors.

    Pass ``query=True`` for search queries, which the model expects prefixed.

    Normalising here means an inner-product index computes cosine similarity,
    which is what MiniLM is trained for. The previous setup used raw vectors
    with an L2 index, so magnitude affected ranking.
    """
    if not texts:
        return np.zeros((0, 384), dtype="float32")

    if query:
        texts = [QUERY_PREFIX + text for text in texts]
    vectors = np.asarray(get_model().encode(texts), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
