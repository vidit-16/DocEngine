"""Sentence embeddings.

The model is loaded on first use, not at import. Loading at import meant that
importing anything downstream — including in a test run or a lint pass —
downloaded and initialised a 90MB model before a single line of logic ran.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

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


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts as L2-normalised float32 vectors.

    Normalising here means an inner-product index computes cosine similarity,
    which is what MiniLM is trained for. The previous setup used raw vectors
    with an L2 index, so magnitude affected ranking.
    """
    if not texts:
        return np.zeros((0, 384), dtype="float32")

    vectors = np.asarray(get_model().encode(texts), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
