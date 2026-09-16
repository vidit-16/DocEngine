"""Cross-encoder reranking.

Keyword and embedding retrieval each score a passage without looking at the
question and passage together. A cross-encoder reads both at once, which is
slower but much better at deciding which of a few dozen candidates actually
answers the question. On the gold set it raised the share of questions whose
best passage ranked first from 44% to 58% (see evaluation/ACCURACY.md).

Like the embedding model, it is loaded on first use, and tests install a stub.
"""

from __future__ import annotations

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME)
    return _model


def set_model(model) -> None:
    """Install a model explicitly. Used by tests to avoid the real download."""
    global _model
    _model = model


def rerank(query: str, texts: list[str]) -> list[float]:
    """Relevance score for each text against the query; higher is more relevant."""
    if not texts:
        return []
    return [float(score) for score in get_model().predict([(query, text) for text in texts])]
