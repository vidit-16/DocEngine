"""FAISS index construction and search.

Inner product over L2-normalised vectors is cosine similarity, so scores are
bounded in [-1, 1] and comparable across queries. The index previously used L2
distance over unnormalised vectors, where a longer chunk could outrank a more
relevant one purely because of vector magnitude.
"""

from __future__ import annotations

import numpy as np


def create_index(embeddings: np.ndarray):
    import faiss

    if embeddings.shape[0] == 0:
        raise ValueError("cannot build an index from zero embeddings")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype="float32"))
    return index


def search_index(index, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return (chunk position, similarity) pairs, best first."""
    k = max(1, min(k, index.ntotal))
    scores, positions = index.search(
        np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32"), k
    )
    return [
        (int(pos), float(score))
        for pos, score in zip(positions[0], scores[0], strict=True)
        if pos >= 0
    ]
