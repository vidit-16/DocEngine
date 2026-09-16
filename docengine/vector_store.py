"""FAISS index construction."""

from __future__ import annotations

import faiss
import numpy as np


def create_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product index; with normalised vectors this is cosine."""
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2-D array")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index
