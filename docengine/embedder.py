"""Sentence-transformer embeddings (L2-normalised float32)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


def load_model(name: str = MODEL_NAME) -> Any:
    from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

    return SentenceTransformer(name)


def get_embeddings(model: Any, texts: Sequence[str]) -> np.ndarray:
    return np.asarray(model.encode(list(texts), normalize_embeddings=True), dtype="float32")
