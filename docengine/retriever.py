"""Semantic, lexical (BM25) and hybrid retrieval over chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from docengine.bm25 import BM25
from docengine.chunker import Chunk
from docengine.config import RETRIEVAL_MODES
from docengine.embedder import get_embeddings
from docengine.vector_store import create_index

RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Fuse ranked id lists; ties are broken by first appearance."""
    scores: dict[int, float] = {}
    order: dict[int, int] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
            order.setdefault(doc, len(order))
    return sorted(scores, key=lambda d: (-scores[d], order[d]))


def search(query: str, model: Any, index: Any, chunks: Sequence[str], k: int = 3) -> list[str]:
    """Pure semantic search returning chunk texts (original API)."""
    k = min(k, len(chunks))
    if k == 0:
        return []
    query_embedding = get_embeddings(model, [query])
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0] if i != -1]


class Retriever:
    def __init__(self, chunks: Sequence[Chunk], model: Any, embeddings: Any = None) -> None:
        self.chunks = list(chunks)
        self.model = model
        texts = [c.text for c in self.chunks]
        self.index = None
        if self.chunks:
            if embeddings is None:
                embeddings = get_embeddings(model, texts)
            self.index = create_index(embeddings)
        self.bm25 = BM25(texts)

    def _semantic_ids(self, query: str, k: int) -> list[int]:
        if self.index is None:
            return []
        _, idx = self.index.search(get_embeddings(self.model, [query]), k)
        return [int(i) for i in idx[0] if i != -1]

    def search(self, query: str, k: int = 3, mode: str = "hybrid") -> list[Chunk]:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}")
        if not query.strip():
            return []
        k = min(k, len(self.chunks))
        if k <= 0:
            return []
        if mode == "semantic":
            ids = self._semantic_ids(query, k)
        elif mode == "bm25":
            ids = self.bm25.top_k(query, k)
        else:
            depth = min(len(self.chunks), max(k * 4, 20))
            ids = reciprocal_rank_fusion(
                [self._semantic_ids(query, depth), self.bm25.top_k(query, depth)]
            )[:k]
        return [self.chunks[i] for i in ids]
