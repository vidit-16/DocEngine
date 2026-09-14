"""The pre-fix retriever, kept so the comparison in RESULTS.md is reproducible.

This is the implementation that shipped before the retrieval rewrite, adapted
only to the current Chunk type. Its logic is unchanged: the tokenisation, the
`any()` test and the `>= 3` shortcut are exactly as they were.

It exists solely to be measured against. Nothing in the application imports it.
"""

from __future__ import annotations

from src.chunker import Chunk


def legacy_search(query: str, chunks: list[Chunk], k: int = 5) -> list[Chunk]:
    query_words = query.lower().split()

    keyword_hits = []
    for chunk in chunks:
        if any(word in chunk.text.lower() for word in query_words):
            keyword_hits.append(chunk)

    if len(keyword_hits) >= 3:
        return keyword_hits[:k]

    # The original fell through to FAISS here. For the baseline measurement the
    # fallback is irrelevant: the shortcut above fires on effectively every
    # natural-language query, which is the finding being quantified.
    return chunks[:k]
