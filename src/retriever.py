"""Hybrid retrieval.

## What was wrong

The previous implementation was keyword-first with a shortcut:

    if any(word in chunk_lower for word in query.lower().split()):
        keyword_hits.append(chunk)
    if len(keyword_hits) >= 3:
        return keyword_hits[:k]

Two problems, and together they meant the retriever ignored the question.

`query.lower().split()` keeps stopwords, so "what is the total revenue?"
contains "the". Almost every chunk of English prose contains "the", so
`keyword_hits` filled from the start of the document, the `>= 3` condition
passed on essentially every query, and the function returned **the first five
chunks of the document** no matter what was asked. The semantic branch below it
was unreachable in practice: the FAISS index was built on every upload and then
never consulted.

The same split also left punctuation attached, so "revenue?" would not match the
word "revenue" in the text. The one term that mattered was the one that failed.

## What it does now

Both strategies always run, and their rankings are combined with Reciprocal Rank
Fusion. Each strategy contributes 1/(K + rank) to the chunks it ranks highly, so
a chunk both agree on rises to the top, while a chunk either one is confident
about can still surface. RRF fuses rankings rather than scores, so a keyword
count and a cosine similarity can be combined without inventing a scale to
normalise them onto.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from src.chunker import Chunk

# Words that carry no retrieval signal. Kept deliberately short: an aggressive
# list starts removing terms that matter in a technical document.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to from by
for with without about into over under again further is are was were be been
being am do does did doing have has had having it its as so such not no nor
only own same too very can will just should now what which who whom whose when
where why how me my we our you your he him his she her they them their i
""".split())

RRF_K = 60  # standard damping constant; larger flattens the rank weighting

# How many passages to retrieve and hand to the model by default.
#
# Chosen by measurement, not taste. On the paraphrase question set, recall rises
# 0.32 -> 0.36 -> 0.50 -> 0.55 at k = 3, 5, 8, 10 and then plateaus, while the
# lexical set sits at 1.00 throughout. Eight costs roughly 800 extra tokens per
# query against gpt-4o-mini and buys 18 points of recall on the questions that
# are actually hard. See evaluation/RESULTS.md.
DEFAULT_K = 8


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float
    keyword_rank: int | None
    semantic_rank: int | None

    @property
    def matched_both(self) -> bool:
        return self.keyword_rank is not None and self.semantic_rank is not None


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with punctuation stripped and stopwords removed.

    Numbers are kept: in a document full of figures, "2024" or "4.2" is often
    the most discriminating term in the question.
    """
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def keyword_scores(query: str, chunks: list[Chunk]) -> dict[int, float]:
    """Score chunks by how much of the query's content vocabulary they contain.

    The score is coverage weighted by term rarity: a chunk containing a term
    that appears in few chunks is worth more than one containing a term that
    appears everywhere. This is a small idf, and it is what stops a common word
    from dominating the way "the" used to.
    """
    query_terms = set(tokenize(query))
    if not query_terms:
        return {}

    chunk_terms = [set(tokenize(chunk.text)) for chunk in chunks]

    document_frequency: dict[str, int] = defaultdict(int)
    for terms in chunk_terms:
        for term in query_terms & terms:
            document_frequency[term] += 1

    total = len(chunks)
    scores: dict[int, float] = {}
    for position, terms in enumerate(chunk_terms):
        matched = query_terms & terms
        if not matched:
            continue
        weight = sum(
            math.log(1 + total / (1 + document_frequency[term])) for term in matched
        )
        # Coverage matters as well as rarity: matching three of three query
        # terms beats matching one rare one.
        coverage = len(matched) / len(query_terms)
        scores[position] = weight * coverage

    return scores


def _ranks(scores: dict[int, float], limit: int) -> dict[int, int]:
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return {position: rank for rank, (position, _) in enumerate(ordered, start=1)}


def fuse(
    keyword: dict[int, int], semantic: dict[int, int], k: int
) -> list[tuple[int, float, int | None, int | None]]:
    """Reciprocal Rank Fusion over two rankings."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in (keyword, semantic):
        for position, rank in ranking.items():
            fused[position] += 1.0 / (RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [
        (position, score, keyword.get(position), semantic.get(position))
        for position, score in ordered
    ]


def search(
    query: str,
    chunks: list[Chunk],
    index=None,
    k: int = DEFAULT_K,
    pool: int = 30,
) -> list[Retrieved]:
    """Retrieve the k chunks most likely to answer the query.

    `index` is optional: without it the search is keyword-only, which is what
    makes this function testable without loading an embedding model.
    """
    if not chunks or not query.strip():
        return []

    keyword = _ranks(keyword_scores(query, chunks), pool)

    semantic: dict[int, int] = {}
    if index is not None:
        from src.embedder import embed
        from src.vector_store import search_index

        hits = search_index(index, embed([query])[0], pool)
        semantic = {position: rank for rank, (position, _) in enumerate(hits, start=1)}

    if not keyword and not semantic:
        return []

    return [
        Retrieved(
            chunk=chunks[position],
            score=score,
            keyword_rank=keyword_rank,
            semantic_rank=semantic_rank,
        )
        for position, score, keyword_rank, semantic_rank in fuse(keyword, semantic, k)
    ]
