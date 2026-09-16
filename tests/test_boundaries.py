"""Exact-value and boundary tests for chunking and ranking.

The behavioural tests elsewhere check that retrieval works. These pin the
arithmetic underneath it: window positions, the short-tail threshold, the
fusion formula and the keyword score. Each was added because a deliberate
mutation of that line (scripts/mutation_test.py) went unnoticed without it.
"""

from __future__ import annotations

import math
import string

import pytest

from src.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    MIN_CHUNK_CHARS,
    Chunk,
    chunk_pages,
)
from src.loader import Page
from src.retriever import RRF_K, _ranks, fuse, keyword_scores, search, tokenize

# A string with no repeating period, so overlap cannot be satisfied by accident.
DISTINCT = "".join(f"{i:04d}" for i in range(400))  # 1,600 characters


def windows(text: str, **sizing) -> list[str]:
    return [chunk.text for chunk in chunk_pages([Page(number=1, text=text)], **sizing)]


# ── chunking ─────────────────────────────────────────────────────────────────


def test_default_window_is_400_with_50_overlap():
    assert (DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, MIN_CHUNK_CHARS) == (400, 50, 40)
    chunks = windows(DISTINCT)
    assert chunks[0] == DISTINCT[:400]
    assert chunks[1] == DISTINCT[350:750]


def test_overlap_is_exact_on_non_repeating_text():
    chunks = windows(DISTINCT[:130], chunk_size=50, overlap=10)
    assert chunks[:2] == [DISTINCT[0:50], DISTINCT[40:90]]


def test_last_window_that_reaches_the_end_stops_the_loop():
    # 90 characters, window 50, step 40: windows at 0 and 40; the second ends
    # exactly at the end of the text, so no third fragment is produced.
    assert windows(DISTINCT[:90], chunk_size=50, overlap=10) == [DISTINCT[0:50], DISTINCT[40:90]]


def test_a_chunk_size_of_one_is_allowed():
    assert windows("ab", chunk_size=1, overlap=0) == ["a b"]


def test_zero_chunk_size_reports_the_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_pages([Page(number=1, text="text")], chunk_size=0, overlap=0)


def test_overlap_equal_to_chunk_size_reports_the_overlap():
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_pages([Page(number=1, text="text")], chunk_size=10, overlap=10)


@pytest.mark.parametrize("tail, merged", [(MIN_CHUNK_CHARS - 1, True), (MIN_CHUNK_CHARS, False)])
def test_short_tail_threshold(tail, merged):
    text = DISTINCT[: 100 + tail]
    chunks = windows(text, chunk_size=100, overlap=0)
    assert len(chunks) == (1 if merged else 2)
    assert chunks[-1].endswith(text[-tail:])


def test_tail_merges_into_the_last_chunk_not_an_earlier_one():
    text = DISTINCT[:210]  # windows of 100, 100 and a 10-character tail
    chunks = windows(text, chunk_size=100, overlap=0)
    assert chunks == [DISTINCT[0:100], DISTINCT[100:200] + " " + DISTINCT[200:210]]


# ── tokenising and keyword scores ────────────────────────────────────────────


def test_two_character_tokens_are_kept():
    assert tokenize("AI and ML") == ["ai", "ml"]


def test_keyword_score_formula():
    chunks = [
        Chunk(text="alpha beta", page=1, index=0),
        Chunk(text="alpha", page=1, index=1),
        Chunk(text="gamma", page=1, index=2),
    ]
    scores = keyword_scores("alpha beta", chunks)
    total = 3
    alpha = math.log(1 + total / (1 + 2))  # alpha is in two chunks
    beta = math.log(1 + total / (1 + 1))  # beta is in one
    assert scores[0] == pytest.approx((alpha + beta) * 1.0)
    assert scores[1] == pytest.approx(alpha * 0.5)
    assert 2 not in scores


# ── ranking and fusion ───────────────────────────────────────────────────────


def test_ranks_start_at_one_and_break_ties_by_position():
    assert _ranks({4: 1.0, 2: 1.0, 7: 3.0}, limit=10) == {7: 1, 2: 2, 4: 3}
    assert _ranks({4: 1.0, 2: 1.0, 7: 3.0}, limit=1) == {7: 1}


def test_fusion_score_is_reciprocal_rank():
    assert RRF_K == 60
    [(position, score, keyword_rank, semantic_rank)] = fuse({3: 2}, {3: 5}, k=1)
    assert (position, keyword_rank, semantic_rank) == (3, 2, 5)
    assert score == pytest.approx(1 / (60 + 2) + 1 / (60 + 5))


def test_fusion_ties_are_broken_by_document_position():
    assert [position for position, *_ in fuse({9: 1}, {4: 1}, k=2)] == [4, 9]


def test_semantic_ranks_start_at_one(chunks, index):
    results = search("total revenue subscription", chunks, index=index, k=5)
    assert min(item.semantic_rank for item in results if item.semantic_rank) == 1


def test_candidate_pool_limits_results_to_thirty():
    chunks = [
        Chunk(text=f"revenue {string.ascii_lowercase[i % 26]}{i}", page=1, index=i)
        for i in range(40)
    ]
    assert len(search("revenue", chunks, index=None, k=40)) == 30
