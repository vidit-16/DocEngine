"""Retrieval behaviour.

The tests that matter here are the ones about *which* chunk comes back. Earlier
there were none, and the retriever silently returned the opening of the document
for every question.
"""

from __future__ import annotations

import pytest

from src.retriever import fuse, keyword_scores, search, tokenize

# ── tokenizing ───────────────────────────────────────────────────────────────


def test_stopwords_are_removed():
    assert tokenize("What is the total revenue?") == ["total", "revenue"]


def test_punctuation_does_not_stay_glued_to_words():
    """"revenue?" used to be a distinct token from "revenue" and matched nothing."""
    assert "revenue" in tokenize("revenue?")
    assert "revenue" in tokenize("(revenue),")


def test_numbers_are_kept():
    """In a document of figures, the number is often the discriminating term."""
    tokens = tokenize("What was revenue in 2024, was it 42.7 crore?")
    assert "2024" in tokens
    assert "42.7" in tokens


def test_single_characters_are_dropped():
    assert tokenize("a b revenue") == ["revenue"]


def test_a_query_of_only_stopwords_yields_nothing():
    assert tokenize("what is the of and") == []


# ── keyword scoring ──────────────────────────────────────────────────────────


def test_a_query_of_only_stopwords_scores_nothing(chunks):
    """The old code matched every chunk on "the" and returned the first five."""
    assert keyword_scores("what is the", chunks) == {}


def test_chunks_without_a_query_term_are_not_scored(chunks):
    scores = keyword_scores("revenue", chunks)
    assert 3 in scores              # the revenue chunk
    assert set(scores) == {3}       # and nothing else


def test_rare_terms_outweigh_common_ones(chunks):
    """"board" appears twice in the corpus, "revenue" once."""
    scores = keyword_scores("board revenue", chunks)
    assert scores[3] > scores[0]


def test_covering_more_of_the_query_scores_higher(chunks):
    scores = keyword_scores("total revenue subscription growth", chunks)
    best = max(scores, key=scores.get)
    assert best == 3


# ── the regression ───────────────────────────────────────────────────────────
#
# Each of these asks for something that is NOT at the start of the document.
# The previous retriever returned chunks 0..4 in order for any query, so every
# one of these would have failed.


@pytest.mark.parametrize(
    "query,expected_chunk",
    [
        ("What was total revenue?", 3),
        ("How much did revenue grow?", 3),
        ("Tell me about headcount", 2),
        ("Did the engineering team grow?", 2),
        ("Was a dividend recommended?", 4),
    ],
)
def test_retrieval_finds_the_relevant_chunk(chunks, index, query, expected_chunk):
    results = search(query, chunks, index=index, k=3)
    assert results[0].chunk.index == expected_chunk


def test_the_opening_chunk_is_not_returned_for_every_query(chunks, index):
    """The single clearest symptom of the old behaviour."""
    tops = {
        search(query, chunks, index=index, k=1)[0].chunk.index
        for query in [
            "revenue",
            "headcount",
            "dividend",
            "engineering organisation",
        ]
    }
    assert len(tops) > 1


def test_the_semantic_index_actually_contributes(chunks, index):
    """The FAISS index used to be built and then never consulted."""
    results = search("subscription growth", chunks, index=index, k=5)
    assert any(item.semantic_rank is not None for item in results)


def test_agreement_between_strategies_ranks_a_chunk_first(chunks, index):
    results = search("total revenue subscription", chunks, index=index, k=5)
    assert results[0].matched_both


# ── contract ─────────────────────────────────────────────────────────────────


def test_k_limits_the_number_of_results(chunks, index):
    assert len(search("revenue growth report", chunks, index=index, k=2)) == 2


def test_results_are_ordered_by_descending_score(chunks, index):
    results = search("revenue and headcount", chunks, index=index, k=5)
    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True)


def test_an_empty_query_returns_nothing(chunks, index):
    assert search("", chunks, index=index) == []
    assert search("   ", chunks, index=index) == []


def test_an_empty_corpus_returns_nothing(index):
    assert search("revenue", [], index=index) == []


def test_keyword_only_search_works_without_an_index(chunks):
    """Retrieval must be usable, and testable, without an embedding model."""
    results = search("total revenue", chunks, index=None, k=3)
    assert results[0].chunk.index == 3
    assert results[0].semantic_rank is None


def test_a_query_matching_nothing_returns_nothing(chunks):
    assert search("xylophone bicycle", chunks, index=None) == []


def test_retrieved_chunks_carry_their_page_number(chunks, index):
    results = search("total revenue", chunks, index=index, k=1)
    assert results[0].chunk.page == 2


# ── fusion ───────────────────────────────────────────────────────────────────


def test_fusion_rewards_a_chunk_both_strategies_rank():
    fused = fuse(keyword={5: 1, 9: 2}, semantic={5: 2, 7: 1}, k=3)
    assert fused[0][0] == 5


def test_fusion_still_surfaces_a_chunk_only_one_strategy_found():
    """Either strategy alone can put something in the results."""
    positions = [position for position, *_ in fuse({1: 1}, {2: 1}, k=5)]
    assert set(positions) == {1, 2}


def test_fusion_returns_at_most_k():
    assert len(fuse({i: i + 1 for i in range(10)}, {}, k=3)) == 3


def test_the_default_depth_matches_what_was_measured(chunks, index):
    """DEFAULT_K is a measured choice, not a taste; pin it so it isn't nudged.

    Paraphrase recall runs 0.32, 0.36, 0.50, 0.55 at k = 3, 5, 8, 10 and then
    plateaus. See evaluation/RESULTS.md.
    """
    from src.retriever import DEFAULT_K

    assert DEFAULT_K == 8
    assert len(search("revenue growth report headcount", chunks, index=index)) <= DEFAULT_K
