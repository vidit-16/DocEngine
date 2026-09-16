"""Sentence chunking, query-side embedding, reranking and query expansion.

No model is downloaded: the embedding model is the bag-of-words stub from
conftest, the cross-encoder is a stub that scores by word overlap, and the
OpenAI client is a stub returning canned JSON.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src import embedder, llm, reranker
from src.chunker import SENTENCE_CHUNK_SIZE, chunk_sentences, split_sentences
from src.loader import Page
from src.retriever import search, tokenize

# ── sentence chunking ────────────────────────────────────────────────────────


def test_abbreviations_do_not_end_a_sentence():
    text = "See Fig. 2 for details. Kingma et al. (2014) showed it. Values e.g. 0.9 work. Done."
    assert split_sentences(text) == [
        "See Fig. 2 for details.",
        "Kingma et al. (2014) showed it.",
        "Values e.g. 0.9 work.",
        "Done.",
    ]


def test_chunks_end_on_sentence_boundaries_and_repeat_one_sentence():
    sentences = [f"Sentence number {i} is here." for i in range(10)]
    chunks = chunk_sentences([Page(3, " ".join(sentences))], chunk_size=60)
    for chunk in chunks:
        assert chunk.text.endswith(".")
        assert chunk.page == 3
    first, second = chunks[0].text, chunks[1].text
    assert second.startswith(split_sentences(first)[-1])


def test_a_sentence_longer_than_the_limit_is_kept_whole():
    long_sentence = "word " * 50 + "end."
    [chunk] = chunk_sentences([Page(1, long_sentence)], chunk_size=40)
    assert chunk.text == " ".join(long_sentence.split())


def test_every_sentence_lands_in_some_chunk():
    sentences = [f"Fact {i} about the report." for i in range(40)]
    chunks = chunk_sentences([Page(1, " ".join(sentences))], chunk_size=SENTENCE_CHUNK_SIZE)
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)


@pytest.mark.parametrize("chunk_size, overlap", [(0, 1), (100, -1)])
def test_sentence_chunker_rejects_invalid_sizes(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_sentences([Page(1, "A. B.")], chunk_size=chunk_size, overlap_sentences=overlap)


# ── embeddings ───────────────────────────────────────────────────────────────


def test_queries_get_the_instruction_prefix_and_passages_do_not():
    seen = []

    class Recorder:
        def encode(self, texts):
            seen.extend(texts)
            return [[1.0, 0.0] for _ in texts]

    embedder.set_model(Recorder())
    embedder.embed(["a passage"])
    embedder.embed(["a question"], query=True)
    assert seen == ["a passage", embedder.QUERY_PREFIX + "a question"]


# ── reranking and multi-query search ─────────────────────────────────────────


class OverlapCrossEncoder:
    """Scores a (query, passage) pair by shared content words."""

    def predict(self, pairs):
        return [len(set(tokenize(q)) & set(tokenize(p))) for q, p in pairs]


@pytest.fixture
def stub_reranker():
    reranker.set_model(OverlapCrossEncoder())
    yield
    reranker._model = None


def test_rerank_scores_each_passage(stub_reranker):
    assert reranker.rerank("total revenue", ["revenue grew", "total revenue", "board"]) == [1, 2, 0]
    assert reranker.rerank("anything", []) == []


def test_reranked_search_puts_the_best_match_first(chunks, index, stub_reranker):
    results = search("total revenue subscription growth", chunks, index=index, k=3, rerank=True)
    assert results[0].chunk.index == 3


def test_an_extra_query_can_surface_what_the_original_wording_misses(chunks, stub_reranker):
    # "workforce expansion" shares no word with the corpus; the rewrite does.
    assert search("workforce expansion", chunks, index=None, k=3) == []
    results = search(
        "workforce expansion",
        chunks,
        index=None,
        k=3,
        rerank=True,
        extra_queries=["employee headcount grew"],
    )
    assert results[0].chunk.index == 2


def test_blank_extra_queries_are_ignored(chunks, index):
    plain = search("total revenue", chunks, index=index, k=3)
    with_blanks = search("total revenue", chunks, index=index, k=3, extra_queries=["", "  "])
    assert [r.chunk.index for r in plain] == [r.chunk.index for r in with_blanks]


def test_multi_query_results_respect_k(chunks, index, stub_reranker):
    results = search("revenue", chunks, index=index, k=2, rerank=True, extra_queries=["board"])
    assert len(results) == 2


# ── query expansion ──────────────────────────────────────────────────────────


class JsonClient:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        content = json.dumps(self.payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def install_client():
    def install(client):
        llm.set_client(client)
        return client

    yield install
    llm._client = None


def test_expansion_returns_rewrite_and_passage(install_client):
    payload = {"query": "Adam non-stationary objectives", "passage": "Adam suits..."}
    client = install_client(JsonClient(payload))
    assert llm.expand_query("How does it cope?", "Adam: a method...") == [
        "Adam non-stationary objectives",
        "Adam suits...",
    ]
    prompt = client.calls[0]["messages"][0]["content"]
    assert "Adam: a method..." in prompt and "How does it cope?" in prompt
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_expansion_failure_never_breaks_retrieval(install_client):
    install_client(JsonClient(error=ConnectionError("offline")))
    assert llm.expand_query("question", "opening") == []


def test_expansion_ignores_missing_or_blank_fields(install_client):
    install_client(JsonClient({"query": "  ", "passage": "only this"}))
    assert llm.expand_query("question", "opening") == ["only this"]


def test_blank_question_is_not_expanded(install_client):
    client = install_client(JsonClient({"query": "x", "passage": "y"}))
    assert llm.expand_query("   ", "opening") == []
    assert client.calls == []


# ── exact packing, rotation and hyphenation rules ────────────────────────────


def test_sentence_packing_boundaries_are_exact():
    # Each sentence is 10 characters; a chunk holds sentences while
    # (characters so far, counting one joining space each) + next <= 32.
    sentences = [f"{letter * 9}." for letter in "ABCDEF"]
    chunks = [c.text for c in chunk_sentences([Page(1, " ".join(sentences))], chunk_size=32)]
    assert chunks == [
        "AAAAAAAAA. BBBBBBBBB. CCCCCCCCC.",
        "CCCCCCCCC. DDDDDDDDD. EEEEEEEEE.",
        "EEEEEEEEE. FFFFFFFFF.",
    ]


def test_sentence_chunks_without_overlap_do_not_repeat():
    sentences = [f"{letter * 9}." for letter in "ABCD"]
    chunks = [
        c.text
        for c in chunk_sentences([Page(1, " ".join(sentences))], chunk_size=21, overlap_sentences=0)
    ]
    assert chunks == ["AAAAAAAAA. BBBBBBBBB.", "CCCCCCCCC. DDDDDDDDD."]


def _rotated_chars(word_groups, counter_clockwise: bool, gap: float, x: float = 100.0):
    """Characters of one rotated line, 5pt tall, `gap` points between words."""
    chars, position = [], 0.0
    for w, word in enumerate(word_groups):
        if w:
            position += gap
        for letter in word:
            top, bottom = position, position + 5.0
            if counter_clockwise:  # text runs bottom to top: later letters sit higher
                top, bottom = 1000.0 - bottom, 1000.0 - top
            matrix = (0, 1, -1, 0, 0, 0) if counter_clockwise else (0, -1, 1, 0, 0, 0)
            chars.append({"text": letter, "x0": x, "top": top, "bottom": bottom,
                          "matrix": matrix, "upright": False, "object_type": "char"})
            position += 5.0
    return chars


@pytest.mark.parametrize("counter_clockwise", [True, False])
def test_rotated_text_reads_in_the_right_direction(counter_clockwise):
    from src.loader import _rotated_text

    chars = _rotated_chars(["see", "fig"], counter_clockwise, gap=1.5)
    assert _rotated_text(chars) == "see fig"


def test_rotated_word_gap_threshold():
    from src.loader import ROTATED_GAP, _rotated_text

    assert ROTATED_GAP == 1.0
    assert _rotated_text(_rotated_chars(["ab", "cd"], True, gap=0.5)) == "abcd"
    assert _rotated_text(_rotated_chars(["ab", "cd"], True, gap=1.5)) == "ab cd"


@pytest.mark.parametrize("counter_clockwise", [True, False])
def test_rotated_lines_are_ordered_by_reading_direction(counter_clockwise):
    from src.loader import _rotated_text

    first_x, second_x = (100.0, 110.0) if counter_clockwise else (110.0, 100.0)
    chars = _rotated_chars(["first"], counter_clockwise, 1.5, x=first_x)
    chars += _rotated_chars(["second"], counter_clockwise, 1.5, x=second_x)
    assert _rotated_text(chars) == "first\nsecond"


def test_two_letter_head_is_joined_even_when_both_halves_are_words():
    from src.loader import clean_pages

    pages = [Page(1, "we meet to-\nday"), Page(2, "go to the day")]
    assert clean_pages(pages)[0].text == "we meet today"


def test_three_letter_head_with_word_halves_keeps_the_hyphen():
    from src.loader import clean_pages

    pages = [Page(1, "mid-\nand late"), Page(2, "mid and end")]
    assert clean_pages(pages)[0].text == "mid-and late"


def test_expansion_is_deterministic_and_bounds_the_opening(install_client):
    client = install_client(JsonClient({"query": "q", "passage": "p"}))
    llm.expand_query("question", "x" * 1199 + "yz")
    call = client.calls[0]
    assert call["temperature"] == 0
    content = call["messages"][0]["content"]
    assert "x" * 1199 + "y" in content and "yz" not in content


def test_app_chunk_size_is_pinned():
    assert SENTENCE_CHUNK_SIZE == 600


def test_chunk_size_of_one_is_allowed():
    chunks = chunk_sentences([Page(1, "A. B.")], chunk_size=1, overlap_sentences=0)
    assert [c.text for c in chunks] == ["A.", "B."]


def test_joining_spaces_count_towards_the_limit():
    # Two 10-character sentences plus one joining space make 21; a third would need 32.
    sentences = [f"{letter * 9}." for letter in "ABC"]
    chunks = chunk_sentences([Page(1, " ".join(sentences))], chunk_size=31, overlap_sentences=0)
    assert [c.text for c in chunks] == ["AAAAAAAAA. BBBBBBBBB.", "CCCCCCCCC."]


def test_rotated_gap_exactly_at_the_threshold_is_not_a_word_break():
    from src.loader import _rotated_text

    assert _rotated_text(_rotated_chars(["ab", "cd"], True, gap=1.0)) == "abcd"


def test_each_querys_ranking_is_cut_at_the_reranked_depth():
    from src.chunker import Chunk
    from src.retriever import RERANKED_DEPTH

    many = [Chunk(text=f"revenue item{i}", page=1, index=i) for i in range(30)]
    results = search("revenue", many, index=None, k=30, extra_queries=["revenue"])
    assert RERANKED_DEPTH == 20
    assert len(results) == RERANKED_DEPTH


def test_tied_fused_scores_are_ordered_by_position(chunks):
    # The original matches only the revenue chunk (3); the rewrite only the headcount chunk (2).
    # Each is first in its own ranking, so they tie, and the earlier chunk comes first.
    results = search("crore", chunks, index=None, k=2, extra_queries=["headcount"])
    assert [r.chunk.index for r in results] == [2, 3]
