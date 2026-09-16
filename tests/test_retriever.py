import pytest

from docengine.chunker import Chunk
from docengine.embedder import get_embeddings
from docengine.retriever import Retriever, reciprocal_rank_fusion, search
from docengine.vector_store import create_index

DOCS = [
    Chunk("The refund policy allows returns within 30 days of purchase.", 1),
    Chunk("Our office is located in Berlin near the central station.", 2),
    Chunk("Employees receive 25 vacation days per year.", 3),
    Chunk("The warranty covers manufacturing defects for two years.", 4),
]


def test_rrf_fusion():
    assert reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]]) == [1, 3, 2]
    assert reciprocal_rank_fusion([[5], [7, 5]]) == [5, 7]
    assert reciprocal_rank_fusion([]) == []


def test_rrf_score_uses_rank():
    assert reciprocal_rank_fusion([[9, 8], [1, 8]])[0] == 8


@pytest.mark.parametrize("mode", ["semantic", "bm25", "hybrid"])
def test_modes_find_relevant_chunk(fake_model, mode):
    r = Retriever(DOCS, fake_model)
    top = r.search("How many vacation days do employees get?", k=2, mode=mode)
    assert top[0].page == 3
    assert len(top) <= 2


def test_invalid_mode(fake_model):
    with pytest.raises(ValueError):
        Retriever(DOCS, fake_model).search("q", mode="magic")


def test_empty_inputs(fake_model):
    assert Retriever([], fake_model).search("anything") == []
    assert Retriever(DOCS, fake_model).search("   ") == []


def test_k_clamped(fake_model):
    assert len(Retriever(DOCS, fake_model).search("berlin", k=50, mode="semantic")) == 4


def test_parity_semantic_matches_legacy_search(fake_model):
    """The Retriever's semantic mode must equal the original search() function."""
    texts = [c.text for c in DOCS]
    index = create_index(get_embeddings(fake_model, texts))
    r = Retriever(DOCS, fake_model)
    for q in ["refund returns", "office Berlin", "warranty defects", "vacation"]:
        for k in (1, 2, 4):
            legacy = search(q, fake_model, index, texts, k=k)
            assert [c.text for c in r.search(q, k=k, mode="semantic")] == legacy


def test_legacy_search_empty(fake_model):
    assert search("q", fake_model, None, [], k=3) == []


def test_create_index_rejects_empty():
    import numpy as np

    with pytest.raises(ValueError):
        create_index(np.zeros((0, 4), dtype="float32"))


def test_rrf_default_k_is_60():
    # A is first in one list; B is at rank 61 in two lists. With k=60 the scores tie
    # (2/122 == 1/61) and first appearance wins; any other k breaks the tie.
    fill1, fill2 = list(range(100, 160)), list(range(200, 261))
    rankings = [["A", *fill1, "B"], [*fill2, "B"]]
    fused = reciprocal_rank_fusion(rankings)
    assert fused.index("A") < fused.index("B")
    fused61 = reciprocal_rank_fusion(rankings, k=61)
    assert fused61.index("B") < fused61.index("A")


def test_default_k_is_three(fake_model):
    many = [Chunk(f"vacation note number {i}", i) for i in range(10)]
    assert len(Retriever(many, fake_model).search("vacation", mode="semantic")) == 3
    texts = [c.text for c in many]
    index = create_index(get_embeddings(fake_model, texts))
    assert len(search("vacation", fake_model, index, texts)) == 3


def test_k_zero_or_negative(fake_model):
    r = Retriever(DOCS, fake_model)
    assert r.search("berlin", k=0) == []
    assert r.search("berlin", k=-2) == []


def test_semantic_mode_does_not_use_bm25(fake_model, monkeypatch):
    r = Retriever(DOCS, fake_model)
    monkeypatch.setattr(r.bm25, "top_k", lambda *a: pytest.fail("bm25 used"))
    assert r.search("office Berlin", k=1, mode="semantic")[0].page == 2


def test_hybrid_candidate_depth(fake_model, monkeypatch):
    many = [Chunk(f"doc {i} text", i) for i in range(40)]
    r = Retriever(many, fake_model)
    seen = []
    monkeypatch.setattr(r, "_semantic_ids", lambda q, k: seen.append(k) or [])
    monkeypatch.setattr(r.bm25, "top_k", lambda q, k: seen.append(k) or [])
    r.search("doc", k=2, mode="hybrid")
    r.search("doc", k=6, mode="hybrid")
    assert seen == [20, 20, 24, 24]
