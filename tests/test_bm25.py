import math

from docengine.bm25 import BM25, tokenize


def test_tokenize():
    assert tokenize("Hello, World! 42x") == ["hello", "world", "42x"]


def test_ranks_matching_doc_first():
    bm = BM25(["the cat sat", "dogs bark loudly", "a cat and a dog"])
    assert bm.top_k("dogs bark", 2) == [1]
    assert bm.top_k("cat", 3) == [0, 2]


def test_rare_terms_weigh_more():
    bm = BM25(["apple common", "banana common", "cherry common"])
    assert bm.idf["apple"] > bm.idf["common"]


def test_score_value_matches_formula():
    bm = BM25(["x y", "x"], k1=1.5, b=0.75)
    idf_y = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    norm = 1.5 * (1 - 0.75 + 0.75 * 2 / 1.5)
    assert math.isclose(bm.scores("y")[0], idf_y * 1 * 2.5 / (1 + norm))
    assert bm.scores("y")[1] == 0.0


def test_term_frequency_increases_score():
    bm = BM25(["spam spam spam eggs", "spam eggs ham toast"])
    s = bm.scores("spam")
    assert s[0] > s[1]


def test_length_normalisation():
    bm = BM25(["key", "key filler filler filler filler filler"])
    s = bm.scores("key")
    assert s[0] > s[1]


def test_empty_corpus_and_query():
    assert BM25([]).top_k("anything", 3) == []
    assert BM25(["a b"]).top_k("", 3) == []


def test_top_k_limits_and_ties():
    bm = BM25(["z", "z", "z"])
    assert bm.top_k("z", 2) == [0, 1]


def test_default_parameters_are_standard():
    docs = ["alpha beta beta", "beta gamma", "alpha alpha delta epsilon zeta"]
    default, explicit = BM25(docs), BM25(docs, k1=1.5, b=0.75)
    assert default.k1 == 1.5 and default.b == 0.75
    assert default.scores("alpha beta") == explicit.scores("alpha beta")
