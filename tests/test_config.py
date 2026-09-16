import pytest

from docengine import embedder, llm
from docengine.config import ConfigError, Settings


def test_defaults_match_legacy_constants():
    s = Settings()
    assert (s.chunk_size, s.chunk_overlap, s.min_chunk_len, s.top_k) == (500, 100, 50, 3)
    assert s.embedding_model == embedder.MODEL_NAME
    assert s.llm_model == llm.MODEL == "gpt-4o-mini"


def test_from_env(monkeypatch):
    monkeypatch.setenv("DOCENGINE_CHUNK_SIZE", "800")
    monkeypatch.setenv("DOCENGINE_RETRIEVAL_MODE", "BM25")
    monkeypatch.setenv("DOCENGINE_LLM_MODEL", "gpt-4.1-nano")
    s = Settings.from_env()
    assert s.chunk_size == 800 and s.retrieval_mode == "bm25"
    assert s.llm_model == "gpt-4.1-nano"


def test_from_env_blank_uses_default(monkeypatch):
    monkeypatch.setenv("DOCENGINE_TOP_K", "")
    assert Settings.from_env().top_k == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_overlap": 500},
        {"chunk_overlap": -1},
        {"min_chunk_len": -1},
        {"top_k": 0},
        {"retrieval_mode": "x"},
        {"llm_provider": "x"},
    ],
)
def test_invalid(kwargs):
    with pytest.raises(ConfigError):
        Settings(**kwargs)


def test_bad_int(monkeypatch):
    monkeypatch.setenv("DOCENGINE_TOP_K", "three")
    with pytest.raises(ConfigError, match="DOCENGINE_TOP_K"):
        Settings.from_env()
