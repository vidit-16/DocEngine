"""Central configuration, read from environment variables with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

RETRIEVAL_MODES = ("semantic", "bm25", "hybrid")
LLM_PROVIDERS = ("openai",)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
}


class ConfigError(ValueError):
    """Raised when configuration values are invalid."""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_len: int = 50
    top_k: int = 3
    retrieval_mode: str = "semantic"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_provider: str = "openai"
    llm_model: str = field(default="")
    temperature: float = 0.2

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ConfigError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.min_chunk_len < 0:
            raise ConfigError("min_chunk_len must be >= 0")
        if self.top_k <= 0:
            raise ConfigError("top_k must be positive")
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ConfigError(f"retrieval_mode must be one of {RETRIEVAL_MODES}")
        if self.llm_provider not in LLM_PROVIDERS:
            raise ConfigError(f"llm_provider must be one of {LLM_PROVIDERS}")
        if not self.llm_model:
            object.__setattr__(self, "llm_model", DEFAULT_MODELS[self.llm_provider])

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            chunk_size=_env_int("DOCENGINE_CHUNK_SIZE", 500),
            chunk_overlap=_env_int("DOCENGINE_CHUNK_OVERLAP", 100),
            min_chunk_len=_env_int("DOCENGINE_MIN_CHUNK_LEN", 50),
            top_k=_env_int("DOCENGINE_TOP_K", 3),
            retrieval_mode=os.getenv("DOCENGINE_RETRIEVAL_MODE", "semantic").lower(),
            embedding_model=os.getenv("DOCENGINE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_provider=os.getenv("DOCENGINE_LLM_PROVIDER", "openai").lower(),
            llm_model=os.getenv("DOCENGINE_LLM_MODEL", ""),
        )
