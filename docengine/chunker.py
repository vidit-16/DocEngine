"""Split text into overlapping character windows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from docengine.loader import Page


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int | None = None


def chunk_text(
    text: str, chunk_size: int = 500, overlap: int = 100, min_len: int = 50
) -> list[str]:
    """Split whitespace-normalised text into windows of ``chunk_size`` chars.

    Consecutive windows share ``overlap`` characters. Windows shorter than
    ``min_len`` are dropped.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = " ".join(text.split())
    chunks: list[str] = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if len(chunk) >= min_len:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break

    return chunks


def chunk_pages(
    pages: Iterable[Page], chunk_size: int = 500, overlap: int = 100, min_len: int = 50
) -> list[Chunk]:
    """Chunk each page separately so every chunk carries its page number."""
    return [
        Chunk(text=c, page=p.number)
        for p in pages
        for c in chunk_text(p.text, chunk_size, overlap, min_len)
    ]
