"""Splitting pages into retrievable chunks.

Chunks are built per page, so each one knows where it came from and no chunk
ever straddles a page boundary. A chunk spanning two pages could not be cited
honestly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.loader import Page

DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 50

# Below this, a fragment carries no usable meaning and only adds retrieval
# noise. It is dropped unless it is the only thing on its page.
MIN_CHUNK_CHARS = 40


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    index: int  # position in the document, for stable ordering


def _normalise(text: str) -> str:
    """Collapse the line breaks PDF extraction leaves mid-sentence."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_pages(
    pages: list[Page],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split each page into overlapping chunks, tagged with the page number.

    The previous version dropped any chunk shorter than 40 characters, which
    silently discarded the tail of most documents: the final slice of a page is
    usually short, and that is often where totals and conclusions live. Short
    tails are now merged into the preceding chunk instead.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    chunks: list[Chunk] = []
    step = chunk_size - overlap

    for page in pages:
        text = _normalise(page.text)
        if not text:
            continue

        page_chunks: list[str] = []
        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size].strip()
            if piece:
                page_chunks.append(piece)
            if start + chunk_size >= len(text):
                break

        # Fold a too-short tail into its predecessor rather than losing it.
        if len(page_chunks) > 1 and len(page_chunks[-1]) < MIN_CHUNK_CHARS:
            tail = page_chunks.pop()
            page_chunks[-1] = f"{page_chunks[-1]} {tail}".strip()

        for piece in page_chunks:
            chunks.append(Chunk(text=piece, page=page.number, index=len(chunks)))

    return chunks
