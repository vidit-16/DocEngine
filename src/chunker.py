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


# A sentence ends at . ? or ! followed by whitespace and an uppercase letter,
# digit or opening bracket. Common abbreviations are protected so "e.g. The"
# and "et al. (2014)" do not end a sentence.
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "Fig.", "Figs.", "Eq.", "No.", "vs.", "cf.", "approx.")
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9(\[“\"])")


def split_sentences(text: str) -> list[str]:
    text = _normalise(text)
    if not text:
        return []
    protected = text
    for abbreviation in _ABBREVIATIONS:
        protected = protected.replace(abbreviation + " ", abbreviation.replace(".", "") + " ")
    return [s.replace("", ".").strip() for s in _SENTENCE_END.split(protected) if s.strip()]


def chunk_sentences(
    pages: list[Page],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Pack whole sentences into chunks of about ``chunk_size`` characters.

    Character windows cut sentences in half, so the passage that answers a
    question often holds half the sentence a reader would quote. Here a chunk
    ends at a sentence boundary and the next one repeats the last sentence.
    A single sentence longer than ``chunk_size`` becomes its own chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    chunks: list[Chunk] = []
    for page in pages:
        sentences = split_sentences(page.text)
        start = 0
        while start < len(sentences):
            end, length = start, 0
            while end < len(sentences) and (end == start or length + len(sentences[end]) <= chunk_size):
                length += len(sentences[end]) + 1
                end += 1
            chunks.append(
                Chunk(text=" ".join(sentences[start:end]), page=page.number, index=len(chunks))
            )
            if end >= len(sentences):
                break
            start = max(end - overlap_sentences, start + 1)
    return chunks
