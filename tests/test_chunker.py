"""Chunking and PDF loading."""

from __future__ import annotations

import pytest

from src.chunker import MIN_CHUNK_CHARS, chunk_pages
from src.loader import Page

# ── chunking ─────────────────────────────────────────────────────────────────


def test_chunks_carry_the_page_they_came_from(pages):
    chunks = chunk_pages(pages, chunk_size=20, overlap=5)
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert all(chunk.page in (1, 2) for chunk in chunks)


def test_a_chunk_never_straddles_a_page_boundary():
    """A chunk spanning pages could not be cited honestly."""
    pages = [Page(number=1, text="alpha " * 40), Page(number=2, text="beta " * 40)]
    for chunk in chunk_pages(pages, chunk_size=100, overlap=10):
        assert not ("alpha" in chunk.text and "beta" in chunk.text)


def test_chunk_indexes_are_sequential_across_the_document(pages):
    chunks = chunk_pages(pages, chunk_size=20, overlap=5)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_line_breaks_inside_a_page_are_collapsed():
    pages = [Page(number=1, text="a sentence broken\nacross two lines")]
    assert "\n" not in chunk_pages(pages)[0].text


def test_the_whole_page_survives_chunking():
    """Nothing may be silently lost between the page and its chunks."""
    text = "word " * 300
    chunks = chunk_pages([Page(number=1, text=text)], chunk_size=100, overlap=0)
    assert "".join(chunk.text for chunk in chunks).replace(" ", "") == text.replace(" ", "")


def test_a_short_tail_is_merged_rather_than_dropped():
    """The old chunker discarded any chunk under 40 chars.

    The last slice of a page is usually short, and that is often where totals
    and conclusions live, so the end of nearly every document was thrown away.
    """
    text = "x" * 100 + " " + "TOTAL: 42"
    chunks = chunk_pages([Page(number=1, text=text)], chunk_size=100, overlap=0)
    assert any("TOTAL: 42" in chunk.text for chunk in chunks)


def test_a_page_shorter_than_the_minimum_is_still_kept():
    """A short page is the only thing on it; dropping it loses the page."""
    chunks = chunk_pages([Page(number=7, text="Appendix A")], chunk_size=400)
    assert len(chunks) == 1
    assert chunks[0].page == 7
    assert len(chunks[0].text) < MIN_CHUNK_CHARS


def test_overlap_repeats_text_between_neighbours():
    chunks = chunk_pages([Page(number=1, text="abcdefghij" * 10)], chunk_size=50, overlap=10)
    assert chunks[0].text[-10:] == chunks[1].text[:10]


def test_blank_pages_produce_no_chunks():
    assert chunk_pages([Page(number=1, text="   \n  ")]) == []


def test_no_pages_produce_no_chunks():
    assert chunk_pages([]) == []


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_sizing_is_rejected(pages, chunk_size, overlap):
    """An overlap at or above chunk_size loops forever; fail loudly instead."""
    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)


# ── loading ──────────────────────────────────────────────────────────────────


def test_page_numbers_are_one_based_like_a_reader_counts_them():
    from src.loader import Page as P

    assert P(number=1, text="x").number == 1


def test_a_pdf_with_no_text_raises_rather_than_returning_empty(monkeypatch):
    """A scanned PDF should say so, not silently index nothing."""
    import src.loader as loader

    class FakePage:
        chars: list = []

        def filter(self, _predicate):
            return self

        def extract_text(self, **kwargs):
            return None

    class FakePDF:
        pages = [FakePage(), FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(loader.pdfplumber, "open", lambda _: FakePDF())

    with pytest.raises(loader.EmptyDocumentError, match="OCR"):
        loader.load_pdf(b"not really a pdf")


def test_pages_without_text_are_skipped_but_numbering_is_preserved(monkeypatch):
    """Page 2 has no text layer; page 3 must still be numbered 3."""
    import src.loader as loader

    class FakePage:
        chars: list = []

        def filter(self, _predicate):
            return self

        def __init__(self, text):
            self.text = text

        def extract_text(self, **kwargs):
            return self.text

    class FakePDF:
        pages = [FakePage("first"), FakePage(None), FakePage("third")]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(loader.pdfplumber, "open", lambda _: FakePDF())

    result = loader.load_pdf(b"pdf")
    assert [page.number for page in result] == [1, 3]


def test_the_space_tolerance_is_passed_to_pdfplumber(monkeypatch):
    """pdfplumber's default x_tolerance glues words together in academic PDFs.

    Measured across five arXiv papers, the default of 3 lost between 23% and
    65% of word tokens. This pins the narrower value in place.
    """
    import src.loader as loader

    seen = {}

    class FakePage:
        chars: list = []

        def filter(self, _predicate):
            return self

        def extract_text(self, **kwargs):
            seen.update(kwargs)
            return "some text"

    class FakePDF:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(loader.pdfplumber, "open", lambda _: FakePDF())
    loader.load_pdf(b"pdf")

    assert seen["x_tolerance"] == loader.X_TOLERANCE
    assert loader.X_TOLERANCE < 3, "the pdfplumber default is what caused glued words"
