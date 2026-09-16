import pytest
from hypothesis import given
from hypothesis import strategies as st

from docengine.chunker import Chunk, chunk_pages, chunk_text
from docengine.loader import Page


def test_empty_text():
    assert chunk_text("") == []


def test_chunks_overlap():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert all(len(c) <= 500 for c in chunks)
    assert len(chunks) == 3


def test_short_tail_dropped():
    assert chunk_text("short", min_len=50) == []


def test_exact_windows():
    text = "".join(chr(ord("a") + i % 26) for i in range(25))
    assert chunk_text(text, chunk_size=10, overlap=5, min_len=1) == [
        text[0:10],
        text[5:15],
        text[10:20],
        text[15:25],
    ]


def test_single_chunk_when_text_fits():
    assert chunk_text("x" * 60, chunk_size=100, overlap=10) == ["x" * 60]


def test_min_len_is_inclusive():
    assert chunk_text("y" * 50, min_len=50) == ["y" * 50]
    assert chunk_text("y" * 49, min_len=50) == []


def test_whitespace_normalised():
    assert chunk_text("hello\n\n  world\t" + "z" * 50, min_len=1)[0].startswith("hello world")


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (10, 10), (10, -1), (10, 11)])
def test_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=size, overlap=overlap)


def test_zero_overlap_allowed():
    assert chunk_text("ab" * 10, chunk_size=10, overlap=0, min_len=1) == ["ab" * 5, "ab" * 5]


def test_chunk_pages_keeps_page_numbers():
    pages = [Page(1, "a" * 120), Page(3, "b" * 60)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20, min_len=10)
    assert chunks == [Chunk("a" * 100, 1), Chunk("a" * 40, 1), Chunk("b" * 60, 3)]


@given(
    text=st.text(alphabet="abc de\n", max_size=600),
    size=st.integers(5, 120),
    data=st.data(),
)
def test_properties(text, size, data):
    overlap = data.draw(st.integers(0, size - 1))
    chunks = chunk_text(text, chunk_size=size, overlap=overlap, min_len=1)
    norm = " ".join(text.split())
    assert all(1 <= len(c) <= size for c in chunks)
    assert all(c in norm for c in chunks)
    if norm:
        assert norm.endswith(chunks[-1])


def test_default_parameters():
    text = "w" * 1000
    assert chunk_text(text) == chunk_text(text, 500, 100, 50)
    assert [len(c) for c in chunk_text(text)] == [500, 500, 200]
    assert chunk_text("q" * 50) == ["q" * 50]
    assert chunk_text("q" * 49) == []
    pages = [Page(1, text)]
    assert chunk_pages(pages) == chunk_pages(pages, 500, 100, 50)
    assert [len(c.text) for c in chunk_pages(pages)] == [500, 500, 200]
    assert chunk_pages([Page(2, "q" * 49)]) == []
    assert chunk_pages([Page(2, "q" * 50)]) == [Chunk("q" * 50, 2)]


def test_chunk_size_one_allowed():
    assert chunk_text("abc", chunk_size=1, overlap=0, min_len=1) == ["a", "b", "c"]


def test_zero_chunk_size_reports_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("some text", chunk_size=0, overlap=0)
