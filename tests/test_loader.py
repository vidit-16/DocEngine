import pytest

from docengine.chunker import chunk_pages
from docengine.loader import DocumentError, load_pdf, load_pdf_pages


def test_pages_have_numbers(pdf_factory):
    pdf = pdf_factory(["First page text", "", "Third page text"])
    pages = load_pdf_pages(pdf)
    assert [p.number for p in pages] == [1, 3]
    assert "Third page" in pages[1].text


def test_load_pdf_parity_with_pages(pdf_factory):
    pdf = pdf_factory(["alpha beta", "gamma delta"])
    assert load_pdf(pdf) == "".join(p.text + "\n" for p in load_pdf_pages(pdf))
    assert "alpha beta" in load_pdf(pdf)


def test_end_to_end_chunks_carry_page(pdf_factory):
    pdf = pdf_factory(["x" * 80, "y" * 80])
    chunks = chunk_pages(load_pdf_pages(pdf), chunk_size=100, overlap=10, min_len=10)
    assert [c.page for c in chunks] == [1, 2]


def test_empty_bytes():
    with pytest.raises(DocumentError, match="empty"):
        load_pdf_pages(b"")


def test_garbage_bytes():
    with pytest.raises(DocumentError, match="PDF"):
        load_pdf_pages(b"this is not a pdf at all")
