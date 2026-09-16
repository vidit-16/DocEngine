"""PDF loading: page numbers, skipped blank pages, and unreadable input.

The PDFs are assembled by hand so the tests need no fixtures on disk and no
PDF-writing dependency.
"""

from __future__ import annotations

import pytest

from src.loader import DocumentError, EmptyDocumentError, load_pdf


def make_pdf(page_texts: list[str]) -> bytes:
    """A minimal valid PDF with one line of Helvetica text per page ('' = blank page)."""
    objects: list[bytes] = []
    n = len(page_texts)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(page_texts):
        content_id = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET".encode() if text else b""
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def test_pages_keep_reader_numbering_and_blank_pages_are_skipped():
    pages = load_pdf(make_pdf(["Revenue grew this year", "", "Headcount also grew"]))
    assert [page.number for page in pages] == [1, 3]
    assert "Revenue grew" in pages[0].text
    assert "Headcount also grew" in pages[1].text


def test_document_with_no_text_layer_is_empty():
    with pytest.raises(EmptyDocumentError, match="OCR"):
        load_pdf(make_pdf(["", ""]))


def test_empty_upload_is_rejected():
    with pytest.raises(DocumentError, match="empty"):
        load_pdf(b"")


def test_non_pdf_bytes_are_reported_not_raised_raw():
    with pytest.raises(DocumentError, match="could not be read"):
        load_pdf(b"this is a text file, not a PDF")


def test_empty_document_error_is_a_document_error():
    assert issubclass(EmptyDocumentError, DocumentError)


def make_rotated_pdf(text: str) -> bytes:
    """One page with a line of upright text and a line set at 90 degrees."""
    stream = (
        f"BT /F1 12 Tf 72 700 Td (Upright heading) Tj ET "
        f"BT /F1 12 Tf 0 1 -1 0 300 200 Tm ({text}) Tj ET"
    ).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n"
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)


def test_rotated_text_is_read_forwards_with_word_breaks():
    """Text set at 90 degrees used to come out reversed and glued together."""
    [page] = load_pdf(make_rotated_pdf("See Appendix A for details"))
    assert "Upright heading" in page.text
    assert "See Appendix A for details" in page.text
