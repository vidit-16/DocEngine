"""PDF text extraction.

Text is returned per page rather than as one concatenated string. Page numbers
are the only provenance a PDF gives you for free, and once the pages are glued
together there is no way to recover which page an answer came from.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pdfplumber


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as a reader would count them
    text: str


class EmptyDocumentError(ValueError):
    """Raised when a PDF yields no extractable text."""


def load_pdf(file_bytes: bytes) -> list[Page]:
    """Extract text from each page of a PDF.

    Pages that yield no text are skipped rather than added as empty entries:
    scanned pages with no text layer are common, and an empty page would
    otherwise become an empty chunk that can still be retrieved.
    """
    pages: list[Page] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(Page(number=number, text=text))

    if not pages:
        raise EmptyDocumentError(
            "No extractable text found. If this is a scanned PDF it needs OCR first."
        )
    return pages
