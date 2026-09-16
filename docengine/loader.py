"""PDF text extraction with page numbers."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import pdfplumber

logger = logging.getLogger(__name__)


class DocumentError(Exception):
    """Raised when a document cannot be read."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based
    text: str


def load_pdf_pages(file_bytes: bytes) -> list[Page]:
    """Return the non-empty pages of a PDF, keeping 1-based page numbers."""
    if not file_bytes:
        raise DocumentError("The uploaded file is empty.")
    pages: list[Page] = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append(Page(number=i, text=text))
    except DocumentError:
        raise
    except Exception as exc:  # pdfplumber raises several unrelated types
        logger.warning("Failed to parse PDF: %s", exc)
        raise DocumentError("Could not read this file as a PDF.") from exc
    return pages


def load_pdf(file_bytes: bytes) -> str:
    """Return the full text of a PDF (pages joined by newlines)."""
    return "".join(p.text + "\n" for p in load_pdf_pages(file_bytes))
