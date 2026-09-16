"""PDF text extraction.

Text is returned per page rather than as one concatenated string. Page numbers
are the only provenance a PDF gives you for free, and once the pages are glued
together there is no way to recover which page an answer came from.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass

import pdfplumber

# pdfplumber inserts a space when the gap between two characters exceeds this
# many points. The library default of 3 is too wide for the Type 1 fonts most
# academic PDFs use: narrow spaces fall under the threshold and words come out
# glued together ("theencoderiscomposedof"). Measured across five arXiv papers,
# dropping to 1.5 recovers between 30% and 182% more word tokens and eliminates
# glued tokens entirely, without splitting words apart.
X_TOLERANCE = float(os.getenv("DOCENGINE_X_TOLERANCE", "1.5"))


# Gap, in points, between two rotated characters on the same line that counts
# as a word break. Rotated text is rebuilt by hand (see _rotated_text), so this
# plays the role X_TOLERANCE plays for upright text.
ROTATED_GAP = 1.0


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as a reader would count them
    text: str


class DocumentError(ValueError):
    """Raised when an upload cannot be read as a document."""


class EmptyDocumentError(DocumentError):
    """Raised when a PDF yields no extractable text."""


def _is_rotated_char(obj: dict) -> bool:
    return obj.get("object_type") == "char" and not obj.get("upright", True)


def _rotated_text(chars: list[dict]) -> str:
    """Rebuild text set at 90 degrees, which pdfplumber returns reversed.

    Landscape figures and sidebars are drawn rotated. pdfplumber reads their
    characters in page order, which for rotated text is backwards and without
    word breaks ("gnitsettuoba" for "about testing"). Each rotated line shares
    one x position, so characters are grouped into lines by x and ordered along
    the rotated baseline. The text matrix says which way the text runs.
    """
    if not chars:
        return ""
    lines: dict[int, list[dict]] = {}
    for char in chars:
        lines.setdefault(round(char["x0"]), []).append(char)

    counter_clockwise = chars[0].get("matrix", (0, 1))[1] > 0
    rendered = []
    for x in sorted(lines, reverse=not counter_clockwise):
        ordered = sorted(lines[x], key=lambda c: -c["bottom"] if counter_clockwise else c["top"])
        text, previous = "", None
        for char in ordered:
            if previous is not None:
                gap = (
                    previous["top"] - char["bottom"]
                    if counter_clockwise
                    else char["top"] - previous["bottom"]
                )
                if gap > ROTATED_GAP:
                    text += " "
            text += char["text"]
            previous = char
        rendered.append(text)
    return "\n".join(rendered)


def _page_text(page) -> str:
    upright = page.filter(lambda obj: not _is_rotated_char(obj))
    text = upright.extract_text(x_tolerance=X_TOLERANCE) or ""
    rotated = _rotated_text([char for char in page.chars if _is_rotated_char(char)])
    return f"{text}\n{rotated}".strip() if rotated else text


def load_pdf(file_bytes: bytes) -> list[Page]:
    """Extract text from each page of a PDF.

    Pages that yield no text are skipped rather than added as empty entries:
    scanned pages with no text layer are common, and an empty page would
    otherwise become an empty chunk that can still be retrieved.
    """
    if not file_bytes:
        raise DocumentError("The file is empty.")

    pages: list[Page] = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for number, page in enumerate(pdf.pages, start=1):
                text = _page_text(page)
                if text and text.strip():
                    pages.append(Page(number=number, text=text))
    except Exception as exc:
        # pdfplumber surfaces damaged or non-PDF input as a variety of parser
        # exceptions; the user only needs to know the file could not be read.
        raise DocumentError(
            f"This file could not be read as a PDF ({type(exc).__name__})."
        ) from exc

    if not pages:
        raise EmptyDocumentError(
            "This PDF contains no readable text. It may be a scanned image, "
            "and text recognition (OCR) is not supported."
        )
    return clean_pages(pages)


_UNMAPPED_GLYPH = re.compile(r"\(cid:\d+\)")
_LINE_BREAK_HYPHEN = re.compile(r"([A-Za-z]+)-\n([a-z]+)")
_WORD = re.compile(r"[A-Za-z]+")
_COMPOUND = re.compile(r"[A-Za-z]+-[A-Za-z]+")


def clean_pages(pages: list[Page]) -> list[Page]:
    """Remove extraction artefacts that break both keyword and semantic matching.

    - ``(cid:NN)`` placeholders pdfplumber emits for glyphs it cannot map.
    - Words hyphenated across a line break ("man-\\nagement"). Typeset reports
      are full of them (233 in the NIST AI RMF), and each one turns a word the
      question uses into two fragments neither retriever can match.

    Whether to join is decided from the document's own vocabulary, so genuine
    compounds that happen to wrap ("third-\\nparty") keep their hyphen:
      1. the hyphenated form appears elsewhere in the document -> keep it
      2. the joined word appears elsewhere                      -> join
      3. both halves are words the document uses                -> keep it
      4. otherwise (a syllable split such as "en-\\nergy")        -> join
    """
    texts = [_UNMAPPED_GLYPH.sub("", page.text) for page in pages]
    # The vocabulary must exclude the fragments being judged, or every
    # fragment ("recommenda", "tions") would count as a word the document uses.
    flat = " ".join(_LINE_BREAK_HYPHEN.sub(" ", text) for text in texts)
    words = {w.lower() for w in _WORD.findall(flat)}
    compounds = {c.lower() for c in _COMPOUND.findall(flat)}

    def repair(match: re.Match) -> str:
        head, tail = match.group(1), match.group(2)
        if f"{head}-{tail}".lower() in compounds:
            return f"{head}-{tail}"
        if (head + tail).lower() in words:
            return head + tail
        if head.lower() in words and tail.lower() in words and len(head) > 2:
            return f"{head}-{tail}"
        return head + tail

    return [
        Page(number=page.number, text=_LINE_BREAK_HYPHEN.sub(repair, text))
        for page, text in zip(pages, texts, strict=True)
    ]
