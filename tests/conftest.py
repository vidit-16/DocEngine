import hashlib
import re

import numpy as np
import pytest


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of Helvetica text per page."""
    objects: list[bytes] = []
    n = len(pages)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n"
    out += trailer.encode() + b"%%EOF\n"
    return bytes(out)


class FakeModel:
    """Deterministic hashed bag-of-words embedder (no downloads)."""

    dim = 64

    def encode(self, texts, normalize_embeddings=True):
        vecs = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vecs[row, h % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def pdf_factory():
    return make_pdf
