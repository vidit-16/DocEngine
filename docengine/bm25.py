"""Dependency-free Okapi BM25 for lexical retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        n = len(self.docs)
        self.avgdl = (sum(self.lengths) / n) if n else 0.0
        df: Counter[str] = Counter()
        for c in self.docs:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out: list[float] = []
        for counts, length in zip(self.docs, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl) if self.avgdl else self.k1
            s = 0.0
            for t in terms:
                tf = counts.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            out.append(s)
        return out

    def top_k(self, query: str, k: int) -> list[int]:
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [i for i in ranked[:k] if scores[i] > 0]
