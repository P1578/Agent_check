"""BM25 con normalización (acentos, stopwords, stemming tosco) y sinónimos. Sin dependencias."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from .text import content_words, norm, stem


class BM25:
    def __init__(self, docs: list[str], synonyms: Iterable[Iterable[str]] = (), k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(content_words(d, ignore=frozenset())) for d in docs]
        self.len = [sum(c.values()) or 1 for c in self.tf]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(w for c in self.tf for w in c)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.syn: dict[str, set[str]] = {}
        for g in synonyms:
            stems = {stem(norm(x)) for x in g}
            for s in stems:
                self.syn.setdefault(s, set()).update(stems)

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[tuple[str, float]]:
        q = set()
        for w in content_words(query, ignore=frozenset()):
            q |= self.syn.get(w, {w})
        scored = []
        for i, c in enumerate(self.tf):
            s = sum(
                self.idf.get(w, 0) * c[w] * (self.k1 + 1) / (c[w] + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
                for w in q if w in c
            )
            if s > min_score:
                scored.append((self.docs[i], s))
        return sorted(scored, key=lambda x: -x[1])[:k]
