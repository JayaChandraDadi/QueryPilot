"""Schema and example retrieval with BM25, plus join-path completion.

Only the tables relevant to a question are put in the prompt. After ranking,
the retriever adds any bridge tables needed to join the chosen tables
(e.g. `orders` between `customers` and `order_items`).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .catalog import Catalog

_STOP = {"the", "a", "an", "of", "in", "for", "by", "to", "and", "or", "is", "are", "was", "were", "what", "which",
         "who", "how", "many", "much", "per", "each", "with", "on", "at", "from", "show", "list", "give", "me", "top",
         "did", "do", "does", "that", "this", "their", "its", "all", "most", "least", "than", "between", "during"}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    out = []
    for w in words:
        if w in _STOP:
            continue
        # Light stemming so "orders"/"ordered" match "order".
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in docs) / max(1, len(docs))
        df = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            for q in query:
                f = tf.get(q, 0)
                if f:
                    s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


class SchemaRetriever:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.names = list(catalog.tables)
        docs = []
        for name in self.names:
            t = catalog.tables[name]
            text = [name, name, t.description, " ".join(t.synonyms * 2)]
            for c in t.columns.values():
                text.append(c.name)
                text.append(c.description)
            for g in catalog.glossary:
                if name in g.get("tables", []):
                    text.append(g["term"])
            docs.append(tokenize(" ".join(text)))
        self.bm25 = BM25(docs)

    def rank(self, question: str) -> list[tuple[str, float]]:
        scores = self.bm25.scores(tokenize(question))
        return sorted(zip(self.names, scores), key=lambda x: -x[1])

    def retrieve(self, question: str, k: int = 3, complete_joins: bool = True) -> list[str]:
        ranked = [t for t, s in self.rank(question) if s > 0][:k] or [self.rank(question)[0][0]]
        if not complete_joins:
            return ranked
        chosen = list(ranked)
        # Connect every chosen table to the first one through the FK graph.
        for t in ranked[1:]:
            for bridge in self.catalog.join_path(ranked[0], t):
                if bridge not in chosen:
                    chosen.append(bridge)
        return chosen


class ExampleRetriever:
    """Picks the most similar solved examples to use as few-shot demonstrations."""

    def __init__(self, examples: list[dict]):
        self.examples = examples
        self.bm25 = BM25([tokenize(e["question"]) for e in examples]) if examples else None

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        if not self.examples or k <= 0:
            return []
        scores = self.bm25.scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.examples[i] for i in order[:k]]
