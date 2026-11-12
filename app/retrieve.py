"""Retrieve: cosine similarity search over chunk embeddings, plus a
keyword-overlap retrieval used only to *surface* untrusted passages so
the defence can inspect and flag them (never to use them as evidence).
"""

from __future__ import annotations

import math

from .embeddings import tokenize


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve(query: str, chunks: list[dict], chunk_vectors: list[list[float]],
             embedder, top_k: int = 3) -> list[tuple[dict, float]]:
    """Top-k trusted chunks by cosine similarity, best first."""
    if not chunks:
        return []
    qvec = embedder.embed([query])[0]
    scored = [(chunk, cosine(qvec, vec)) for chunk, vec in zip(chunks, chunk_vectors)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def keyword_retrieve(query: str, chunks: list[dict], top_k: int = 2) -> list[tuple[dict, int]]:
    """Rank chunks by shared-token count with the query. Used only to find
    untrusted passages that are topically near the question, so the defence
    can flag them. Their scores are never evidence scores."""
    qtokens = set(tokenize(query))
    scored = []
    for chunk in chunks:
        overlap = len(qtokens & set(tokenize(chunk["text"])))
        scored.append((chunk, overlap))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [(c, s) for c, s in scored[:top_k] if s > 0]
