"""Embeddings: a clean ``Embedder`` interface with two implementations.

- ``SentenceTransformerEmbedder`` — dense embeddings (all-MiniLM-L6-v2).
  Preferred when available, but never required: it needs a model download
  (~90 MB) and the ``sentence-transformers`` package.
- ``TfidfEmbedder`` — pure-Python TF-IDF, no download, fully offline and
  deterministic. This is the fallback, and the default in tests.

``get_embedder("auto")`` tries the dense model first and falls back to
TF-IDF on any failure (missing package, no network, download error),
reporting which one is actually in use so the API never silently claims
dense retrieval it is not doing.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")

STOPWORDS = frozenset(
    "a an the and or of to in on for with is are was were be been do does did "
    "what when where who whom which how why can could should would may might "
    "it its this that these those there here as at by from about into under "
    "over after before between not no yes if then than so such any all each "
    "per via i you he she we they me him her us them my your his our their "
    "have has had will shall must".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class TfidfEmbedder:
    """Pure-Python TF-IDF with L2-normalised vectors (cosine = dot product)."""

    name = "tfidf"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []

    def fit(self, texts: list[str]) -> None:
        docs = [tokenize(t) for t in texts]
        df: dict[str, int] = {}
        for tokens in docs:
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1
        n = max(len(docs), 1)
        self.vocab = {tok: i for i, tok in enumerate(sorted(df))}
        self.idf = [math.log((n + 1) / (df[tok] + 1)) + 1.0 for tok in sorted(df)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * len(self.vocab)
            tokens = tokenize(text)
            if not tokens:
                out.append(vec)
                continue
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for tok, c in counts.items():
                idx = self.vocab.get(tok)
                if idx is not None:
                    vec[idx] = (c / len(tokens)) * self.idf[idx]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class SentenceTransformerEmbedder:
    """Dense embeddings via sentence-transformers (all-MiniLM-L6-v2)."""

    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy: optional dep

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def fit(self, texts: list[str]) -> None:
        return None  # pre-trained; nothing to fit

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


def get_embedder(preference: str = "auto") -> tuple[object, str, str | None]:
    """Return (embedder, used_name, fallback_reason).

    ``preference``: "auto" | "tfidf" | "dense". "auto" tries dense, then
    falls back to TF-IDF. "dense" raises if the model cannot load.
    """
    if preference == "tfidf":
        return TfidfEmbedder(), "tfidf", None
    try:
        emb = SentenceTransformerEmbedder()
        return emb, emb.name, None
    except Exception as exc:  # missing package, no network, download failure
        if preference == "dense":
            raise RuntimeError(f"dense embedder unavailable: {exc}") from exc
        return TfidfEmbedder(), "tfidf", f"dense embedder unavailable, fell back to TF-IDF: {exc}"
