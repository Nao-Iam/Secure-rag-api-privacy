"""Ingest: load Markdown documents from data/corpus/ (tier 1, trusted) and
data/untrusted/ (tier 2, untrusted test passages) into chunks.

Each document carries its metadata in the header lines:
    - Source: <url or SYNTHETIC>
    - Accessed: <date>
    - Trust tier: 1 or 2
The body (after the ``## Summary`` / ``## Content`` heading) is split into
paragraph chunks. A chunk never changes trust tier: untrusted content stays
untrusted no matter where it is retrieved from.
"""

from __future__ import annotations

import re
from pathlib import Path

META_RE = re.compile(r"^-\s*(Source|Accessed|Trust tier):\s*(.*)$")


def doc_id_from_filename(path: Path) -> str:
    """'01-pipeda-principles.md' -> 'pipeda-principles'."""
    stem = path.stem
    return re.sub(r"^\d+-", "", stem)


def parse_document(path: Path, trust_tier: int) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = ""
    source = ""
    accessed = ""
    body_lines: list[str] = []
    in_body = False
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        m = META_RE.match(line)
        if m:
            if m.group(1) == "Source":
                source = m.group(2).strip()
            elif m.group(1) == "Accessed":
                accessed = m.group(2).strip()
            continue
        if line.startswith("## "):
            in_body = True
            continue
        if in_body:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    return {
        "doc_id": doc_id_from_filename(path),
        "title": title,
        "source": source,
        "accessed": accessed,
        "trust_tier": trust_tier,
        "body": body,
        "path": str(path),
    }


def chunk_document(doc: dict) -> list[dict]:
    """Split a document body into paragraph chunks (one chunk per paragraph)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc["body"]) if p.strip()]
    chunks = []
    for i, para in enumerate(paragraphs):
        para = " ".join(para.split())
        if len(para) < 20:
            continue
        chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}#c{i}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source": doc["source"],
                "trust_tier": doc["trust_tier"],
                "text": para,
            }
        )
    return chunks


def load_corpus(corpus_dir: Path, untrusted_dir: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Return (trusted_chunks, untrusted_chunks). Untrusted docs are loaded
    only when ``untrusted_dir`` is given — never mixed into the trusted set."""
    trusted: list[dict] = []
    for path in sorted(corpus_dir.glob("*.md")):
        trusted.extend(chunk_document(parse_document(path, trust_tier=1)))
    untrusted: list[dict] = []
    if untrusted_dir is not None:
        for path in sorted(untrusted_dir.glob("*.md")):
            untrusted.extend(chunk_document(parse_document(path, trust_tier=2)))
    return trusted, untrusted


def list_sources(corpus_dir: Path) -> list[dict]:
    """Metadata for GET /sources: one entry per trusted document."""
    out = []
    for path in sorted(corpus_dir.glob("*.md")):
        doc = parse_document(path, trust_tier=1)
        out.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source": doc["source"],
                "accessed": doc["accessed"],
                "trust_tier": 1,
            }
        )
    return out
