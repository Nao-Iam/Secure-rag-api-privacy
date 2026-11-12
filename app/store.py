"""Optional SQLite persistence for chunks + embeddings.

The API runs fine fully in memory; this is a convenience so a built
index can be reloaded without re-embedding. One table, JSON-encoded
vectors, no server required. PostgreSQL/pgvector is documented in the
README as an *optional* extension, not implemented here.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def save_index(db_path: Path, chunks: list[dict], vectors: list[list[float]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE IF EXISTS chunks")
        con.execute(
            "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT, title TEXT,"
            " source TEXT, trust_tier INTEGER, text TEXT, vector TEXT)"
        )
        for chunk, vec in zip(chunks, vectors):
            con.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                (
                    chunk["chunk_id"],
                    chunk["doc_id"],
                    chunk["title"],
                    chunk["source"],
                    chunk["trust_tier"],
                    chunk["text"],
                    json.dumps(vec),
                ),
            )
        con.commit()
    finally:
        con.close()


def load_index(db_path: Path) -> tuple[list[dict], list[list[float]]] | None:
    if not db_path.exists():
        return None
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT chunk_id, doc_id, title, source, trust_tier, text, vector"
            " FROM chunks ORDER BY chunk_id"
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return None
    chunks = [
        {
            "chunk_id": r[0],
            "doc_id": r[1],
            "title": r[2],
            "source": r[3],
            "trust_tier": r[4],
            "text": r[5],
        }
        for r in rows
    ]
    vectors = [json.loads(r[6]) for r in rows]
    return chunks, vectors
