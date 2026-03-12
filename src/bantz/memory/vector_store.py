"""
Bantz v3 — Local Vector Store (Pure SQLite)

Stores embeddings in a normal SQLite table and performs brute-force cosine
similarity search.  This avoids any native extension dependency (sqlite-vec,
lancedb, etc.) — works everywhere Python + sqlite3 runs.

For Bantz's scale (~10k–100k messages), brute-force on 768-dim vectors is
fast enough (< 50ms for 10k rows on a modern laptop).  If the corpus grows
large we can add sqlite-vec / FAISS as an optional accelerator later.

Schema:
    message_vectors(
        message_id  INTEGER PRIMARY KEY,   -- FK → messages.id
        embedding   BLOB NOT NULL,         -- float32 array as bytes
        dim         INTEGER NOT NULL,      -- vector dimension
        model       TEXT NOT NULL,         -- embedding model name
        created_at  TEXT NOT NULL
    )

Usage:
    from bantz.memory.vector_store import VectorStore

    vs = VectorStore(conn)      # pass existing Memory sqlite connection
    vs.migrate()                # create table if needed
    vs.store(42, [0.1, 0.2, ...], model="nomic-embed-text")
    results = vs.search([0.1, 0.2, ...], limit=5)
"""
from __future__ import annotations

import math
import sqlite3
import struct
import threading
from datetime import datetime
from typing import Optional


def _vec_to_blob(vec: list[float]) -> bytes:
    """Pack a float list into a compact binary blob (float32)."""
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes, dim: int) -> list[float]:
    """Unpack a binary blob back to a float list."""
    return list(struct.unpack(f"{dim}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0:
        return 0.0
    return dot / denom


class VectorStore:
    """Pure-SQLite vector store with brute-force cosine search.

    Shares the same sqlite3.Connection as Memory for zero extra I/O.
    Thread-safe via a shared lock.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        lock: Optional[threading.Lock] = None,
    ) -> None:
        self._conn = conn
        self._lock = lock or threading.Lock()

    # ── Schema ──────────────────────────────────────────────────────────

    def migrate(self) -> None:
        """Create the vector table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS message_vectors (
                message_id  INTEGER PRIMARY KEY,
                embedding   BLOB NOT NULL,
                dim         INTEGER NOT NULL,
                model       TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mv_created
                ON message_vectors(created_at)
        """)

    # ── Write ───────────────────────────────────────────────────────────

    def store(
        self,
        message_id: int,
        embedding: list[float],
        model: str = "nomic-embed-text",
    ) -> None:
        """Store an embedding for a message. Overwrites if exists."""
        blob = _vec_to_blob(embedding)
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO message_vectors
                   (message_id, embedding, dim, model, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (message_id, blob, len(embedding), model, now),
            )

    def store_batch(
        self,
        items: list[tuple[int, list[float]]],
        model: str = "nomic-embed-text",
    ) -> int:
        """Store multiple embeddings.  Returns count stored."""
        now = datetime.now().isoformat(timespec="seconds")
        count = 0
        with self._lock:
            for msg_id, vec in items:
                blob = _vec_to_blob(vec)
                self._conn.execute(
                    """INSERT OR REPLACE INTO message_vectors
                       (message_id, embedding, dim, model, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (msg_id, blob, len(vec), model, now),
                )
                count += 1
        return count

    # ── Read ────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: list[float],
        limit: int = 5,
        min_score: float = 0.3,
        recency_weight: float = 0.0,
    ) -> list[dict]:
        """Brute-force cosine similarity search.

        Args:
            query_vec: Query embedding vector.
            limit: Maximum number of results.
            min_score: Minimum cosine similarity threshold.
            recency_weight: Time-decay factor (0.0 = disabled). When > 0,
                the final score is ``(1 - w) * cosine + w * recency`` where
                recency is an exponential decay ``exp(-age_days / 30)`` so
                messages from the last ~month are boosted and 6-month-old
                topics naturally fade.  Recommended: 0.2–0.4.

        Returns list of ``{message_id, score, role, content, created_at}``
        sorted by descending final score.
        """
        rows = self._conn.execute(
            """SELECT mv.message_id, mv.embedding, mv.dim,
                      m.role, m.content, m.tool_used, m.created_at,
                      m.conversation_id
               FROM message_vectors mv
               JOIN messages m ON m.id = mv.message_id"""
        ).fetchall()

        now_str = datetime.utcnow().isoformat()
        scored: list[tuple[float, dict]] = []
        for row in rows:
            vec = _blob_to_vec(row["embedding"], row["dim"])
            cosine = _cosine_similarity(query_vec, vec)

            # Apply time-decay recency boost (#167)
            if recency_weight > 0.0 and row["created_at"]:
                try:
                    created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00").replace("+00:00", ""))
                    age_days = max((datetime.utcnow() - created).total_seconds() / 86400, 0.0)
                    recency = math.exp(-age_days / 30.0)
                except (ValueError, TypeError):
                    recency = 0.5
                final_score = (1.0 - recency_weight) * cosine + recency_weight * recency
            else:
                final_score = cosine

            if final_score >= min_score:
                scored.append((final_score, {
                    "message_id": row["message_id"],
                    "score": round(final_score, 4),
                    "role": row["role"],
                    "content": row["content"],
                    "tool_used": row["tool_used"],
                    "created_at": row["created_at"],
                    "conv_id": row["conversation_id"],
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def has_embedding(self, message_id: int) -> bool:
        """Check if a message already has an embedding."""
        row = self._conn.execute(
            "SELECT 1 FROM message_vectors WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        return row is not None

    def count(self) -> int:
        """Total number of stored embeddings."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM message_vectors"
        ).fetchone()
        return row[0] if row else 0

    def unembedded_messages(self, limit: int = 100) -> list[dict]:
        """Find messages that don't have embeddings yet.

        Useful for backfilling embeddings on existing conversations.
        """
        # Use column names explicitly — works with or without row_factory
        cur = self._conn.execute(
            """SELECT m.id, m.role, m.content, m.created_at
               FROM messages m
               LEFT JOIN message_vectors mv ON mv.message_id = m.id
               WHERE mv.message_id IS NULL
                 AND m.role IN ('user', 'assistant')
                 AND length(m.content) > 10
               ORDER BY m.created_at DESC
               LIMIT ?""",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Cleanup ─────────────────────────────────────────────────────────

    def prune_orphans(self) -> int:
        """Remove vectors whose messages don't exist anymore."""
        with self._lock:
            cur = self._conn.execute(
                """DELETE FROM message_vectors
                   WHERE message_id NOT IN (SELECT id FROM messages)"""
            )
        return cur.rowcount

    def stats(self) -> dict:
        """Quick stats for diagnostics."""
        total = self.count()
        total_messages = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE role IN ('user','assistant')"
        ).fetchone()[0]
        return {
            "total_embeddings": total,
            "total_messages": total_messages,
            "coverage_pct": round(total / max(total_messages, 1) * 100, 1),
        }
