"""
Bantz v2 — Conversation Memory
SQLite-backed persistent memory. Stores all messages across sessions.

DB schema:
  conversations(id, started_at, last_active)
  messages(id, conversation_id, role, content, tool_used, created_at)

Usage:
    from bantz.core.memory import memory

    # App startup — creates a new session
    memory.new_session()

    # After each exchange
    memory.add("user", user_input)
    memory.add("assistant", response, tool_used="weather")

    # Get last N messages for LLM context
    ctx = memory.context(n=10)   # list[{"role":..., "content":...}]

    # Search past messages
    hits = memory.search("hava durumu", limit=5)
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from bantz.data.store import ConversationStore

log = logging.getLogger("bantz.memory")


class Memory(ConversationStore):
    def __init__(self) -> None:
        self._db_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._session_id: Optional[int] = None
        self._vector_store = None  # VectorStore — lazy init
        self._embed_queue: list[tuple[int, str]] = []  # (msg_id, content) pending embed

    # ── Init ──────────────────────────────────────────────────────────────

    def init(self, db_path: Path) -> None:
        """Call once at startup with the DB path from config."""
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,   # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self._init_vector_store()

    def _migrate(self) -> None:
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at  TEXT NOT NULL,
                last_active TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                role            TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                content         TEXT NOT NULL,
                tool_used       TEXT,
                created_at      TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id, created_at)
        """)
        # FTS for search — optional, graceful fallback if unavailable
        try:
            c.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                USING fts5(content, content='messages', content_rowid='id')
            """)
            c.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
                END
            """)
        except sqlite3.OperationalError:
            pass   # FTS5 not compiled in — search will fall back to LIKE

        # Distillation table (#118)
        try:
            from bantz.memory.distiller import migrate_distillation_table
            migrate_distillation_table(c, self._lock)
        except Exception as exc:
            log.debug("Distillation table migration skipped: %s", exc)

    def _init_vector_store(self) -> None:
        """Initialize the vector store table (shares our sqlite connection)."""
        try:
            from bantz.memory.vector_store import VectorStore
            self._vector_store = VectorStore(self._conn, self._lock)
            self._vector_store.migrate()
            log.debug("Vector store initialized")
        except Exception as exc:
            log.debug("Vector store init failed: %s", exc)
            self._vector_store = None

    # ── Session management ────────────────────────────────────────────────

    def new_session(self) -> int:
        """Start a new conversation session. Returns session id.

        If a previous session exists and meets the distillation threshold,
        fire-and-forget distillation of the old session (#118).
        """
        prev_session = self._session_id

        # Fire distillation of previous session (async, non-blocking)
        if prev_session is not None:
            self._fire_distillation(prev_session)

        now = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO conversations(started_at, last_active) VALUES (?,?)",
                (now, now),
            )
            self._session_id = cur.lastrowid
        return self._session_id

    def _fire_distillation(self, session_id: int) -> None:
        """Fire-and-forget distillation of a completed session."""
        try:
            from bantz.config import config
            if not config.distillation_enabled:
                return

            from bantz.memory.distiller import distill_session
            asyncio.ensure_future(
                distill_session(
                    self._conn,
                    self._lock,
                    session_id,
                    min_exchanges=config.distillation_min_exchanges,
                    embed=config.embedding_enabled,
                )
            )
            log.debug("Distillation fired for session %d", session_id)
        except Exception as exc:
            log.debug("Distillation fire failed: %s", exc)

    def resume_session(self, session_id: int) -> bool:
        """Resume an existing session by id. Returns False if not found."""
        row = self._conn.execute(
            "SELECT id FROM conversations WHERE id=?", (session_id,)
        ).fetchone()
        if row:
            self._session_id = session_id
            return True
        return False

    @property
    def session_id(self) -> Optional[int]:
        return self._session_id

    # ── Writing ───────────────────────────────────────────────────────────

    def add(
        self,
        role: str,
        content: str,
        tool_used: Optional[str] = None,
    ) -> int:
        """Save a message to the current session. Returns message id.

        If embeddings are enabled, queues the message for async embedding.
        """
        if not self._session_id:
            self.new_session()

        now = _now()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO messages(conversation_id, role, content, tool_used, created_at)
                   VALUES (?,?,?,?,?)""",
                (self._session_id, role, content, tool_used, now),
            )
            self._conn.execute(
                "UPDATE conversations SET last_active=? WHERE id=?",
                (now, self._session_id),
            )
        msg_id = cur.lastrowid

        # Queue for async embedding (processed in embed_pending)
        if role in ("user", "assistant") and len(content) > 10:
            self._embed_queue.append((msg_id, content))

        return msg_id

    # ── Reading ───────────────────────────────────────────────────────────

    def context(self, n: int = 12) -> list[dict]:
        """
        Return last n messages from current session as
        [{"role": "user"|"assistant", "content": "..."}]
        Ready to pass directly to Ollama messages list.
        """
        if not self._session_id:
            return []
        rows = self._conn.execute(
            """SELECT role, content FROM messages
               WHERE conversation_id=? AND role IN ('user','assistant')
               ORDER BY created_at DESC LIMIT ?""",
            (self._session_id, n),
        ).fetchall()
        # Return in chronological order
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def last_n(self, n: int = 20) -> list[dict]:
        """Return last n messages with full metadata (for UI history display)."""
        if not self._session_id:
            return []
        rows = self._conn.execute(
            """SELECT role, content, tool_used, created_at FROM messages
               WHERE conversation_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (self._session_id, n),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """
        Full-text search across ALL conversations.
        Falls back to LIKE if FTS5 is unavailable.
        """
        try:
            rows = self._conn.execute(
                """SELECT m.role, m.content, m.tool_used, m.created_at,
                          c.id as conv_id
                   FROM messages_fts f
                   JOIN messages m ON m.id = f.rowid
                   JOIN conversations c ON c.id = m.conversation_id
                   WHERE messages_fts MATCH ?
                   ORDER BY m.created_at DESC LIMIT ?""",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS fallback
            rows = self._conn.execute(
                """SELECT m.role, m.content, m.tool_used, m.created_at,
                          c.id as conv_id
                   FROM messages m
                   JOIN conversations c ON c.id = m.conversation_id
                   WHERE m.content LIKE ?
                   ORDER BY m.created_at DESC LIMIT ?""",
                (f"%{query}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_by_date(self, date: "datetime", limit: int = 20) -> list[dict]:
        """
        Return messages from a specific date (all conversations).
        Useful for "what did we do yesterday" type queries.
        """
        date_str = date.strftime("%Y-%m-%d")
        rows = self._conn.execute(
            """SELECT m.role, m.content, m.tool_used, m.created_at,
                      c.id as conv_id
               FROM messages m
               JOIN conversations c ON c.id = m.conversation_id
               WHERE m.created_at LIKE ?
               ORDER BY m.created_at ASC LIMIT ?""",
            (f"{date_str}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def conversation_list(self, limit: int = 20) -> list[dict]:
        """Return recent conversations with message count (for a history UI)."""
        rows = self._conn.execute(
            """SELECT c.id, c.started_at, c.last_active,
                      COUNT(m.id) as message_count,
                      (SELECT content FROM messages
                       WHERE conversation_id=c.id AND role='user'
                       ORDER BY created_at LIMIT 1) as first_message
               FROM conversations c
               LEFT JOIN messages m ON m.conversation_id=c.id
               GROUP BY c.id
               ORDER BY c.last_active DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Quick stats for --doctor."""
        total_conv = self._conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        total_msg = self._conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]
        session_msg = 0
        if self._session_id:
            session_msg = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id=?",
                (self._session_id,),
            ).fetchone()[0]
        return {
            "db_path": str(self._db_path),
            "total_conversations": total_conv,
            "total_messages": total_msg,
            "current_session_id": self._session_id,
            "current_session_messages": session_msg,
        }

    # ── Cleanup ───────────────────────────────────────────────────────────

    def prune(self, keep_days: int = 90) -> int:
        """Delete conversations older than keep_days. Returns deleted count."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM conversations WHERE last_active < ?", (cutoff,)
            )
        return cur.rowcount

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Vector / Semantic Search ──────────────────────────────────────────

    @property
    def vector_store(self):
        """Access the underlying VectorStore (or None if unavailable)."""
        return self._vector_store

    async def embed_pending(self) -> int:
        """Process the embed queue — call after each exchange.

        Embeds queued messages via Ollama and stores vectors.
        Returns the number of embeddings stored.
        Fire-and-forget safe — errors are logged but don't propagate.
        """
        if not self._vector_store or not self._embed_queue:
            return 0

        from bantz.config import config
        if not config.embedding_enabled:
            self._embed_queue.clear()
            return 0

        from bantz.memory.embeddings import embedder

        queue = self._embed_queue[:]
        self._embed_queue.clear()
        stored = 0

        for msg_id, content in queue:
            try:
                vec = await embedder.embed(content)
                if vec:
                    self._vector_store.store(msg_id, vec, model=embedder.model)
                    stored += 1
            except Exception as exc:
                log.debug("Embed msg %d failed: %s", msg_id, exc)

        if stored:
            log.debug("Embedded %d/%d messages", stored, len(queue))
        return stored

    async def semantic_search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> list[dict]:
        """Semantic search using vector cosine similarity.

        Embeds the query, then finds closest messages by meaning.
        Returns [] if embeddings are disabled or unavailable.
        """
        if not self._vector_store:
            return []

        from bantz.config import config
        if not config.embedding_enabled:
            return []

        from bantz.memory.embeddings import embedder

        query_vec = await embedder.embed(query)
        if not query_vec:
            return []

        return self._vector_store.search(query_vec, limit=limit, min_score=min_score)

    async def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        vector_weight: Optional[float] = None,
    ) -> list[dict]:
        """Hybrid search combining FTS5 lexical + vector semantic results.

        Merges and re-ranks results from both backends using a weighted score.
        ``vector_weight`` controls the blend (0.0 = pure FTS, 1.0 = pure vector).
        Defaults to ``config.vector_search_weight``.
        """
        from bantz.config import config
        if vector_weight is None:
            vector_weight = config.vector_search_weight
        fts_weight = 1.0 - vector_weight

        # Gather results from both backends
        fts_results = self.search(query, limit=limit * 2)
        sem_results = await self.semantic_search(query, limit=limit * 2, min_score=0.2)

        # Build a lookup by message content (since FTS doesn't have message_id easily)
        merged: dict[str, dict] = {}

        # FTS results — assign a normalized score based on position
        for i, r in enumerate(fts_results):
            key = r.get("content", "")[:200]
            fts_score = 1.0 - (i / max(len(fts_results), 1))
            merged[key] = {
                **r,
                "fts_score": fts_score,
                "vec_score": 0.0,
                "hybrid_score": fts_score * fts_weight,
                "source": "fts",
            }

        # Vector results — use actual cosine similarity
        for r in sem_results:
            key = r.get("content", "")[:200]
            vec_score = r.get("score", 0.0)
            if key in merged:
                # Both backends found it — boost score
                merged[key]["vec_score"] = vec_score
                merged[key]["hybrid_score"] = (
                    merged[key]["fts_score"] * fts_weight + vec_score * vector_weight
                )
                merged[key]["source"] = "both"
            else:
                merged[key] = {
                    **r,
                    "fts_score": 0.0,
                    "vec_score": vec_score,
                    "hybrid_score": vec_score * vector_weight,
                    "source": "vector",
                }

        # Sort by hybrid score descending
        ranked = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:limit]

    async def backfill_embeddings(self, batch_size: int = 50) -> int:
        """Backfill embeddings for messages that don't have them yet.

        Useful for upgrading existing databases.  Returns count embedded.
        """
        if not self._vector_store:
            return 0

        from bantz.config import config
        if not config.embedding_enabled:
            return 0

        from bantz.memory.embeddings import embedder

        unembedded = self._vector_store.unembedded_messages(limit=batch_size)
        if not unembedded:
            return 0

        stored = 0
        for msg in unembedded:
            try:
                vec = await embedder.embed(msg["content"])
                if vec:
                    self._vector_store.store(msg["id"], vec, model=embedder.model)
                    stored += 1
            except Exception as exc:
                log.debug("Backfill embed %d failed: %s", msg["id"], exc)

        log.info("Backfilled %d/%d embeddings", stored, len(unembedded))
        return stored

    def vector_stats(self) -> dict:
        """Vector store statistics for diagnostics."""
        if not self._vector_store:
            return {"enabled": False}
        return {
            "enabled": True,
            **self._vector_store.stats(),
        }

    # ── Distillation queries (#118) ───────────────────────────────────────

    async def search_distillations(
        self,
        query: str,
        limit: int = 3,
        min_score: float = 0.3,
    ) -> list[dict]:
        """Search past session distillations by semantic similarity."""
        if not self._conn:
            return []

        from bantz.config import config
        if not config.embedding_enabled or not config.distillation_enabled:
            return []

        try:
            from bantz.memory.embeddings import embedder
            query_vec = await embedder.embed(query)
            if not query_vec:
                return []
            from bantz.memory.distiller import search_distillations
            return search_distillations(self._conn, query_vec, limit, min_score)
        except Exception as exc:
            log.debug("Distillation search failed: %s", exc)
            return []

    def distillation_stats(self) -> dict:
        """Distillation statistics for diagnostics."""
        if not self._conn:
            return {"enabled": False}
        try:
            from bantz.memory.distiller import distillation_stats
            return {"enabled": True, **distillation_stats(self._conn)}
        except Exception:
            return {"enabled": False}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# Singleton — import this everywhere
memory = Memory()