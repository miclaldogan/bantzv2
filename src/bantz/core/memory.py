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

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class Memory:
    def __init__(self) -> None:
        self._db_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._session_id: Optional[int] = None

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

    # ── Session management ────────────────────────────────────────────────

    def new_session(self) -> int:
        """Start a new conversation session. Returns session id."""
        now = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO conversations(started_at, last_active) VALUES (?,?)",
                (now, now),
            )
            self._session_id = cur.lastrowid
        return self._session_id

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
        """Save a message to the current session. Returns message id."""
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
        return cur.lastrowid

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


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# Singleton — import this everywhere
memory = Memory()