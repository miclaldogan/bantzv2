"""
Bantz v3 — SQLite Store Implementations

Concrete SQLite-backed stores for conversations and reminders.
Schema matches the existing v2 tables — zero-migration upgrade.

These are standalone implementations usable in tests and migration
scripts without importing the full ``core/`` layer.  At runtime the
DataLayer wires the existing ``Memory`` and ``Scheduler`` singletons
(which inherit from these ABCs) instead.

All database access goes through the central ``SQLitePool`` from
``connection_pool.py`` — no store class opens its own connection.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from bantz.data.connection_pool import get_pool
from bantz.data.store import (
    ConversationStore,
    PlaceStore,
    ProfileStore,
    ReminderStore,
    ScheduleStore,
    SessionStore,
)

log = logging.getLogger("bantz.data.sqlite")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ━━ Conversation Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SQLiteConversationStore(ConversationStore):
    """SQLite conversation + message store, compatible with v2 schema."""

    def __init__(self) -> None:
        self._db_path: Optional[Path] = None
        self._session_id: Optional[int] = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def init(self, db_path: Path) -> None:
        self._db_path = db_path
        get_pool(db_path)          # ensure singleton is ready
        self._migrate()

    def _migrate(self) -> None:
        with get_pool().connection(write=True) as c:
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
            # FTS5 for full-text search — graceful fallback if unavailable
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
                pass  # FTS5 not compiled in

    # ── session management ────────────────────────────────────────────────

    def new_session(self) -> int:
        now = _now()
        with get_pool().connection(write=True) as conn:
            cur = conn.execute(
                "INSERT INTO conversations(started_at, last_active) VALUES (?,?)",
                (now, now),
            )
            self._session_id = cur.lastrowid
        return self._session_id

    def resume_session(self, session_id: int) -> bool:
        with get_pool().connection() as conn:
            row = conn.execute(
                "SELECT id FROM conversations WHERE id=?", (session_id,)
            ).fetchone()
        if row:
            self._session_id = session_id
            return True
        return False

    @property
    def session_id(self) -> Optional[int]:
        return self._session_id

    # ── writing ───────────────────────────────────────────────────────────

    def add(self, role: str, content: str, tool_used: Optional[str] = None) -> int:
        if not self._session_id:
            self.new_session()
        now = _now()
        with get_pool().connection(write=True) as conn:
            cur = conn.execute(
                """INSERT INTO messages(conversation_id, role, content, tool_used, created_at)
                   VALUES (?,?,?,?,?)""",
                (self._session_id, role, content, tool_used, now),
            )
            conn.execute(
                "UPDATE conversations SET last_active=? WHERE id=?",
                (now, self._session_id),
            )
            lastrowid = cur.lastrowid
        return lastrowid

    # ── reading ───────────────────────────────────────────────────────────

    def context(self, n: int = 12) -> list[dict]:
        if not self._session_id:
            return []
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT role, content FROM messages
                   WHERE conversation_id=? AND role IN ('user','assistant')
                   ORDER BY created_at DESC LIMIT ?""",
                (self._session_id, n),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def last_n(self, n: int = 20) -> list[dict]:
        if not self._session_id:
            return []
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT role, content, tool_used, created_at FROM messages
                   WHERE conversation_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (self._session_id, n),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        with get_pool().connection() as conn:
            try:
                rows = conn.execute(
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
                rows = conn.execute(
                    """SELECT m.role, m.content, m.tool_used, m.created_at,
                              c.id as conv_id
                       FROM messages m
                       JOIN conversations c ON c.id = m.conversation_id
                       WHERE m.content LIKE ?
                       ORDER BY m.created_at DESC LIMIT ?""",
                    (f"%{query}%", limit),
                ).fetchall()
        return [dict(r) for r in rows]

    def search_by_date(self, date: datetime, limit: int = 20) -> list[dict]:
        date_str = date.strftime("%Y-%m-%d")
        with get_pool().connection() as conn:
            rows = conn.execute(
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
        with get_pool().connection() as conn:
            rows = conn.execute(
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

    # ── diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        with get_pool().connection() as conn:
            total_conv = conn.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
            total_msg = conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
            session_msg = 0
            if self._session_id:
                session_msg = conn.execute(
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

    def prune(self, keep_days: int = 90) -> int:
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()
        with get_pool().connection(write=True) as conn:
            cur = conn.execute(
                "DELETE FROM conversations WHERE last_active < ?", (cutoff,)
            )
            rowcount = cur.rowcount
        return rowcount

    def close(self) -> None:
        # Pool manages connections — nothing to close here.
        pass


# ━━ Reminder Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_REPEAT_MODES = ("none", "daily", "weekly", "weekdays", "custom")


class SQLiteReminderStore(ReminderStore):
    """SQLite reminder store, compatible with v2 schema."""

    def __init__(self) -> None:
        self._db_path: Optional[Path] = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def init(self, db_path: Path) -> None:
        self._db_path = db_path
        get_pool(db_path)
        self._migrate()
        log.debug("SQLiteReminderStore initialized: %s", db_path)

    def _migrate(self) -> None:
        with get_pool().connection(write=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    title            TEXT NOT NULL,
                    fire_at          TEXT NOT NULL,
                    repeat           TEXT NOT NULL DEFAULT 'none',
                    repeat_interval  INTEGER DEFAULT 0,
                    created_at       TEXT NOT NULL,
                    fired            INTEGER NOT NULL DEFAULT 0,
                    snoozed_until    TEXT,
                    trigger_place    TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reminders_fire
                    ON reminders(fire_at, fired)
            """)
            try:
                conn.execute("ALTER TABLE reminders ADD COLUMN trigger_place TEXT")
            except Exception:
                pass  # column already exists

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add(
        self,
        title: str,
        fire_at: datetime,
        repeat: str = "none",
        repeat_interval: int = 0,
        trigger_place: Optional[str] = None,
    ) -> int:
        if repeat not in _REPEAT_MODES:
            repeat = "none"
        with get_pool().connection(write=True) as conn:
            cur = conn.execute(
                """INSERT INTO reminders
                       (title, fire_at, repeat, repeat_interval, created_at, trigger_place)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    title,
                    fire_at.isoformat(),
                    repeat,
                    repeat_interval,
                    datetime.now().isoformat(),
                    trigger_place,
                ),
            )
            return cur.lastrowid

    def check_due(self) -> list[dict]:
        now_iso = datetime.now().isoformat()
        with get_pool().connection(write=True) as conn:
            rows = conn.execute(
                """SELECT * FROM reminders
                   WHERE fired = 0
                     AND fire_at <= ?
                     AND (snoozed_until IS NULL OR snoozed_until <= ?)
                   ORDER BY fire_at""",
                (now_iso, now_iso),
            ).fetchall()
            due = []
            one_offs = []
            repeating = []
            for row in rows:
                item = dict(row)
                due.append(item)
                repeat = item["repeat"]
                if repeat == "none":
                    one_offs.append((item["id"],))
                else:
                    next_fire = self._next_occurrence(
                        datetime.fromisoformat(item["fire_at"]),
                        repeat,
                        item["repeat_interval"],
                    )
                    repeating.append((next_fire.isoformat(), item["id"]))

            # ⚡ Bolt: Replace N+1 queries with bulk operations
            if one_offs:
                conn.executemany(
                    "UPDATE reminders SET fired = 1 WHERE id = ?",
                    one_offs,
                )
            if repeating:
                conn.executemany(
                    "UPDATE reminders SET fire_at = ?, snoozed_until = NULL WHERE id = ?",
                    repeating,
                )
            return due

    def list_upcoming(self, limit: int = 10) -> list[dict]:
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT * FROM reminders WHERE fired = 0
                   ORDER BY fire_at LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_all(self, limit: int = 20) -> list[dict]:
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT * FROM reminders ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def cancel(self, reminder_id: int) -> bool:
        with get_pool().connection(write=True) as conn:
            cur = conn.execute(
                "DELETE FROM reminders WHERE id = ?", (reminder_id,)
            )
            return cur.rowcount > 0

    def cancel_by_title(self, title: str) -> int:
        with get_pool().connection(write=True) as conn:
            cur = conn.execute(
                "DELETE FROM reminders WHERE lower(title) = lower(?)", (title,)
            )
            return cur.rowcount

    def snooze(self, reminder_id: int, minutes: int = 10) -> bool:
        until = datetime.now() + timedelta(minutes=minutes)
        with get_pool().connection(write=True) as conn:
            cur = conn.execute(
                "UPDATE reminders SET snoozed_until = ?, fired = 0 WHERE id = ?",
                (until.isoformat(), reminder_id),
            )
            return cur.rowcount > 0

    def due_today(self) -> list[dict]:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT * FROM reminders
                   WHERE fired = 0 AND fire_at >= ? AND fire_at < ?
                   ORDER BY fire_at""",
                (today_start.isoformat(), today_end.isoformat()),
            ).fetchall()
        return [dict(r) for r in rows]

    def check_place_due(self, place_key: str) -> list[dict]:
        with get_pool().connection(write=True) as conn:
            rows = conn.execute(
                """SELECT * FROM reminders
                   WHERE fired = 0 AND trigger_place = ?
                     AND (snoozed_until IS NULL OR snoozed_until <= ?)
                   ORDER BY created_at""",
                (place_key, datetime.now().isoformat()),
            ).fetchall()
            due = []
            one_offs = []
            for row in rows:
                item = dict(row)
                due.append(item)
                if item["repeat"] == "none":
                    one_offs.append((item["id"],))

            # ⚡ Bolt: Replace N+1 queries with bulk operations
            if one_offs:
                conn.executemany(
                    "UPDATE reminders SET fired = 1 WHERE id = ?",
                    one_offs,
                )
            return due

    def count_active(self) -> int:
        with get_pool().connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM reminders WHERE fired = 0"
            ).fetchone()
        return row[0] if row else 0

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _next_occurrence(current: datetime, repeat: str, interval: int) -> datetime:
        if repeat == "daily":
            return current + timedelta(days=1)
        elif repeat == "weekly":
            return current + timedelta(weeks=1)
        elif repeat == "weekdays":
            nxt = current + timedelta(days=1)
            while nxt.weekday() >= 5:
                nxt += timedelta(days=1)
            return nxt
        elif repeat == "custom" and interval > 0:
            return current + timedelta(seconds=interval)
        return current + timedelta(days=1)


# ━━ Profile Store (SQLite) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SQLiteProfileStore(ProfileStore):
    """SQLite-backed profile store — key/value table for user identity."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        get_pool(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        with get_pool().connection(write=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def load(self) -> dict[str, Any]:
        with get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM user_profile"
            ).fetchall()
        result: dict[str, Any] = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                result[row["key"]] = row["value"]
        return result

    def save(self, data: dict[str, Any]) -> None:
        now = _now()
        with get_pool().connection(write=True) as conn:
            conn.execute("DELETE FROM user_profile")
            # ⚡ Bolt: Replace N+1 queries with bulk operations
            conn.executemany(
                "INSERT INTO user_profile(key, value, updated_at) VALUES (?,?,?)",
                [(k, json.dumps(v, ensure_ascii=False), now) for k, v in data.items()],
            )

    def exists(self) -> bool:
        with get_pool().connection() as conn:
            # ⚡ Bolt: Optimize existence check by avoiding full table scan using SELECT 1 ... LIMIT 1
            row = conn.execute(
                "SELECT 1 FROM user_profile LIMIT 1"
            ).fetchone()
        return row is not None

    @property
    def path(self) -> Path:
        return self._db_path


# ━━ Place Store (SQLite) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SQLitePlaceStore(PlaceStore):
    """SQLite-backed place store — named GPS locations."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        get_pool(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        with get_pool().connection(write=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS places (
                    key    TEXT PRIMARY KEY,
                    label  TEXT NOT NULL,
                    lat    REAL NOT NULL DEFAULT 0.0,
                    lon    REAL NOT NULL DEFAULT 0.0,
                    radius REAL NOT NULL DEFAULT 100.0
                )
            """)

    def load_all(self) -> dict[str, dict]:
        with get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT key, label, lat, lon, radius FROM places"
            ).fetchall()
        return {
            row["key"]: {
                "label": row["label"],
                "lat": row["lat"],
                "lon": row["lon"],
                "radius": row["radius"],
            }
            for row in rows
        }

    def save_all(self, data: dict[str, dict]) -> None:
        with get_pool().connection(write=True) as conn:
            conn.execute("DELETE FROM places")
            # ⚡ Bolt: Replace N+1 queries with bulk operations
            conn.executemany(
                "INSERT INTO places(key, label, lat, lon, radius) VALUES (?,?,?,?,?)",
                [
                    (
                        key,
                        place.get("label", key),
                        place.get("lat", 0.0),
                        place.get("lon", 0.0),
                        place.get("radius", 100.0),
                    )
                    for key, place in data.items()
                ],
            )

    def upsert(self, key: str, place: dict) -> None:
        """Insert or update a single place."""
        with get_pool().connection(write=True) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO places(key, label, lat, lon, radius)
                   VALUES (?,?,?,?,?)""",
                (
                    key,
                    place.get("label", key),
                    place.get("lat", 0.0),
                    place.get("lon", 0.0),
                    place.get("radius", 100.0),
                ),
            )

    def delete(self, key: str) -> bool:
        """Delete a single place. Returns True if it existed."""
        with get_pool().connection(write=True) as conn:
            cur = conn.execute("DELETE FROM places WHERE key = ?", (key,))
            return cur.rowcount > 0

    def exists(self) -> bool:
        with get_pool().connection() as conn:
            # ⚡ Bolt: Optimize existence check by avoiding full table scan using SELECT 1 ... LIMIT 1
            row = conn.execute(
                "SELECT 1 FROM places LIMIT 1"
            ).fetchone()
        return row is not None

    @property
    def path(self) -> Path:
        return self._db_path


# ━━ Schedule Store (SQLite) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SQLiteScheduleStore(ScheduleStore):
    """SQLite-backed schedule store — weekly timetable."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        get_pool(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        with get_pool().connection(write=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schedule_entries (
                    day      TEXT NOT NULL,
                    idx      INTEGER NOT NULL,
                    name     TEXT NOT NULL,
                    time     TEXT NOT NULL,
                    duration INTEGER NOT NULL DEFAULT 60,
                    location TEXT DEFAULT '',
                    type     TEXT DEFAULT '',
                    PRIMARY KEY (day, idx)
                )
            """)

    def load(self) -> dict[str, list[dict]]:
        with get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT day, idx, name, time, duration, location, type "
                "FROM schedule_entries ORDER BY day, idx"
            ).fetchall()
        result: dict[str, list[dict]] = {}
        for row in rows:
            entry = {
                "name": row["name"],
                "time": row["time"],
                "duration": row["duration"],
                "location": row["location"],
                "type": row["type"],
            }
            result.setdefault(row["day"], []).append(entry)
        return result

    def save(self, data: dict[str, list[dict]]) -> None:
        with get_pool().connection(write=True) as conn:
            conn.execute("DELETE FROM schedule_entries")
            # ⚡ Bolt: Replace N+1 queries with bulk operations
            conn.executemany(
                """INSERT INTO schedule_entries
                   (day, idx, name, time, duration, location, type)
                   VALUES (?,?,?,?,?,?,?)""",
                [
                    (
                        day, idx,
                        entry.get("name", ""),
                        entry.get("time", ""),
                        entry.get("duration", 60),
                        entry.get("location", ""),
                        entry.get("type", ""),
                    )
                    for day, entries in data.items()
                    for idx, entry in enumerate(entries)
                ],
            )

    def exists(self) -> bool:
        with get_pool().connection() as conn:
            # ⚡ Bolt: Optimize existence check by avoiding full table scan using SELECT 1 ... LIMIT 1
            row = conn.execute(
                "SELECT 1 FROM schedule_entries LIMIT 1"
            ).fetchone()
        return row is not None

    @property
    def path(self) -> Path:
        return self._db_path


# ━━ Session Store (SQLite) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SQLiteSessionStore(SessionStore):
    """SQLite-backed session state — launch tracking key/value."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        get_pool(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        with get_pool().connection(write=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_state (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def load(self) -> dict:
        with get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM session_state"
            ).fetchall()
        result: dict = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                result[row["key"]] = row["value"]
        return result

    def save(self, data: dict) -> None:
        now = _now()
        with get_pool().connection(write=True) as conn:
            conn.execute("DELETE FROM session_state")
            # ⚡ Bolt: Replace N+1 queries with bulk operations
            conn.executemany(
                "INSERT INTO session_state(key, value, updated_at) VALUES (?,?,?)",
                [(k, json.dumps(v, ensure_ascii=False), now) for k, v in data.items()],
            )

    @property
    def path(self) -> Path:
        return self._db_path


# ━━ Key-Value Store (SQLite) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SQLiteKVStore:
    """Simple key-value store backed by SQLite.

    Used for lightweight state like briefing cache, feature flags, etc.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        get_pool(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        with get_pool().connection(write=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def get(self, key: str, default: str = "") -> str:
        with get_pool().connection() as conn:
            row = conn.execute(
                "SELECT value FROM kv_store WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        now = _now()
        with get_pool().connection(write=True) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv_store(key, value, updated_at) VALUES (?,?,?)",
                (key, value, now),
            )

    def delete(self, key: str) -> None:
        with get_pool().connection(write=True) as conn:
            conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))

    def all(self) -> dict[str, str]:
        with get_pool().connection() as conn:
            rows = conn.execute("SELECT key, value FROM kv_store").fetchall()
        return {row["key"]: row["value"] for row in rows}

    @property
    def path(self) -> Path:
        return self._db_path
