"""
Bantz — Central SQLite Connection Pool (#222)

Singleton pool enforcing:
  - Single-writer via application-level ``_write_lock`` (threading.Lock)
  - Auto-commit / auto-rollback via ``with conn:`` context manager
  - WAL mode + PRAGMA optimisations on every connection

Usage::

    from bantz.data.connection_pool import get_pool

    # Read — no write-lock, any pooled connection
    with get_pool().connection() as conn:
        rows = conn.execute("SELECT * FROM foo").fetchall()

    # Write — acquires _write_lock, serialises all writers
    with get_pool().connection(write=True) as conn:
        conn.execute("INSERT INTO foo VALUES (?)", (bar,))

Closes #208 (merged into #222).
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

log = logging.getLogger("bantz.data.pool")


class SQLitePool:
    """Central SQLite connection pool — single writer, multiple readers.

    Connections are pre-created with WAL mode, NORMAL synchronous,
    ``busy_timeout``, and foreign-key enforcement.

    ``connection(write=True)`` acquires an application-level mutex so
    that **only one thread** can hold a write transaction at any time.
    SQLite allows unlimited concurrent readers under WAL, but only a
    single writer — this lock makes that guarantee explicit and avoids
    the 5-second ``busy_timeout`` casino entirely.
    """

    _instance: Optional["SQLitePool"] = None
    _init_lock = threading.Lock()

    def __init__(self, db_path: Path, max_connections: int = 5) -> None:
        self._db_path = Path(db_path).resolve()
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(
            maxsize=max_connections,
        )
        self._write_lock = threading.Lock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        for _ in range(max_connections):
            conn = sqlite3.connect(
                str(self._db_path), check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._pool.put(conn)

        log.info(
            "SQLitePool ready  db=%s  connections=%d",
            self._db_path,
            max_connections,
        )

    # ── public API ────────────────────────────────────────────────────────

    @contextmanager
    def connection(self, write: bool = False):
        """Borrow a connection from the pool.

        Parameters
        ----------
        write : bool
            If *True*, acquires ``_write_lock`` first so that all writers
            across the entire application are serialised.  Readers (the
            default) pass through without the write-lock.

        The inner ``with conn:`` block guarantees:
          - **success** → ``conn.commit()``
          - **exception** → ``conn.rollback()``

        No dirty connections ever return to the pool.
        """
        conn: sqlite3.Connection = self._pool.get(timeout=10)
        locked = False
        try:
            if write:
                self._write_lock.acquire()
                locked = True

            with conn:  # auto-commit / auto-rollback
                yield conn
        finally:
            if locked:
                self._write_lock.release()
            self._pool.put(conn)

    # ── singleton management ──────────────────────────────────────────────

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "SQLitePool":
        """Return (or create) the singleton pool.

        Thread-safe.  If *db_path* differs from the running instance the
        old pool is torn down first (happens in tests with ``tmp_path``).
        """
        with cls._init_lock:
            if cls._instance is not None:
                if db_path is not None:
                    new = Path(db_path).resolve()
                    if cls._instance._db_path != new:
                        cls._instance._close_all()
                        cls._instance = cls(new)
                return cls._instance

            if db_path is None:
                raise RuntimeError(
                    "SQLitePool not yet initialised — pass db_path on first call"
                )
            cls._instance = cls(Path(db_path))
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton.  Intended for test teardown."""
        with cls._init_lock:
            if cls._instance is not None:
                cls._instance._close_all()
                cls._instance = None

    # ── internals ─────────────────────────────────────────────────────────

    def _close_all(self) -> None:
        while not self._pool.empty():
            try:
                c = self._pool.get_nowait()
                c.close()
            except queue.Empty:
                break
        log.info("SQLitePool closed  db=%s", self._db_path)

    @property
    def db_path(self) -> Path:
        return self._db_path


# ── convenience accessor ──────────────────────────────────────────────────


def get_pool(db_path: Optional[Path] = None) -> SQLitePool:
    """Shorthand for ``SQLitePool.get_instance(db_path)``."""
    return SQLitePool.get_instance(db_path)
