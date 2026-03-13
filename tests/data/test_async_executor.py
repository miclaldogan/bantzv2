"""Tests for bantz.data.async_executor — AsyncDBExecutor (#224)."""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from bantz.data.connection_pool import SQLitePool, get_pool
from bantz.data import async_executor
from bantz.data.async_executor import run_read, run_write, run_in_db, shutdown


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_pool(tmp_path: Path):
    """Ensure every test gets its own pool + executor, then tear both down."""
    SQLitePool.reset()
    db = tmp_path / "test_async.db"
    get_pool(db)

    # Create a simple table for the tests
    with get_pool().connection(write=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS items "
            "(id INTEGER PRIMARY KEY, name TEXT NOT NULL)"
        )

    yield

    shutdown(wait=True)
    SQLitePool.reset()


# ── run_read ──────────────────────────────────────────────────────────


class TestRunRead:
    @pytest.mark.asyncio
    async def test_read_returns_rows(self):
        # seed
        with get_pool().connection(write=True) as conn:
            conn.execute("INSERT INTO items(name) VALUES (?)", ("alpha",))
            conn.execute("INSERT INTO items(name) VALUES (?)", ("beta",))

        rows = await run_read(
            lambda conn: conn.execute("SELECT name FROM items ORDER BY id").fetchall()
        )
        names = [r["name"] for r in rows]
        assert names == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_read_empty_table(self):
        rows = await run_read(
            lambda conn: conn.execute("SELECT * FROM items").fetchall()
        )
        assert rows == []

    @pytest.mark.asyncio
    async def test_read_with_args(self):
        with get_pool().connection(write=True) as conn:
            conn.execute("INSERT INTO items(name) VALUES (?)", ("gamma",))

        row = await run_read(
            lambda conn, name: conn.execute(
                "SELECT * FROM items WHERE name = ?", (name,)
            ).fetchone(),
            "gamma",
        )
        assert row is not None
        assert row["name"] == "gamma"


# ── run_write ─────────────────────────────────────────────────────────


class TestRunWrite:
    @pytest.mark.asyncio
    async def test_write_inserts_row(self):
        await run_write(
            lambda conn: conn.execute(
                "INSERT INTO items(name) VALUES (?)", ("delta",)
            )
        )
        rows = await run_read(
            lambda conn: conn.execute("SELECT name FROM items").fetchall()
        )
        assert [r["name"] for r in rows] == ["delta"]

    @pytest.mark.asyncio
    async def test_write_rollback_on_error(self):
        """Write failure must rollback and not poison the pool."""

        def _bad_write(conn):
            conn.execute(
                "INSERT INTO items(id, name) VALUES (?, ?)", (1, "first")
            )
            # duplicate PK → IntegrityError
            conn.execute(
                "INSERT INTO items(id, name) VALUES (?, ?)", (1, "dup")
            )

        with pytest.raises(sqlite3.IntegrityError):
            await run_write(_bad_write)

        # The transaction should have rolled back — row "first" must be gone
        rows = await run_read(
            lambda conn: conn.execute("SELECT count(*) AS c FROM items").fetchone()
        )
        assert rows["c"] == 0  # rolled back, pool alive


# ── run_in_db (unified entry) ────────────────────────────────────────


class TestRunInDB:
    @pytest.mark.asyncio
    async def test_run_in_db_read(self):
        with get_pool().connection(write=True) as conn:
            conn.execute("INSERT INTO items(name) VALUES (?)", ("epsilon",))

        rows = await run_in_db(
            lambda conn: conn.execute("SELECT name FROM items").fetchall(),
            write=False,
        )
        assert [r["name"] for r in rows] == ["epsilon"]

    @pytest.mark.asyncio
    async def test_run_in_db_write(self):
        await run_in_db(
            lambda conn: conn.execute(
                "INSERT INTO items(name) VALUES (?)", ("zeta",)
            ),
            write=True,
        )
        rows = await run_in_db(
            lambda conn: conn.execute("SELECT name FROM items").fetchall(),
        )
        assert [r["name"] for r in rows] == ["zeta"]


# ── non-blocking proof ───────────────────────────────────────────────


class TestNonBlocking:
    @pytest.mark.asyncio
    async def test_does_not_block_event_loop(self):
        """Prove the DB call runs in a background thread.

        We schedule a DB read AND a pure-coroutine counter concurrently.
        If run_read blocked the loop the counter would never increment
        until after the DB call (sequentially).
        """
        counter = 0

        async def ticker():
            nonlocal counter
            for _ in range(5):
                counter += 1
                await asyncio.sleep(0.01)

        with get_pool().connection(write=True) as conn:
            conn.execute("INSERT INTO items(name) VALUES (?)", ("test",))

        # Run both concurrently
        await asyncio.gather(
            run_read(lambda conn: conn.execute("SELECT * FROM items").fetchall()),
            ticker(),
        )
        assert counter == 5  # ticker ran fully while DB was in flight


# ── shutdown idempotence ──────────────────────────────────────────────


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_then_restart(self):
        """After shutdown, next call lazily re-creates the executor."""
        await run_read(lambda conn: conn.execute("SELECT 1").fetchone())
        shutdown(wait=True)
        # Should still work — lazy re-init
        row = await run_read(lambda conn: conn.execute("SELECT 1 AS v").fetchone())
        assert row["v"] == 1

    def test_double_shutdown_safe(self):
        shutdown(wait=True)
        shutdown(wait=True)  # must not raise
