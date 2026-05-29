"""Tests for the paper-1 uncensored finalizer logging (patch 1).

Validates that:
  - log_finalizer_call writes one row per call regardless of confidence.
  - The schema includes mode + flagged columns.
  - flagged is set to 1 only when confidence < HALLUCINATION_FLAG_THRESHOLD.
  - finalize_stream's wrapper accumulates tokens and logs on close.
  - The legacy log_hallucination alias still works.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from bantz.core import finalizer
from bantz.core.memory import memory
from bantz.tools import ToolResult


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Initialise the SQLite memory store at a temp path."""
    db = tmp_path / "bantz.db"
    memory.init(db)
    memory.new_session()
    yield db
    memory.close()


def _fetch_rows() -> list[dict]:
    import sqlite3
    from bantz.data.connection_pool import get_pool
    with get_pool().connection() as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT confidence, mode, flagged, tool_used, response"
                " FROM hallucination_log ORDER BY id"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(r) for r in rows]


def test_log_finalizer_call_writes_high_confidence_row(tmp_db):
    """A confidence=1.0 call must produce a row with flagged=0."""
    finalizer.log_finalizer_call(
        user_input="ping",
        tool_output="pong",
        response="pong",
        confidence=1.0,
        tool_used="weather",
        mode="short",
    )
    rows = _fetch_rows()
    assert len(rows) == 1
    assert rows[0]["confidence"] == 1.0
    assert rows[0]["flagged"] == 0
    assert rows[0]["mode"] == "short"


def test_log_finalizer_call_flags_low_confidence(tmp_db):
    """A confidence below 0.8 must produce a row with flagged=1."""
    finalizer.log_finalizer_call(
        user_input="check mail",
        tool_output="",
        response="fake@example.com sent you a draft",
        confidence=0.55,
        tool_used="gmail",
        mode="short",
    )
    rows = _fetch_rows()
    assert len(rows) == 1
    assert rows[0]["flagged"] == 1


def test_log_finalizer_call_distinguishes_modes(tmp_db):
    """short and stream rows both land in the table with correct mode tags."""
    finalizer.log_finalizer_call(
        user_input="a", tool_output="x", response="x",
        confidence=0.9, tool_used="shell", mode="short",
    )
    finalizer.log_finalizer_call(
        user_input="b", tool_output="x", response="x",
        confidence=0.9, tool_used="shell", mode="stream",
    )
    rows = _fetch_rows()
    assert [r["mode"] for r in rows] == ["short", "stream"]


def test_log_hallucination_alias_still_works(tmp_db):
    """Backward-compat: log_hallucination is an alias for log_finalizer_call."""
    assert finalizer.log_hallucination is finalizer.log_finalizer_call
    finalizer.log_hallucination(
        user_input="x", tool_output="y", response="y",
        confidence=0.5, tool_used="shell",
    )
    rows = _fetch_rows()
    assert len(rows) == 1
    assert rows[0]["flagged"] == 1


def test_legacy_schema_is_migrated(tmp_db):
    """An old hallucination_log without mode/flagged columns gets ALTERed in."""
    from bantz.data.connection_pool import get_pool

    # Recreate the legacy schema (no mode/flagged columns).
    with get_pool().connection(write=True) as conn:
        conn.execute("DROP TABLE IF EXISTS hallucination_log")
        conn.execute(
            "CREATE TABLE hallucination_log ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  timestamp TEXT NOT NULL,"
            "  user_input TEXT,"
            "  tool_used TEXT,"
            "  tool_output TEXT,"
            "  response TEXT,"
            "  confidence REAL NOT NULL"
            ")"
        )

    finalizer.log_finalizer_call(
        user_input="x", tool_output="y", response="y",
        confidence=0.95, tool_used="shell", mode="short",
    )
    rows = _fetch_rows()
    assert len(rows) == 1
    assert rows[0]["mode"] == "short"
    assert rows[0]["flagged"] == 0


async def _drain(agen):
    chunks = []
    async for tok in agen:
        chunks.append(tok)
    return "".join(chunks)


def test_finalize_stream_logs_on_close(tmp_db):
    """The streaming finalizer must log a 'stream' row after the iterator drains."""
    tc = {"prompt_hint": ""}
    # Tool output long enough (>= 800 chars) to trigger the LLM path.
    long_output = "x " * 500
    result = ToolResult(success=True, output=long_output)

    async def fake_stream(_messages, **__):
        for tok in ("hello ", "world"):
            yield tok

    with patch("bantz.llm.ollama.ollama.chat_stream", fake_stream):
        agen = asyncio.run(finalizer.finalize_stream(
            "user query", result, tc,
        ))
        assert agen is not None
        out = asyncio.run(_drain(agen))

    assert out == "hello world"
    rows = _fetch_rows()
    assert len(rows) == 1
    assert rows[0]["mode"] == "stream"
    assert rows[0]["response"].startswith("hello world")


def test_finalize_stream_skips_short_output(tmp_db):
    """Short tool output (<800 chars) bypasses the LLM and logs nothing."""
    result = ToolResult(success=True, output="short")
    agen = asyncio.run(finalizer.finalize_stream(
        "user query", result, {"prompt_hint": ""},
    ))
    assert agen is None
    assert _fetch_rows() == []
