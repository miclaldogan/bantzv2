"""Tests for paper-1 evaluation join view (patch 3).

Validates that:
  - log_finalizer_call now returns the inserted row id (int or None).
  - The paper1_eval view joins messages → hallucination_log → route_log
    by timestamp proximity.
  - Stream-mode finalizer rows join the same way as short-mode rows.
  - flagged_only and tool_filter narrow the result correctly.
  - stats() reports counts and mode distribution.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bantz.core import finalizer, route_log, eval_view
from bantz.core.memory import memory


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "bantz.db"
    memory.init(db)
    memory.new_session()
    yield db
    memory.close()


def _insert_message(role: str, content: str, tool_used: str | None = None,
                    created_at: str | None = None) -> int:
    """Insert a message at a specific timestamp (overrides default 'now')."""
    from bantz.data.connection_pool import get_pool
    ts = created_at or datetime.now().isoformat(timespec="milliseconds")
    with get_pool().connection(write=True) as conn:
        cur = conn.execute(
            "INSERT INTO messages(conversation_id, role, content, tool_used,"
            " created_at) VALUES (?,?,?,?,?)",
            (memory.session_id, role, content, tool_used, ts),
        )
        return cur.lastrowid


# ── log_finalizer_call now returns row id ───────────────────────────────────


def test_log_finalizer_call_returns_row_id(tmp_db):
    rid1 = finalizer.log_finalizer_call(
        user_input="a", tool_output="x", response="x",
        confidence=0.9, tool_used="shell", mode="short",
    )
    rid2 = finalizer.log_finalizer_call(
        user_input="b", tool_output="y", response="y",
        confidence=0.6, tool_used="gmail", mode="stream",
    )
    assert isinstance(rid1, int)
    assert isinstance(rid2, int)
    assert rid2 > rid1


def test_log_finalizer_call_returns_none_when_uninitialised():
    memory.close()
    rid = finalizer.log_finalizer_call(
        user_input="a", tool_output="x", response="x",
        confidence=0.9, tool_used="shell", mode="short",
    )
    assert rid is None


# ── view creation + basic join ──────────────────────────────────────────────


def test_ensure_view_creates_paper1_eval(tmp_db):
    assert eval_view.ensure_view() is True
    # second call is idempotent
    assert eval_view.ensure_view() is True


def test_ensure_view_returns_false_when_uninitialised():
    memory.close()
    assert eval_view.ensure_view() is False


def test_joined_row_pairs_assistant_message_with_finalizer(tmp_db):
    # Same-second timestamps simulate a real turn.
    ts = datetime.now().replace(microsecond=0).isoformat()
    route_log.log_route(
        en_input="weather in Istanbul", route="tool", tool_name="weather",
        confidence=0.92, reasoning="weather query",
        source="cot", attempt=1, outcome="ok",
    )
    finalizer.log_finalizer_call(
        user_input="weather in Istanbul",
        tool_output="22C sunny",
        response="22C sunny, ma'am.",
        confidence=0.95, tool_used="weather", mode="short",
    )
    _insert_message("user", "weather in Istanbul")
    _insert_message("assistant", "22C sunny, ma'am.", tool_used="weather")

    rows = list(eval_view.fetch_eval_rows())
    assert len(rows) == 1
    r = rows[0]
    assert r["response_text"] == "22C sunny, ma'am."
    assert r["finalizer_confidence"] == 0.95
    assert r["finalizer_mode"] == "short"
    assert r["route"] == "tool"
    assert r["route_tool"] == "weather"
    assert r["route_outcome"] == "ok"


def test_joined_row_works_for_stream_mode(tmp_db):
    route_log.log_route(
        en_input="summarize this long thing", route="tool", tool_name="summarizer",
        confidence=0.88, reasoning="summarization",
        source="cot", attempt=1, outcome="ok",
    )
    finalizer.log_finalizer_call(
        user_input="summarize this long thing",
        tool_output="x" * 1500,
        response="A brief precis follows, ma'am: ...",
        confidence=0.7, tool_used="summarizer", mode="stream",
    )
    _insert_message("assistant", "A brief precis follows, ma'am: ...",
                    tool_used="summarizer")

    rows = list(eval_view.fetch_eval_rows())
    assert len(rows) == 1
    assert rows[0]["finalizer_mode"] == "stream"
    assert rows[0]["finalizer_flagged"] == 1


def test_assistant_message_without_finalizer_row_still_appears(tmp_db):
    """A chat-route response has no finalizer log; the view must still
    include it (with NULL finalizer fields) so we can count un-finalized turns."""
    _insert_message("assistant", "Welcome, ma'am.")
    rows = list(eval_view.fetch_eval_rows())
    assert len(rows) == 1
    assert rows[0]["finalizer_id"] is None
    assert rows[0]["route_id"] is None


def test_user_messages_are_excluded(tmp_db):
    _insert_message("user", "hello")
    rows = list(eval_view.fetch_eval_rows())
    assert rows == []


# ── filter knobs ────────────────────────────────────────────────────────────


def test_flagged_only_filter(tmp_db):
    finalizer.log_finalizer_call(
        user_input="x", tool_output="x", response="hi",
        confidence=0.95, tool_used="shell", mode="short",
    )
    _insert_message("assistant", "hi", tool_used="shell")
    finalizer.log_finalizer_call(
        user_input="y", tool_output="y", response="bye",
        confidence=0.4, tool_used="shell", mode="short",
    )
    _insert_message("assistant", "bye", tool_used="shell")

    all_rows = list(eval_view.fetch_eval_rows())
    flagged_rows = list(eval_view.fetch_eval_rows(flagged_only=True))
    assert len(all_rows) == 2
    assert len(flagged_rows) == 1
    assert flagged_rows[0]["finalizer_confidence"] == 0.4


def test_tool_filter(tmp_db):
    finalizer.log_finalizer_call(
        user_input="x", tool_output="x", response="a",
        confidence=0.9, tool_used="weather", mode="short",
    )
    _insert_message("assistant", "a", tool_used="weather")
    finalizer.log_finalizer_call(
        user_input="y", tool_output="y", response="b",
        confidence=0.9, tool_used="gmail", mode="short",
    )
    _insert_message("assistant", "b", tool_used="gmail")

    weather_rows = list(eval_view.fetch_eval_rows(tool_filter="weather"))
    gmail_rows = list(eval_view.fetch_eval_rows(tool_filter="gmail"))
    assert len(weather_rows) == 1
    assert len(gmail_rows) == 1


def test_limit_caps_results(tmp_db):
    for i in range(5):
        finalizer.log_finalizer_call(
            user_input=f"q{i}", tool_output="x", response=f"a{i}",
            confidence=0.9, tool_used="shell", mode="short",
        )
        _insert_message("assistant", f"a{i}", tool_used="shell")

    capped = list(eval_view.fetch_eval_rows(limit=2))
    assert len(capped) == 2


# ── stats ───────────────────────────────────────────────────────────────────


def test_stats_reports_counts_and_modes(tmp_db):
    finalizer.log_finalizer_call(
        user_input="a", tool_output="a", response="x",
        confidence=0.95, tool_used="shell", mode="short",
    )
    _insert_message("assistant", "x", tool_used="shell")
    finalizer.log_finalizer_call(
        user_input="b", tool_output="b" * 1500, response="y",
        confidence=0.5, tool_used="shell", mode="stream",
    )
    _insert_message("assistant", "y", tool_used="shell")

    s = eval_view.stats()
    assert s["available"] is True
    assert s["assistant_messages"] == 2
    assert s["joined_to_finalizer"] == 2
    assert s["flagged_finalizer_rows"] == 1
    assert s["finalizer_mode_counts"] == {"short": 1, "stream": 1}


def test_stats_unavailable_when_uninitialised():
    memory.close()
    assert eval_view.stats() == {"available": False}
