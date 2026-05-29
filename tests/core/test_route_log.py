"""Tests for paper-1 routing decision log (patch 2).

Validates that:
  - log_route() writes a row and returns its id.
  - Schema is created lazily on first call and survives reuse.
  - All cot_route return paths fire a log_route call with the right
    (source, attempt, outcome) tuple.
  - The thinking-block extractor handles missing / malformed / valid blocks.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

from bantz.core import route_log, intent
from bantz.core.memory import memory


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "bantz.db"
    memory.init(db)
    memory.new_session()
    yield db
    memory.close()


def _rows() -> list[dict]:
    from bantz.data.connection_pool import get_pool
    with get_pool().connection() as conn:
        conn.row_factory = sqlite3.Row
        try:
            rs = conn.execute(
                "SELECT source, attempt, outcome, route, tool_name,"
                " confidence, error FROM route_log ORDER BY id"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(r) for r in rs]


def test_log_route_writes_row_returns_id(tmp_db):
    rid = route_log.log_route(
        en_input="hello",
        route="chat",
        tool_name=None,
        confidence=0.95,
        reasoning="greeting",
        source="fast_chat",
        attempt=0,
        outcome="ok",
    )
    assert isinstance(rid, int) and rid > 0
    rs = _rows()
    assert len(rs) == 1
    assert rs[0]["source"] == "fast_chat"
    assert rs[0]["attempt"] == 0
    assert rs[0]["outcome"] == "ok"
    assert rs[0]["route"] == "chat"
    assert rs[0]["confidence"] == 0.95


def test_log_route_swallows_errors_when_db_uninitialised():
    """If memory isn't initialised, log_route must return None silently."""
    memory.close()
    rid = route_log.log_route(
        en_input="x", route=None, tool_name=None, confidence=None,
        reasoning=None, source="cot", attempt=1, outcome="parse_error",
    )
    assert rid is None


def test_extract_thinking_handles_missing_block():
    assert route_log._extract_thinking("") == ""
    assert route_log._extract_thinking("no tags here") == ""


def test_extract_thinking_extracts_inner_text():
    raw = "<thinking>Step 1: weather query.</thinking>\n{\"route\":\"tool\"}"
    assert route_log._extract_thinking(raw) == "Step 1: weather query."


def test_extract_thinking_truncates_long_blocks():
    long_body = "a" * 5000
    out = route_log._extract_thinking(f"<thinking>{long_body}</thinking>")
    assert len(out) <= 4000


async def test_cot_route_fast_reminder_logs(tmp_db):
    plan, err = await intent.cot_route(
        "remind me in 5 minutes to drink water",
        tool_schemas=[],
    )
    assert plan is not None and plan["tool_name"] == "reminder"
    rs = _rows()
    assert len(rs) == 1
    assert rs[0]["source"] == "fast_reminder"
    assert rs[0]["attempt"] == 0
    assert rs[0]["outcome"] == "ok"


async def test_cot_route_fast_chat_logs(tmp_db):
    plan, err = await intent.cot_route(
        "I feel exhausted today",
        tool_schemas=[],
    )
    assert plan is not None and plan["route"] == "chat"
    rs = _rows()
    assert len(rs) == 1
    assert rs[0]["source"] == "fast_chat"


async def test_cot_route_llm_ok_logs(tmp_db):
    """A successful LLM call must produce one cot/attempt=1/ok row."""
    raw = (
        "<thinking>It's a weather query.</thinking>"
        '{"route":"tool","tool_name":"weather","tool_args":{"city":"Istanbul"},'
        '"risk_level":"safe","confidence":0.92,"reasoning":"weather lookup"}'
    )

    async def fake_stream(*_a, **_k):
        async def gen():
            yield raw
        return await _drain(gen())

    # Patch _stream_and_collect to bypass streaming and return the raw blob.
    with patch.object(intent, "_stream_and_collect", AsyncMock(return_value=raw)):
        plan, err = await intent.cot_route(
            "weather in Istanbul",
            tool_schemas=[{"name": "weather", "description": "weather"}],
        )

    assert plan is not None
    assert plan["tool_name"] == "weather"
    rs = _rows()
    assert len(rs) == 1
    assert (rs[0]["source"], rs[0]["attempt"], rs[0]["outcome"]) == ("cot", 1, "ok")
    assert rs[0]["confidence"] == 0.92


async def test_cot_route_low_confidence_logs(tmp_db):
    raw = (
        "<thinking>unsure.</thinking>"
        '{"route":"tool","tool_name":"shell","tool_args":{},'
        '"risk_level":"safe","confidence":0.2,"reasoning":"unsure"}'
    )
    with patch.object(intent, "_stream_and_collect", AsyncMock(return_value=raw)):
        plan, err = await intent.cot_route(
            "do something vague",
            tool_schemas=[{"name": "shell", "description": "shell"}],
        )
    assert plan is None  # filtered out by confidence threshold
    rs = _rows()
    assert len(rs) == 1
    assert (rs[0]["source"], rs[0]["outcome"]) == ("cot", "low_confidence")
    assert rs[0]["confidence"] == 0.2


async def test_cot_route_refusal_logs(tmp_db):
    raw = "I cannot help with that request."
    with patch.object(intent, "_stream_and_collect", AsyncMock(return_value=raw)):
        plan, err = await intent.cot_route(
            "something refused",
            tool_schemas=[{"name": "shell", "description": "shell"}],
        )
    assert plan is None
    rs = _rows()
    assert len(rs) == 1
    assert (rs[0]["source"], rs[0]["outcome"]) == ("cot", "refusal")


# tiny helper for the unused fake_stream above
async def _drain(agen):
    out = ""
    async for chunk in agen:
        out += chunk
    return out
