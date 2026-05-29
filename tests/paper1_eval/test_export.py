"""Tests for paper1_eval.export_pairs."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from bantz.core import finalizer, route_log
from bantz.core.memory import memory

from paper1_eval import export_pairs, labels


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "bantz.db"
    memory.init(p)
    memory.new_session()
    # Seed one turn end-to-end.
    route_log.log_route(
        en_input="weather in Istanbul", route="tool", tool_name="weather",
        confidence=0.92, reasoning="weather", source="cot", attempt=1,
        outcome="ok",
    )
    finalizer.log_finalizer_call(
        user_input="weather in Istanbul",
        tool_output="22C sunny", response="22C sunny, ma'am.",
        confidence=0.95, tool_used="weather", mode="short",
    )
    from bantz.data.connection_pool import get_pool
    ts = datetime.now().isoformat(timespec="milliseconds")
    with get_pool().connection(write=True) as conn:
        conn.execute(
            "INSERT INTO messages(conversation_id, role, content, tool_used,"
            " created_at) VALUES (?,?,?,?,?)",
            (memory.session_id, "assistant", "22C sunny, ma'am.", "weather", ts),
        )
    yield p
    memory.close()


def test_export_writes_one_row_per_assistant_message(db, tmp_path):
    out = tmp_path / "pairs.jsonl"
    n = export_pairs.export(db, out)
    assert n == 1
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["response_text"] == "22C sunny, ma'am."
    assert payload["finalizer_confidence"] == 0.95
    assert payload["route"] == "tool"
    assert payload["label"] is None  # unlabeled


def test_export_includes_labels_when_present(db, tmp_path):
    # Find the message_id we just inserted.
    from bantz.data.connection_pool import get_pool
    with get_pool().connection() as conn:
        msg_id = conn.execute(
            "SELECT id FROM messages WHERE role='assistant' LIMIT 1"
        ).fetchone()[0]
    labels.upsert_label(db, msg_id, "faithful", labeler="alice")

    out = tmp_path / "pairs.jsonl"
    export_pairs.export(db, out)
    payload = json.loads(out.read_text().splitlines()[0])
    assert payload["label"] == "faithful"
    assert payload["labeler"] == "alice"


def test_export_respects_flagged_only_and_limit(db, tmp_path):
    out = tmp_path / "pairs.jsonl"
    n = export_pairs.export(db, out, flagged_only=True)
    # The seeded row has confidence=0.95 → not flagged → empty file.
    assert n == 0
    assert out.read_text() == ""

    out2 = tmp_path / "pairs2.jsonl"
    n2 = export_pairs.export(db, out2, limit=0)
    assert n2 == 0
