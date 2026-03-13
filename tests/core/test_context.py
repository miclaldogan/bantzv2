"""Tests for bantz.core.context — BantzContext dataclass (#224)."""
from __future__ import annotations

import time

import pytest

from bantz.core.context import BantzContext


# ── construction & defaults ───────────────────────────────────────────


class TestDefaults:
    def test_default_session_id_is_12_hex(self):
        ctx = BantzContext()
        assert len(ctx.session_id) == 12
        assert ctx.session_id.isalnum()

    def test_two_contexts_have_unique_ids(self):
        a = BantzContext()
        b = BantzContext()
        assert a.session_id != b.session_id

    def test_default_field_values(self):
        ctx = BantzContext()
        assert ctx.user_input == ""
        assert ctx.en_input == ""
        assert ctx.source_lang == "en"
        assert ctx.is_remote is False
        assert ctx.confirmed is False
        assert ctx.feedback is None
        assert ctx.route == "chat"
        assert ctx.tool_name is None
        assert ctx.tool_args == {}
        assert ctx.risk_level == "safe"
        assert ctx.tool_success is None
        assert ctx.response == ""
        assert ctx.is_streaming is False
        assert ctx.needs_confirm is False
        assert ctx.completed_at is None

    def test_mutable_defaults_are_independent(self):
        """Each instance must get its own dict/list — no shared mutable default."""
        a = BantzContext()
        b = BantzContext()
        a.tool_args["foo"] = 1
        assert "foo" not in b.tool_args

    def test_created_at_is_recent(self):
        before = time.time()
        ctx = BantzContext()
        after = time.time()
        assert before <= ctx.created_at <= after


# ── explicit construction ─────────────────────────────────────────────


class TestExplicit:
    def test_set_fields_at_construction(self):
        ctx = BantzContext(
            user_input="merhaba",
            en_input="hello",
            source_lang="tr",
            is_remote=True,
            route="tool",
            tool_name="weather",
            tool_args={"city": "Istanbul"},
            risk_level="safe",
        )
        assert ctx.user_input == "merhaba"
        assert ctx.en_input == "hello"
        assert ctx.source_lang == "tr"
        assert ctx.is_remote is True
        assert ctx.is_tool_call is True
        assert ctx.tool_args == {"city": "Istanbul"}

    def test_feedback_positive(self):
        ctx = BantzContext(feedback="positive", feedback_hint="praise")
        assert ctx.feedback == "positive"
        assert ctx.feedback_hint == "praise"

    def test_feedback_negative(self):
        ctx = BantzContext(feedback="negative")
        assert ctx.feedback == "negative"


# ── helper properties ─────────────────────────────────────────────────


class TestHelpers:
    def test_is_tool_call_true(self):
        ctx = BantzContext(route="tool", tool_name="shell")
        assert ctx.is_tool_call is True

    def test_is_tool_call_false_when_chat(self):
        ctx = BantzContext(route="chat", tool_name="shell")
        assert ctx.is_tool_call is False

    def test_is_tool_call_false_when_no_tool(self):
        ctx = BantzContext(route="tool", tool_name=None)
        assert ctx.is_tool_call is False

    def test_has_memory_false_by_default(self):
        ctx = BantzContext()
        assert ctx.has_memory is False

    def test_has_memory_with_graph(self):
        ctx = BantzContext(graph_context="entities…")
        assert ctx.has_memory is True

    def test_has_memory_with_vector(self):
        ctx = BantzContext(vector_context="past msgs…")
        assert ctx.has_memory is True

    def test_has_memory_with_deep(self):
        ctx = BantzContext(deep_memory="deep recall…")
        assert ctx.has_memory is True

    def test_elapsed_ms_none_before_complete(self):
        ctx = BantzContext()
        assert ctx.elapsed_ms is None

    def test_elapsed_ms_after_mark_complete(self):
        ctx = BantzContext()
        # Simulate some "work"
        ctx.mark_complete()
        ms = ctx.elapsed_ms
        assert ms is not None
        assert ms >= 0


# ── mark_complete & as_log_dict ───────────────────────────────────────


class TestLogDict:
    def test_as_log_dict_keys(self):
        ctx = BantzContext(
            user_input="hello world",
            route="tool",
            tool_name="weather",
            risk_level="safe",
        )
        ctx.mark_complete()
        d = ctx.as_log_dict()
        assert set(d.keys()) == {
            "sid", "input", "lang", "route",
            "tool", "risk", "ok", "stream", "ms",
        }
        assert d["input"] == "hello world"
        assert d["route"] == "tool"
        assert d["tool"] == "weather"
        assert d["ms"] is not None

    def test_log_dict_truncates_long_input(self):
        ctx = BantzContext(user_input="x" * 200)
        d = ctx.as_log_dict()
        assert len(d["input"]) == 80
