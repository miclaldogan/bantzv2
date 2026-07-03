"""Terse follow-ups keep prior-tool context (issue #495).

A short, context-dependent reply right after a Gmail turn — e.g.
"u sure? from medium daily?" — carries no email keyword, so the old
``_build_tool_context`` (gated purely on ``_EMAIL_HINTS``) dropped the email
context and the router fell back to chat. These tests pin the fix: a terse
follow-up to the last tool re-surfaces that tool's context, while a genuine
topic switch still does not.
"""
from __future__ import annotations

import pytest

import bantz.core.brain as brain_mod
from bantz.core.brain import Brain


@pytest.fixture(autouse=True)
def _no_awareness(monkeypatch):
    # The tail of _build_tool_context consults config.awareness_enabled; keep
    # it off so the helper stays hermetic.
    monkeypatch.setattr(brain_mod.config, "awareness_enabled", False, raising=False)


def _brain(*, last_tool="gmail", messages=None, events=None, tool_output=""):
    b = Brain.__new__(Brain)
    b._turn_counter = 5
    b._context_turn = 5            # fresh — within TTL
    b._CONTEXT_TTL = 3
    b._last_messages = messages or []
    b._last_events = events or []
    b._last_tool_output = tool_output
    b._last_tool_name = last_tool
    b._last_screen_description = ""
    b._screen_description_turn = 0
    return b


_MSGS = [
    {"id": "m1", "from": "Medium Daily <noreply@medium.com>", "subject": "Today's picks"},
    {"id": "m2", "from": "Defne <defne@gmail.com>", "subject": "dinner?"},
]


# ── _is_tool_followup unit ───────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "u sure? from medium daily?",
    "u sure?",
    "from medium daily?",
    "which one?",
    "reply to that one",          # anaphora "that"
    "just the unread ones",
])
def test_is_followup_true(text):
    assert Brain._is_tool_followup(_brain(), text.lower()) is True


@pytest.mark.parametrize("text", [
    "what's the weather in elazig?",
    "play some music",
    "run maintenance",
])
def test_is_followup_false(text):
    assert Brain._is_tool_followup(_brain(), text.lower()) is False


# ── context injection for terse Gmail follow-ups ─────────────────────────────

def test_terse_followup_after_gmail_injects_email_context():
    b = _brain(last_tool="gmail", messages=_MSGS)
    ctx = b._build_tool_context("u sure? from medium daily?")
    assert "RECENT EMAIL RESULTS" in ctx
    assert "m1" in ctx and "Medium Daily" in ctx


def test_topic_switch_after_gmail_does_not_inject():
    # A genuine new topic (no anaphora / follow-up marker) must NOT drag the
    # stale email context along — the pre-#495 discipline is preserved.
    b = _brain(last_tool="gmail", messages=_MSGS)
    ctx = b._build_tool_context("what's the weather in elazig?")
    assert "RECENT EMAIL RESULTS" not in ctx


def test_email_keyword_still_injects():
    # Regression: the original keyword path is unchanged.
    b = _brain(last_tool="gmail", messages=_MSGS)
    ctx = b._build_tool_context("any new emails?")
    assert "RECENT EMAIL RESULTS" in ctx


def test_terse_followup_gated_to_matching_tool():
    # Terse follow-up but the last tool was NOT gmail → don't inject email
    # context off a stale message list.
    b = _brain(last_tool="weather", messages=_MSGS)
    ctx = b._build_tool_context("u sure?")
    assert "RECENT EMAIL RESULTS" not in ctx


def test_terse_followup_after_calendar_injects_events():
    events = [{"id": "e1", "summary": "Standup", "start": "09:00"}]
    b = _brain(last_tool="calendar", events=events)
    ctx = b._build_tool_context("which one?")
    assert "RECENT CALENDAR EVENTS" in ctx
    assert "Standup" in ctx
