"""Tests for Brain._dedup_history — context-window loop breaker (#184)."""
from __future__ import annotations

import pytest

from bantz.core.brain import Brain


def _asst(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _sys(text: str) -> dict:
    return {"role": "system", "content": text}


# ── 1. No dedup below threshold ───────────────────────────────────────────────

def test_no_dedup_below_threshold():
    """2 identical assistant messages → history unchanged."""
    history = [_user("hi"), _asst("Drink water"), _user("ok"), _asst("Drink water")]
    result = Brain._dedup_history(history)
    assert result == history


# ── 2. Dedup at threshold ─────────────────────────────────────────────────────

def test_dedup_at_threshold():
    """3 identical 'Drink water' messages → collapsed to 1 + anti-loop warning."""
    history = [
        _user("hi"), _asst("Drink water"),
        _user("and"), _asst("Drink water"),
        _user("so"), _asst("Drink water"),
    ]
    result = Brain._dedup_history(history)

    assistant_texts = [m["content"] for m in result if m["role"] == "assistant"]
    assert assistant_texts.count("Drink water") == 1


# ── 3. Anti-loop warning injected ────────────────────────────────────────────

def test_anti_loop_warning_injected():
    """After dedup triggers, a system warning is appended."""
    history = [_asst("Drink water")] * 3
    result = Brain._dedup_history(history)

    system_msgs = [m["content"] for m in result if m["role"] == "system"]
    assert any("repeating" in s for s in system_msgs)
    assert any("NEW" in s for s in system_msgs)


# ── 4. User messages never deduplicated ──────────────────────────────────────

def test_dedup_preserves_user_messages():
    """Identical user messages are never removed."""
    history = [
        _user("Drink water"), _user("Drink water"), _user("Drink water"),
        _asst("Sure, I will."),
    ]
    result = Brain._dedup_history(history)
    user_msgs = [m for m in result if m["role"] == "user"]
    assert len(user_msgs) == 3


# ── 5. Unique assistant messages preserved ───────────────────────────────────

def test_dedup_preserves_unique_messages():
    """Different assistant messages are not removed even if one set repeats."""
    history = [
        _asst("Hello"), _asst("Hello"), _asst("Hello"),
        _asst("Goodbye"),
    ]
    result = Brain._dedup_history(history)
    texts = [m["content"] for m in result if m["role"] == "assistant"]
    assert "Goodbye" in texts


# ── 6. Multiple repeated topics collapsed ────────────────────────────────────

def test_multiple_repeated_topics():
    """Two different repeated messages → both collapsed."""
    history = (
        [_asst("Drink water")] * 3
        + [_asst("Exercise daily")] * 3
    )
    result = Brain._dedup_history(history)
    assistant_texts = [m["content"] for m in result if m["role"] == "assistant"]
    assert assistant_texts.count("Drink water") == 1
    assert assistant_texts.count("Exercise daily") == 1


# ── 7. Case-insensitive dedup ─────────────────────────────────────────────────

def test_case_insensitive_dedup():
    """'Drink Water' and 'drink water' are treated as the same message."""
    history = [
        _asst("Drink Water"),
        _asst("drink water"),
        _asst("DRINK WATER"),
    ]
    result = Brain._dedup_history(history)
    assistant_texts = [m["content"].lower().strip() for m in result if m["role"] == "assistant"]
    assert assistant_texts.count("drink water") == 1


# ── 8. Whitespace normalised ──────────────────────────────────────────────────

def test_whitespace_normalised():
    """'  Drink water  ' and 'Drink water' are treated as the same message."""
    history = [
        _asst("  Drink water  "),
        _asst("Drink water"),
        _asst("Drink water"),
    ]
    result = Brain._dedup_history(history)
    assistant_texts = [m["content"] for m in result if m["role"] == "assistant"]
    # Only one copy should survive
    normalised = [t.strip().lower() for t in assistant_texts]
    assert normalised.count("drink water") == 1


# ── 9. Empty history → empty list ────────────────────────────────────────────

def test_empty_history():
    assert Brain._dedup_history([]) == []


# ── 10. No repeated messages → identical list returned ───────────────────────

def test_no_repetition_returns_unchanged():
    history = [_user("hello"), _asst("Hi there"), _user("bye"), _asst("Farewell")]
    assert Brain._dedup_history(history) == history
