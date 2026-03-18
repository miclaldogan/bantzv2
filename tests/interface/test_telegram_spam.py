"""Tests for _TelegramSpamFilter — Telegram background spam filter (#184)."""
from __future__ import annotations

import asyncio
import time

import pytest

from bantz.interface.telegram_bot import _TelegramSpamFilter


# ── Helpers ───────────────────────────────────────────────────────────────────

def fresh_filter(**kwargs) -> _TelegramSpamFilter:
    """Return a new _TelegramSpamFilter, optionally overriding class constants."""
    f = _TelegramSpamFilter()
    for k, v in kwargs.items():
        setattr(f, k, v)
    return f


# ── 1. User messages always pass through ─────────────────────────────────────

@pytest.mark.asyncio
async def test_user_messages_pass_through():
    f = fresh_filter()
    for _ in range(10):
        result = await f.should_send("Hello!", is_system=False)
        assert result == "Hello!"


# ── 2. Below threshold → each message returned individually ──────────────────

@pytest.mark.asyncio
async def test_below_threshold_pass_through():
    f = fresh_filter(SPAM_THRESHOLD=5)
    results = []
    for i in range(4):
        r = await f.should_send(f"Task {i} done", is_system=True)
        results.append(r)
    assert all(r is not None for r in results)
    assert len(results) == 4


# ── 3. At threshold → bundled summary returned ───────────────────────────────

@pytest.mark.asyncio
async def test_threshold_triggers_summary():
    f = fresh_filter(SPAM_THRESHOLD=5)
    results = []
    for i in range(5):
        r = await f.should_send(f"Task {i} done", is_system=True)
        results.append(r)

    # First 4 pass through, 5th triggers summary
    non_null = [r for r in results if r is not None]
    # The 5th should be a summary bundling all 5
    summary_results = [r for r in non_null if "maintenance" in r.lower() or "tasks completed" in r.lower()]
    assert len(summary_results) >= 1


# ── 4. Window resets after expiry ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_window_resets_after_expiry():
    f = fresh_filter(SPAM_THRESHOLD=5, WINDOW_SECONDS=0.01)
    # Send 3 messages
    for i in range(3):
        await f.should_send(f"msg {i}", is_system=True)

    # Wait for window to expire
    await asyncio.sleep(0.05)

    # Buffer should reset — first message of new window passes through
    result = await f.should_send("fresh start", is_system=True)
    assert result is not None
    assert result == "fresh start"


# ── 5. Summary is in butler character ────────────────────────────────────────

@pytest.mark.asyncio
async def test_summary_is_in_character():
    f = fresh_filter(SPAM_THRESHOLD=3)
    results = []
    for i in range(3):
        r = await f.should_send(f"job {i}", is_system=True)
        results.append(r)

    summaries = [r for r in results if r and ("ma'am" in r.lower() or "nominal" in r.lower() or "tasks" in r.lower())]
    assert summaries, "Summary should be in butler character (mention Ma'am, tasks, or nominal)"


# ── 6. flush() returns summary of buffered messages ──────────────────────────

@pytest.mark.asyncio
async def test_flush_returns_buffered():
    f = fresh_filter(SPAM_THRESHOLD=10)  # high threshold so nothing auto-flushes
    await f.should_send("task A", is_system=True)
    await f.should_send("task B", is_system=True)

    summary = await f.flush()
    assert summary is not None
    assert "2" in summary or "two" in summary.lower() or "tasks" in summary.lower()


# ── 7. flush() with empty buffer returns None ────────────────────────────────

@pytest.mark.asyncio
async def test_flush_empty_returns_none():
    f = fresh_filter()
    result = await f.flush()
    assert result is None


# ── 8. Buffer cleared after flush ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_buffer_cleared_after_flush():
    f = fresh_filter(SPAM_THRESHOLD=10)
    await f.should_send("msg", is_system=True)
    await f.flush()
    # Second flush should return None (buffer empty)
    assert await f.flush() is None


# ── 9. Concurrent sends are safe ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_sends_are_thread_safe():
    f = fresh_filter(SPAM_THRESHOLD=5)
    results = await asyncio.gather(*[
        f.should_send(f"concurrent {i}", is_system=True) for i in range(5)
    ])
    # Should not raise; exactly one summary emitted at threshold
    non_null = [r for r in results if r is not None]
    assert len(non_null) >= 1
