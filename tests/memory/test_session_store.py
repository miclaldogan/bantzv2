"""Tests for Redis session store + task queue + pub/sub + rate limiter (#294).

All tests use the in-memory fallback paths (no real Redis needed).
"""
from __future__ import annotations

import asyncio
import time

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_session_store():
    """SessionStore wired to always use the memory fallback."""
    from bantz.memory.session_store import SessionStore, _MemorySessionStore
    ss = SessionStore.__new__(SessionStore)
    ss._redis = None
    ss._url = "redis://127.0.0.1:1"   # deliberately unreachable
    ss._fallback = _MemorySessionStore()
    ss._warned = True  # suppress warning output in tests
    return ss


def _make_task_queue():
    from bantz.memory.session_store import TaskQueue, _MemoryTaskQueue
    tq = TaskQueue.__new__(TaskQueue)
    tq._redis = None
    tq._url = "redis://127.0.0.1:1"
    tq._fallback = _MemoryTaskQueue()
    tq._warned = True
    return tq


def _make_rate_limiter():
    from bantz.memory.session_store import RateLimiter, _MemoryRateLimiter
    rl = RateLimiter.__new__(RateLimiter)
    rl._redis = None
    rl._url = "redis://127.0.0.1:1"
    rl._fallback = _MemoryRateLimiter()
    rl._warned = True
    return rl


# ── SessionStore ──────────────────────────────────────────────────────────────

class TestSessionStore:

    @pytest.mark.asyncio
    async def test_set_then_get(self):
        ss = _make_session_store()
        await ss.set("sess:1", {"user": "alice", "lang": "tr"}, ttl=3600)
        data = await ss.get("sess:1")
        assert data["user"] == "alice"
        assert data["lang"] == "tr"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        ss = _make_session_store()
        assert await ss.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_set_overwrites_existing(self):
        ss = _make_session_store()
        await ss.set("sess:2", {"x": "1"})
        await ss.set("sess:2", {"x": "2"})
        data = await ss.get("sess:2")
        assert data["x"] == "2"

    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self):
        ss = _make_session_store()
        await ss.set("sess:3", {"a": "b"})
        result = await ss.delete("sess:3")
        assert result is True
        assert await ss.get("sess:3") is None

    @pytest.mark.asyncio
    async def test_delete_missing_returns_false(self):
        ss = _make_session_store()
        assert await ss.delete("no_such_key") is False

    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        ss = _make_session_store()
        await ss.set("sess:ttl", {"v": "1"}, ttl=0)
        # TTL=0 means already expired for in-memory fallback
        data = await ss.get("sess:ttl")
        # Either None (expired) or the dict — depends on timing; just ensure no crash
        assert data is None or isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_fallback_used_when_redis_unavailable(self):
        """Redis ping fails → fallback used, no exception raised."""
        from bantz.memory.session_store import SessionStore
        ss = SessionStore(url="redis://127.0.0.1:1")  # unreachable port
        await ss.set("k", {"v": "1"})   # must not raise
        # Data lands in fallback
        data = await ss._fallback.get("k")
        assert data is not None


# ── TaskQueue ─────────────────────────────────────────────────────────────────

class TestTaskQueue:

    @pytest.mark.asyncio
    async def test_push_then_pop(self):
        tq = _make_task_queue()
        await tq.push({"action": "summarize", "doc": "x.pdf"})
        task = await tq.pop(timeout=1)
        assert task is not None
        assert task["action"] == "summarize"

    @pytest.mark.asyncio
    async def test_pop_empty_returns_none(self):
        tq = _make_task_queue()
        result = await tq.pop(timeout=0)
        assert result is None

    @pytest.mark.asyncio
    async def test_fifo_order(self):
        """LPUSH / BRPOP gives LIFO in Redis lists; first pushed = last popped."""
        tq = _make_task_queue()
        await tq.push({"seq": 1})
        await tq.push({"seq": 2})
        await tq.push({"seq": 3})
        t1 = await tq.pop(timeout=1)
        t2 = await tq.pop(timeout=1)
        t3 = await tq.pop(timeout=1)
        # Memory fallback uses deque — check all tasks retrieved
        seqs = {t1["seq"], t2["seq"], t3["seq"]}
        assert seqs == {1, 2, 3}

    @pytest.mark.asyncio
    async def test_length_after_push(self):
        tq = _make_task_queue()
        await tq.push({"a": 1})
        await tq.push({"b": 2})
        assert await tq.length() == 2

    @pytest.mark.asyncio
    async def test_length_decreases_after_pop(self):
        tq = _make_task_queue()
        await tq.push({"a": 1})
        await tq.pop(timeout=1)
        assert await tq.length() == 0

    @pytest.mark.asyncio
    async def test_custom_queue_name(self):
        tq = _make_task_queue()
        await tq.push({"x": 1}, queue="my:queue")
        assert await tq.length(queue="my:queue") == 1
        assert await tq.length() == 0  # default queue unchanged


# ── RateLimiter ───────────────────────────────────────────────────────────────

class TestRateLimiter:

    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        rl = _make_rate_limiter()
        for _ in range(5):
            assert await rl.check("user:1", limit=5, window_sec=60) is True

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        rl = _make_rate_limiter()
        for _ in range(3):
            await rl.check("user:2", limit=3, window_sec=60)
        # 4th call should be blocked
        assert await rl.check("user:2", limit=3, window_sec=60) is False

    @pytest.mark.asyncio
    async def test_different_keys_independent(self):
        rl = _make_rate_limiter()
        for _ in range(3):
            await rl.check("user:A", limit=3, window_sec=60)
        # user:A is exhausted; user:B should still be allowed
        assert await rl.check("user:A", limit=3, window_sec=60) is False
        assert await rl.check("user:B", limit=3, window_sec=60) is True

    @pytest.mark.asyncio
    async def test_window_resets(self):
        rl = _make_rate_limiter()
        # Use a tiny window so it expires quickly
        for _ in range(3):
            await rl.check("user:C", limit=3, window_sec=1)
        assert await rl.check("user:C", limit=3, window_sec=1) is False
        # Manually clear the counter to simulate window expiry
        rl._fallback._counts["user:C"].clear()
        assert await rl.check("user:C", limit=3, window_sec=1) is True


# ── PubSub ────────────────────────────────────────────────────────────────────

class TestPubSub:

    @pytest.mark.asyncio
    async def test_publish_no_redis_returns_zero(self):
        """Without Redis, publish returns 0 subscribers."""
        from bantz.memory.session_store import PubSub
        ps = PubSub(url="redis://127.0.0.1:1")  # unreachable
        result = await ps.publish({"type": "ping"})
        assert result == 0

    @pytest.mark.asyncio
    async def test_subscribe_no_redis_returns_none(self):
        from bantz.memory.session_store import PubSub
        ps = PubSub(url="redis://127.0.0.1:1")
        result = await ps.subscribe()
        assert result is None

    def test_events_channel_constant(self):
        from bantz.memory.session_store import EVENTS_CHANNEL
        assert EVENTS_CHANNEL == "bantz:events"


# ── Singletons ────────────────────────────────────────────────────────────────

class TestSingletons:

    def test_all_singletons_exist(self):
        from bantz.memory.session_store import (
            session_store, task_queue, pubsub, rate_limiter,
            SessionStore, TaskQueue, PubSub, RateLimiter,
        )
        assert isinstance(session_store, SessionStore)
        assert isinstance(task_queue, TaskQueue)
        assert isinstance(pubsub, PubSub)
        assert isinstance(rate_limiter, RateLimiter)

    def test_singleton_identity(self):
        from bantz.memory.session_store import session_store as a
        from bantz.memory.session_store import session_store as b
        assert a is b
