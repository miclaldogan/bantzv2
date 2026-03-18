"""
Bantz — Redis session store + task queue + pub/sub + rate limiter (#294)

Provides four classes backed by Redis async:
    SessionStore   — HSET/HGETALL with TTL for conversation session data
    TaskQueue      — LPUSH/BRPOP list-based task queue
    PubSub         — publish / subscribe on bantz:events channel
    RateLimiter    — sliding-window counter (INCR + EXPIRE)

All classes degrade gracefully to in-memory fallbacks when Redis is
unavailable, emitting a warning on first failure.

Usage:
    from bantz.memory.session_store import session_store, task_queue, pubsub, rate_limiter

    await session_store.set("sess:123", {"user": "alice"}, ttl=3600)
    data = await session_store.get("sess:123")
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from typing import Any

log = logging.getLogger("bantz.session_store")

# Default Redis URL — overridden by config.yaml / env vars
_DEFAULT_URL = "redis://localhost:6379/0"

# Pub/sub channel for TUI ↔ Telegram inter-process events
EVENTS_CHANNEL = "bantz:events"


def _get_redis_url() -> str:
    try:
        from bantz.config import config as _cfg
        host = getattr(_cfg, "redis_host", "localhost")
        port = getattr(_cfg, "redis_port", 6379)
        db = getattr(_cfg, "redis_db", 0)
        return f"redis://{host}:{port}/{db}"
    except Exception:
        return _DEFAULT_URL


# ── Fallback in-memory implementations ───────────────────────────────────────

class _MemorySessionStore:
    """Thread-unsafe in-memory fallback for SessionStore."""

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}
        self._expiry: dict[str, float] = {}

    async def set(self, session_id: str, data: dict, ttl: int = 3600) -> None:
        self._data[session_id] = dict(data)
        self._expiry[session_id] = time.time() + ttl

    async def get(self, session_id: str) -> dict | None:
        if session_id not in self._data:
            return None
        if time.time() > self._expiry.get(session_id, float("inf")):
            del self._data[session_id]
            return None
        return dict(self._data[session_id])

    async def delete(self, session_id: str) -> bool:
        existed = session_id in self._data
        self._data.pop(session_id, None)
        self._expiry.pop(session_id, None)
        return existed


class _MemoryTaskQueue:
    """In-memory fallback for TaskQueue."""

    def __init__(self) -> None:
        self._queues: dict[str, deque] = defaultdict(deque)

    async def push(self, task: dict, queue: str = "bantz:tasks") -> None:
        self._queues[queue].appendleft(json.dumps(task))

    async def pop(self, queue: str = "bantz:tasks", timeout: int = 5) -> dict | None:
        q = self._queues[queue]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if q:
                return json.loads(q.pop())
            await asyncio.sleep(0.05)
        return None

    async def length(self, queue: str = "bantz:tasks") -> int:
        return len(self._queues[queue])


class _MemoryRateLimiter:
    """In-memory sliding-window rate limiter fallback."""

    def __init__(self) -> None:
        self._counts: dict[str, list[float]] = defaultdict(list)

    async def check(self, key: str, limit: int, window_sec: int) -> bool:
        now = time.time()
        timestamps = self._counts[key]
        cutoff = now - window_sec
        # Remove expired entries
        self._counts[key] = [t for t in timestamps if t > cutoff]
        if len(self._counts[key]) >= limit:
            return False
        self._counts[key].append(now)
        return True


# ── Redis-backed implementations ──────────────────────────────────────────────

class SessionStore:
    """Redis HSET-based session store with TTL.

    Falls back to in-memory dict with a warning if Redis is unavailable.
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url or _get_redis_url()
        self._redis: Any = None
        self._fallback = _MemorySessionStore()
        self._warned = False

    async def _r(self) -> Any:
        """Lazy connect; return Redis client or None on failure."""
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(self._url, decode_responses=True)
            await client.ping()
            self._redis = client
            return self._redis
        except Exception as exc:
            if not self._warned:
                log.warning("SessionStore: Redis unavailable (%s) — using memory fallback", exc)
                self._warned = True
            return None

    async def set(self, session_id: str, data: dict, ttl: int = 3600) -> None:
        """Store session data under session_id, expiring after ttl seconds.

        Args:
            session_id: Redis key (e.g. "sess:abc123").
            data:       Flat dict of string values.
            ttl:        Time-to-live in seconds (default 3600).
        """
        r = await self._r()
        if r is None:
            return await self._fallback.set(session_id, data, ttl)
        try:
            # Convert all values to strings for HSET compatibility
            str_data = {k: str(v) for k, v in data.items()}
            await r.hset(session_id, mapping=str_data)
            await r.expire(session_id, ttl)
        except Exception as exc:
            log.debug("SessionStore.set error: %s", exc)
            await self._fallback.set(session_id, data, ttl)

    async def get(self, session_id: str) -> dict | None:
        """Retrieve session data. Returns None if the key does not exist."""
        r = await self._r()
        if r is None:
            return await self._fallback.get(session_id)
        try:
            result = await r.hgetall(session_id)
            return result if result else None
        except Exception as exc:
            log.debug("SessionStore.get error: %s", exc)
            return await self._fallback.get(session_id)

    async def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if the key existed."""
        r = await self._r()
        if r is None:
            return await self._fallback.delete(session_id)
        try:
            deleted = await r.delete(session_id)
            return bool(deleted)
        except Exception as exc:
            log.debug("SessionStore.delete error: %s", exc)
            return await self._fallback.delete(session_id)


class TaskQueue:
    """Redis list-based task queue (LPUSH / BRPOP).

    Falls back to in-memory deque with a warning if Redis is unavailable.
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url or _get_redis_url()
        self._redis: Any = None
        self._fallback = _MemoryTaskQueue()
        self._warned = False

    async def _r(self) -> Any:
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(self._url, decode_responses=True)
            await client.ping()
            self._redis = client
            return self._redis
        except Exception as exc:
            if not self._warned:
                log.warning("TaskQueue: Redis unavailable (%s) — using memory fallback", exc)
                self._warned = True
            return None

    async def push(self, task: dict, queue: str = "bantz:tasks") -> None:
        """Push a task dict to the left of the queue list."""
        r = await self._r()
        if r is None:
            return await self._fallback.push(task, queue)
        try:
            await r.lpush(queue, json.dumps(task))
        except Exception as exc:
            log.debug("TaskQueue.push error: %s", exc)
            await self._fallback.push(task, queue)

    async def pop(self, queue: str = "bantz:tasks", timeout: int = 5) -> dict | None:
        """Blocking pop from the right of the queue. Returns None on timeout."""
        r = await self._r()
        if r is None:
            return await self._fallback.pop(queue, timeout)
        try:
            result = await r.brpop(queue, timeout=timeout)
            if result is None:
                return None
            _, value = result
            return json.loads(value)
        except Exception as exc:
            log.debug("TaskQueue.pop error: %s", exc)
            return await self._fallback.pop(queue, timeout)

    async def length(self, queue: str = "bantz:tasks") -> int:
        """Return the current queue depth."""
        r = await self._r()
        if r is None:
            return await self._fallback.length(queue)
        try:
            return await r.llen(queue)
        except Exception as exc:
            log.debug("TaskQueue.length error: %s", exc)
            return await self._fallback.length(queue)


class PubSub:
    """Redis pub/sub on the bantz:events channel.

    Falls back to a no-op with warning if Redis is unavailable.
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url or _get_redis_url()
        self._redis: Any = None
        self._warned = False

    async def _r(self) -> Any:
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(self._url, decode_responses=True)
            await client.ping()
            self._redis = client
            return self._redis
        except Exception as exc:
            if not self._warned:
                log.warning("PubSub: Redis unavailable (%s) — events will be dropped", exc)
                self._warned = True
            return None

    async def publish(self, event: dict, channel: str = EVENTS_CHANNEL) -> int:
        """Publish an event dict. Returns number of subscribers that received it."""
        r = await self._r()
        if r is None:
            return 0
        try:
            return await r.publish(channel, json.dumps(event))
        except Exception as exc:
            log.debug("PubSub.publish error: %s", exc)
            return 0

    async def subscribe(self, channel: str = EVENTS_CHANNEL):
        """Return a Redis pubsub subscription context, or None if unavailable."""
        r = await self._r()
        if r is None:
            return None
        try:
            ps = r.pubsub()
            await ps.subscribe(channel)
            return ps
        except Exception as exc:
            log.debug("PubSub.subscribe error: %s", exc)
            return None


class RateLimiter:
    """Sliding-window rate limiter using Redis INCR + EXPIRE.

    Falls back to in-memory counter if Redis is unavailable.
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url or _get_redis_url()
        self._redis: Any = None
        self._fallback = _MemoryRateLimiter()
        self._warned = False

    async def _r(self) -> Any:
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(self._url, decode_responses=True)
            await client.ping()
            self._redis = client
            return self._redis
        except Exception as exc:
            if not self._warned:
                log.warning("RateLimiter: Redis unavailable (%s) — using memory fallback", exc)
                self._warned = True
            return None

    async def check(self, key: str, limit: int, window_sec: int) -> bool:
        """Increment counter for key; return True if under limit, False if over.

        Uses a fixed window keyed by floor(now / window_sec) so counters
        auto-expire without a background task.

        Args:
            key:        Identifier (e.g. "telegram:user:42").
            limit:      Max allowed calls in window.
            window_sec: Window duration in seconds.

        Returns:
            True  — request is allowed (count now ≤ limit)
            False — rate limit exceeded
        """
        r = await self._r()
        if r is None:
            return await self._fallback.check(key, limit, window_sec)
        try:
            # Bucket by time window so each window gets its own key
            bucket = int(time.time() // window_sec)
            rkey = f"rl:{key}:{bucket}"
            count = await r.incr(rkey)
            if count == 1:
                await r.expire(rkey, window_sec * 2)
            return count <= limit
        except Exception as exc:
            log.debug("RateLimiter.check error: %s", exc)
            return await self._fallback.check(key, limit, window_sec)


# ── Module singletons ─────────────────────────────────────────────────────────

session_store = SessionStore()
task_queue = TaskQueue()
pubsub = PubSub()
rate_limiter = RateLimiter()
