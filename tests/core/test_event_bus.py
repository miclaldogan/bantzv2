"""Tests for ``bantz.core.event_bus`` — Thread-safe async EventBus (#220).

Covers:
  1. Event dataclass (immutability, auto-timestamp)
  2. Subscribe / unsubscribe / clear
  3. Async emit + dispatch
  4. emit_threadsafe from background threads
  5. Wildcard subscribers
  6. Subscriber error isolation
  7. Lifecycle: bind_loop / shutdown / reset
  8. Introspection helpers
"""
from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from bantz.core.event_bus import Event, EventBus


# ── Helpers ───────────────────────────────────────────────────────────

async def _make_bus() -> EventBus:
    """Create a fresh EventBus bound to the current running loop."""
    b = EventBus()
    b.bind_loop(asyncio.get_running_loop())
    return b


@pytest.fixture()
async def eb():
    """Fresh EventBus bound to the test's event loop — auto-shutdown."""
    b = await _make_bus()
    yield b
    await b.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Event dataclass
# ═══════════════════════════════════════════════════════════════════════════

class TestEvent:
    def test_name_and_data(self):
        e = Event(name="TEST", data={"k": 1})
        assert e.name == "TEST"
        assert e.data == {"k": 1}

    def test_auto_timestamp(self):
        before = time.monotonic()
        e = Event(name="T")
        after = time.monotonic()
        assert before <= e.ts <= after

    def test_frozen(self):
        e = Event(name="T")
        with pytest.raises(AttributeError):
            e.name = "CHANGED"  # type: ignore[misc]

    def test_default_data_is_empty_dict(self):
        e = Event(name="T")
        assert e.data == {}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Subscribe / Unsubscribe
# ═══════════════════════════════════════════════════════════════════════════

class TestSubscription:
    def test_on_registers_callback(self):
        b = EventBus()
        cb = MagicMock()
        b.on("X", cb)
        assert b.subscriber_count("X") == 1

    def test_on_returns_callback_for_decorator_use(self):
        b = EventBus()
        cb = MagicMock()
        assert b.on("X", cb) is cb

    def test_off_removes_callback(self):
        b = EventBus()
        cb = MagicMock()
        b.on("X", cb)
        b.off("X", cb)
        assert b.subscriber_count("X") == 0

    def test_off_noop_for_unknown(self):
        """off() must not raise for unregistered callbacks."""
        b = EventBus()
        b.off("NEVER", MagicMock())  # no error

    def test_clear_specific(self):
        b = EventBus()
        b.on("A", MagicMock())
        b.on("B", MagicMock())
        b.clear("A")
        assert b.subscriber_count("A") == 0
        assert b.subscriber_count("B") == 1

    def test_clear_all(self):
        b = EventBus()
        b.on("A", MagicMock())
        b.on("B", MagicMock())
        b.clear()
        assert b.subscriber_count() == 0

    def test_multiple_subscribers_same_event(self):
        b = EventBus()
        b.on("X", MagicMock())
        b.on("X", MagicMock())
        assert b.subscriber_count("X") == 2


# ═══════════════════════════════════════════════════════════════════════════
# 3. Async emit + dispatch
# ═══════════════════════════════════════════════════════════════════════════

class TestAsyncEmit:
    async def test_async_subscriber_receives_event(self, eb: EventBus):
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        eb.on("PING", handler)
        await eb.emit("PING", value=42)
        # Give dispatcher a tick to drain the queue
        await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0].name == "PING"
        assert received[0].data == {"value": 42}

    async def test_sync_subscriber_receives_event(self, eb: EventBus):
        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        eb.on("PONG", handler)
        await eb.emit("PONG", v=1)
        await asyncio.sleep(0.05)
        assert len(received) == 1

    async def test_multiple_events_ordered(self, eb: EventBus):
        names: list[str] = []

        async def handler(event: Event) -> None:
            names.append(event.name)

        eb.on("A", handler)
        eb.on("B", handler)
        await eb.emit("A")
        await eb.emit("B")
        await eb.emit("A")
        await asyncio.sleep(0.05)
        assert names == ["A", "B", "A"]

    async def test_no_subscribers_no_crash(self, eb: EventBus):
        """Emit with zero subscribers must not raise."""
        await eb.emit("NOBODY_LISTENS")
        await asyncio.sleep(0.02)


# ═══════════════════════════════════════════════════════════════════════════
# 4. emit_threadsafe — cross-thread publishing
# ═══════════════════════════════════════════════════════════════════════════

class TestEmitThreadsafe:
    async def test_event_arrives_from_background_thread(self, eb: EventBus):
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        eb.on("THREAD_EVENT", handler)

        # Fire from a background thread (simulates sensor)
        def bg():
            eb.emit_threadsafe("THREAD_EVENT", source="bg_thread")

        t = threading.Thread(target=bg)
        t.start()
        t.join(timeout=2)
        await asyncio.sleep(0.1)
        assert len(received) == 1
        assert received[0].data["source"] == "bg_thread"

    async def test_multiple_threads_concurrent(self, eb: EventBus):
        counter: list[int] = []

        async def handler(event: Event) -> None:
            counter.append(1)

        eb.on("MT", handler)

        threads = []
        for i in range(10):
            t = threading.Thread(target=lambda: eb.emit_threadsafe("MT", i=i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=2)

        await asyncio.sleep(0.2)
        assert len(counter) == 10

    def test_emit_threadsafe_without_loop_does_not_crash(self):
        """Before bind_loop(), emit_threadsafe logs a warning and drops."""
        b = EventBus()  # no bind_loop
        b.emit_threadsafe("DROPPED")  # must not raise


# ═══════════════════════════════════════════════════════════════════════════
# 5. Wildcard subscribers
# ═══════════════════════════════════════════════════════════════════════════

class TestWildcard:
    async def test_wildcard_catches_all(self, eb: EventBus):
        log: list[str] = []

        async def catch_all(event: Event) -> None:
            log.append(event.name)

        eb.on("*", catch_all)
        await eb.emit("ALPHA")
        await eb.emit("BETA")
        await asyncio.sleep(0.05)
        assert log == ["ALPHA", "BETA"]

    async def test_wildcard_plus_specific(self, eb: EventBus):
        specific_log: list[str] = []
        wild_log: list[str] = []

        async def specific(e: Event) -> None:
            specific_log.append(e.name)

        async def wild(e: Event) -> None:
            wild_log.append(e.name)

        eb.on("ONLY_THIS", specific)
        eb.on("*", wild)

        await eb.emit("ONLY_THIS")
        await eb.emit("OTHER")
        await asyncio.sleep(0.05)
        assert specific_log == ["ONLY_THIS"]
        assert wild_log == ["ONLY_THIS", "OTHER"]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Subscriber error isolation
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorIsolation:
    async def test_failing_sub_does_not_block_others(self, eb: EventBus):
        results: list[str] = []

        async def boom(event: Event) -> None:
            raise RuntimeError("subscriber crash")

        async def safe(event: Event) -> None:
            results.append("ok")

        eb.on("ERR", boom)
        eb.on("ERR", safe)
        await eb.emit("ERR")
        await asyncio.sleep(0.05)
        assert results == ["ok"]

    async def test_sync_subscriber_exception_isolated(self, eb: EventBus):
        results: list[str] = []

        def boom(event: Event) -> None:
            raise ValueError("sync crash")

        def safe(event: Event) -> None:
            results.append("ok")

        eb.on("SE", boom)
        eb.on("SE", safe)
        await eb.emit("SE")
        await asyncio.sleep(0.05)
        assert results == ["ok"]


# ═══════════════════════════════════════════════════════════════════════════
# 7. Lifecycle: bind_loop / shutdown / reset
# ═══════════════════════════════════════════════════════════════════════════

class TestLifecycle:
    async def test_bind_loop_is_idempotent(self):
        b = EventBus()
        loop = asyncio.get_running_loop()
        b.bind_loop(loop)
        task1 = b._dispatcher_task
        b.bind_loop(loop)  # second call
        assert b._dispatcher_task is task1  # same task, not duplicated
        await b.shutdown()

    async def test_shutdown_drains_pending(self):
        b = EventBus()
        b.bind_loop(asyncio.get_running_loop())

        received: list[Event] = []

        async def handler(e: Event) -> None:
            received.append(e)

        b.on("DRAIN", handler)
        # Put event on queue without giving dispatcher time
        await b._queue.put(Event(name="DRAIN"))
        await b.shutdown()  # should drain
        assert any(e.name == "DRAIN" for e in received)

    def test_reset_clears_everything(self):
        b = EventBus()
        b.on("X", MagicMock())
        b.reset()
        assert b.subscriber_count() == 0
        assert not b._started

    async def test_is_running(self):
        b = EventBus()
        assert not b.is_running
        b.bind_loop(asyncio.get_running_loop())
        assert b.is_running
        await b.shutdown()
        assert not b.is_running


# ═══════════════════════════════════════════════════════════════════════════
# 8. Introspection
# ═══════════════════════════════════════════════════════════════════════════

class TestIntrospection:
    def test_subscriber_count_specific(self):
        b = EventBus()
        b.on("A", MagicMock())
        b.on("A", MagicMock())
        b.on("B", MagicMock())
        assert b.subscriber_count("A") == 2
        assert b.subscriber_count("B") == 1

    def test_subscriber_count_total(self):
        b = EventBus()
        b.on("A", MagicMock())
        b.on("B", MagicMock())
        b.on("*", MagicMock())
        assert b.subscriber_count() == 3

    def test_subscriber_count_empty(self):
        b = EventBus()
        assert b.subscriber_count("NOPE") == 0
        assert b.subscriber_count() == 0
