"""
Bantz — Thread-Safe Async EventBus (#220, Sprint 3 Part 1)

A **singleton** publish/subscribe event bus that bridges background
sensor threads (wake_word, observer, ambient) with the asyncio-based
Brain and Textual TUI.

Design constraints solved here
-------------------------------
1. **Thread safety.**  Sensors run on daemon threads (Picovoice C-thread,
   observer thread).  ``emit_threadsafe()`` enqueues events via
   ``loop.call_soon_threadsafe`` so the emitter *never* touches the
   asyncio loop directly.

2. **TUI crash prevention.**  Textual raises ``ThreadError`` when UI is
   updated outside the main thread.  The bus dispatches events through
   ``asyncio.Queue`` → an async dispatcher task; TUI subscribers should
   further relay through ``app.call_from_thread(app.post_message, ...)``
   inside their own callback.

3. **Zero coupling.**  Sensors import *only* this module; they never
   import brain.py, tui/app.py, or each other.

Public API (all importable from ``bantz.core.event_bus``)
---------------------------------------------------------
.. code-block:: python

    from bantz.core.event_bus import bus, Event

    # ── Publishing (from any thread) ──────────────
    bus.emit_threadsafe("WAKE_WORD_DETECTED", confidence=0.93)

    # ── Publishing (from inside the asyncio loop) ─
    await bus.emit("AMBIENT_NOISE_HIGH", rms=450.2)

    # ── Subscribing (async) ───────────────────────
    async def on_wake(event: Event) -> None:
        print(event.name, event.data)

    bus.on("WAKE_WORD_DETECTED", on_wake)

    # ── Wildcard (catches everything) ─────────────
    bus.on("*", my_logger)

    # ── Unsubscribe ───────────────────────────────
    bus.off("WAKE_WORD_DETECTED", on_wake)

This module is **dependency-free** within the ``bantz`` package — it
imports nothing from ``bantz.*`` and can be safely imported anywhere.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

log = logging.getLogger("bantz.event_bus")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Event dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Event:
    """Immutable carrier for a single bus event.

    *name* is the channel string (e.g. ``"WAKE_WORD_DETECTED"``).
    *data* holds arbitrary keyword payload from the emitter.
    *ts* is auto-stamped at creation (``time.monotonic()``).
    """

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.monotonic)


# Type aliases for subscriber callbacks
SyncCallback = Callable[[Event], None]
AsyncCallback = Callable[[Event], Awaitable[None]]
Callback = SyncCallback | AsyncCallback

# Sentinel for wildcard subscriptions
_WILDCARD = "*"


# ═══════════════════════════════════════════════════════════════════════════
# 2. EventBus — thread-safe, asyncio-native singleton
# ═══════════════════════════════════════════════════════════════════════════

class EventBus:
    """Thread-safe publish/subscribe event bus.

    The bus owns an internal ``asyncio.Queue`` and a dispatcher task.
    Events land in the queue from *any* thread via ``emit_threadsafe()``
    or from the async loop via ``emit()``.  The dispatcher drains the
    queue and fans out to subscribers one-by-one (async or sync).

    Subscribers are called **in registration order**.  A failing
    subscriber logs the exception and does **not** prevent subsequent
    subscribers from running.
    """

    def __init__(self) -> None:
        # {event_name: [callback, ...]}  —  "*" key for wildcard
        self._subs: dict[str, list[Callback]] = {}
        self._lock = threading.Lock()  # protects _subs

        # Queue + dispatcher task (created lazily on first emit)
        self._queue: asyncio.Queue[Event] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dispatcher_task: asyncio.Task | None = None
        self._started = False

    # ── lifecycle ─────────────────────────────────────────────────────

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Attach the bus to an asyncio event loop and start dispatching.

        If *loop* is ``None``, uses ``asyncio.get_running_loop()``.
        Safe to call multiple times (idempotent).
        """
        if self._started:
            return
        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop
        self._queue = asyncio.Queue()
        self._dispatcher_task = loop.create_task(
            self._dispatcher(), name="bantz-event-bus",
        )
        self._started = True
        log.debug("EventBus bound to loop %s", id(loop))

    async def shutdown(self) -> None:
        """Drain remaining events, cancel the dispatcher, and reset.

        Call during application teardown.
        """
        if not self._started:
            return
        # Flush remaining items
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    event = self._queue.get_nowait()
                    await self._dispatch(event)
                except asyncio.QueueEmpty:
                    break
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
        self._started = False
        self._queue = None
        self._loop = None
        self._dispatcher_task = None
        log.debug("EventBus shut down")

    def reset(self) -> None:
        """Remove all subscribers and reset state.  For testing only."""
        with self._lock:
            self._subs.clear()
        if self._dispatcher_task is not None and not self._dispatcher_task.done():
            self._dispatcher_task.cancel()
        self._started = False
        self._queue = None
        self._loop = None
        self._dispatcher_task = None

    # ── subscribe / unsubscribe ───────────────────────────────────────

    def on(self, event_name: str, callback: Callback) -> Callback:
        """Register *callback* for *event_name*.

        Use ``"*"`` as event_name to catch **all** events (wildcard).
        Returns the callback for use as a decorator::

            @bus.on("WAKE_WORD_DETECTED")
            async def handle_wake(event: Event) -> None: ...
        """
        with self._lock:
            self._subs.setdefault(event_name, []).append(callback)
        return callback

    def off(self, event_name: str, callback: Callback) -> None:
        """Remove *callback* from *event_name*.  No-op if not registered."""
        with self._lock:
            cbs = self._subs.get(event_name)
            if cbs:
                try:
                    cbs.remove(callback)
                except ValueError:
                    pass
                if not cbs:
                    del self._subs[event_name]

    def clear(self, event_name: str | None = None) -> None:
        """Remove all subscribers for *event_name*, or all if ``None``."""
        with self._lock:
            if event_name is None:
                self._subs.clear()
            else:
                self._subs.pop(event_name, None)

    # ── emit (from asyncio loop) ──────────────────────────────────────

    async def emit(self, event_name: str, **data: Any) -> None:
        """Emit an event **from inside the running asyncio loop**.

        The event is placed on the internal queue; the dispatcher
        task picks it up and fans out to subscribers.
        """
        event = Event(name=event_name, data=data)
        if self._queue is not None:
            await self._queue.put(event)
        else:
            # Bus not started yet — dispatch inline as best-effort
            log.warning("EventBus.emit() called before bind_loop(); "
                        "dispatching inline for %r", event_name)
            await self._dispatch(event)

    # ── emit_threadsafe (from any thread) ─────────────────────────────

    def emit_threadsafe(self, event_name: str, **data: Any) -> None:
        """Emit an event **from any thread** (including C-extension threads).

        This is the primary entry point for sensors running on daemon
        threads (wake_word, observer).  It uses
        ``loop.call_soon_threadsafe`` to safely bridge into the async
        world, so the calling thread *never* blocks or corrupts the
        event loop.
        """
        event = Event(name=event_name, data=data)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._enqueue_sync, event)
        else:
            log.warning(
                "emit_threadsafe(%r) dropped — no active event loop",
                event_name,
            )

    def _enqueue_sync(self, event: Event) -> None:
        """Put *event* on the queue from within the loop thread.

        Called by ``call_soon_threadsafe`` — guaranteed to run on
        the loop thread so ``Queue.put_nowait`` is safe.
        """
        if self._queue is not None:
            self._queue.put_nowait(event)

    # ── dispatcher task ───────────────────────────────────────────────

    async def _dispatcher(self) -> None:
        """Long-running task: drain the queue and fan out to subscribers."""
        assert self._queue is not None
        try:
            while True:
                event = await self._queue.get()
                await self._dispatch(event)
                self._queue.task_done()
        except asyncio.CancelledError:
            return

    async def _dispatch(self, event: Event) -> None:
        """Fan out *event* to matching subscribers + wildcard."""
        with self._lock:
            specific = list(self._subs.get(event.name, []))
            wildcard = list(self._subs.get(_WILDCARD, []))
        for cb in specific + wildcard:
            try:
                ret = cb(event)
                if asyncio.iscoroutine(ret):
                    await ret
            except Exception:
                log.exception(
                    "Subscriber %r failed for event %r",
                    getattr(cb, "__name__", cb),
                    event.name,
                )

    # ── introspection ─────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True when the dispatcher task is active."""
        return self._started and self._dispatcher_task is not None \
            and not self._dispatcher_task.done()

    def subscriber_count(self, event_name: str | None = None) -> int:
        """Number of subscribers for *event_name*, or total if ``None``."""
        with self._lock:
            if event_name is not None:
                return len(self._subs.get(event_name, []))
            return sum(len(v) for v in self._subs.values())


# ═══════════════════════════════════════════════════════════════════════════
# 3. Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

bus = EventBus()
"""The global event bus instance.  Import and use directly::

    from bantz.core.event_bus import bus
    bus.emit_threadsafe("MY_EVENT", key="value")
"""
