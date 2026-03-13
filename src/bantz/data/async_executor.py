"""
Bantz — AsyncDBExecutor: non-blocking DB access for async callers (#224)

Wraps blocking ``SQLitePool.connection()`` calls in a
``ThreadPoolExecutor`` so that coroutines never stall the
asyncio event loop.

Architecture notes
------------------
* ``max_workers=5`` matches ``SQLitePool``'s 5 pre-created connections.
  Each worker thread checks out **one** pooled connection, executes the
  supplied callable, and returns the connection — so the thread count
  and pool size stay in balance.

* ``run_read`` / ``run_write`` accept a plain ``Callable[[Connection], T]``
  and return an ``Awaitable[T]``.  The callable runs inside the pool's
  ``connection()`` context manager on a worker thread.

* Callers do **not** need to touch ``get_pool()`` or manage connections
  themselves — this executor is the single async bridge.

Closes #224 (Part 1-B of #218).
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from bantz.data.connection_pool import get_pool

log = logging.getLogger("bantz.data.async_executor")

T = TypeVar("T")

# Module-level executor — created once, shared across all callers.
_executor: ThreadPoolExecutor | None = None
_MAX_WORKERS = 5  # mirrors SQLitePool's max_connections


def _get_executor() -> ThreadPoolExecutor:
    """Lazy-init the module-level ``ThreadPoolExecutor``."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_MAX_WORKERS,
            thread_name_prefix="bantz-db",
        )
        log.info("AsyncDBExecutor ready  workers=%d", _MAX_WORKERS)
    return _executor


# ── public async API ──────────────────────────────────────────────────


async def run_read(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute *fn(conn, \\*args, \\*\\*kwargs)* on a **read** connection.

    The callable receives an open ``sqlite3.Connection`` as its first
    argument and must return the desired value.  The connection is
    automatically returned to the pool after *fn* completes.

    Example::

        rows = await run_read(lambda conn: conn.execute(
            "SELECT * FROM conversations ORDER BY ts DESC LIMIT 10"
        ).fetchall())
    """
    loop = asyncio.get_running_loop()

    def _work() -> T:
        with get_pool().connection(write=False) as conn:
            return fn(conn, *args, **kwargs)

    return await loop.run_in_executor(_get_executor(), _work)


async def run_write(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute *fn(conn, \\*args, \\*\\*kwargs)* on a **write** connection.

    Acquires the pool's ``_write_lock`` so only one writer at a time.
    Auto-commits on success, auto-rolls-back on exception.

    Example::

        await run_write(lambda conn: conn.execute(
            "INSERT INTO conversations(role, content) VALUES (?, ?)",
            ("user", msg),
        ))
    """
    loop = asyncio.get_running_loop()

    def _work() -> T:
        with get_pool().connection(write=True) as conn:
            return fn(conn, *args, **kwargs)

    return await loop.run_in_executor(_get_executor(), _work)


async def run_in_db(
    fn: Callable[..., T],
    *args: Any,
    write: bool = False,
    **kwargs: Any,
) -> T:
    """Unified entry point — delegates to :func:`run_read` or :func:`run_write`.

    Parameters
    ----------
    fn : callable
        ``fn(conn, *args, **kwargs)`` — receives an open connection.
    write : bool
        Pass ``True`` to acquire the write-lock.
    """
    if write:
        return await run_write(fn, *args, **kwargs)
    return await run_read(fn, *args, **kwargs)


# ── lifecycle ─────────────────────────────────────────────────────────


def shutdown(wait: bool = True) -> None:
    """Shut down the thread-pool.  Called during application teardown."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait)
        _executor = None
        log.info("AsyncDBExecutor shut down")
