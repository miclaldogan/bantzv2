"""Bantz — Paper-1 evaluation join.

Joins three independently-written tables — :data:`messages`,
:data:`hallucination_log` (finalizer audit), and :data:`route_log`
(routing decisions) — into a single per-turn dict the eval scripts
can consume.

The three writers run within the same user turn, milliseconds apart, but
none of them hard-links to the others. We do a timestamp-proximity join
in Python (correlated subqueries inside SQLite views are too brittle):

* For each assistant message, find the closest preceding
  :data:`hallucination_log` row within
  :data:`FINALIZER_JOIN_WINDOW_S` seconds.
* For that finalizer row, find the closest preceding
  :data:`route_log` row within :data:`ROUTE_JOIN_WINDOW_S` seconds.

Soft join is sufficient because the butler runs a single user turn at a
time — concurrent turns are not possible in the current architecture, so
ambiguous matches inside the window are extremely rare.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterator, Optional

log = logging.getLogger("bantz.eval_view")

# Window (seconds) for joining the assistant message to its finalizer
# log row. Both timestamps are written by the same brain.process() call
# within ~1-2s, so 10s is generous.
FINALIZER_JOIN_WINDOW_S: int = 10

# Window for joining the finalizer row back to its routing decision.
# cot_route runs first in the pipeline, finalize runs at the end of the
# same turn — typical gap is 1-5s on local Ollama, longer on slow models.
ROUTE_JOIN_WINDOW_S: int = 60


_DEPENDENT_TABLE_DDL = (
    # finalizer audit log — schema mirrors core.finalizer.log_finalizer_call
    """
    CREATE TABLE IF NOT EXISTS hallucination_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      user_input TEXT,
      tool_used TEXT,
      tool_output TEXT,
      response TEXT,
      confidence REAL NOT NULL,
      mode TEXT,
      flagged INTEGER
    )
    """,
    # routing decision log — schema mirrors core.route_log.log_route
    """
    CREATE TABLE IF NOT EXISTS route_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      en_input TEXT,
      route TEXT,
      tool_name TEXT,
      confidence REAL,
      reasoning TEXT,
      source TEXT,
      attempt INTEGER,
      outcome TEXT,
      error TEXT,
      thinking TEXT
    )
    """,
)


def ensure_view() -> bool:
    """Ensure the two dependent tables exist so joins never fail.

    Kept under the name ``ensure_view`` for API stability; there is no
    SQL view any more — the join is done in Python.

    Idempotent. Returns True on success, False if the DB is not ready.
    """
    try:
        from bantz.core.memory import memory
        from bantz.data.connection_pool import get_pool

        if not memory._initialized:
            return False
        with get_pool().connection(write=True) as conn:
            for ddl in _DEPENDENT_TABLE_DDL:
                conn.execute(ddl)
        return True
    except Exception as exc:
        log.debug("ensure_view failed: %s", exc)
        return False


def _ts_seconds(iso: str) -> float:
    """Convert an ISO-8601 timestamp string to epoch seconds (float)."""
    return datetime.fromisoformat(iso).timestamp()


def _nearest_preceding(
    target_ts: float,
    rows: list[dict],
    window_s: float,
    *,
    content_hint: Optional[str] = None,
    content_field: Optional[str] = None,
) -> Optional[dict]:
    """Return the best match with timestamp ≤ target_ts within window.

    Returns None if no row falls in [target_ts - window_s, target_ts].
    Rows must be pre-parsed with a ``_ts`` field (epoch seconds).

    When ``content_hint`` and ``content_field`` are given and two or more
    rows share the same timestamp resolution, the row whose
    ``content_field`` value starts with the same 32-char prefix as
    ``content_hint`` wins. This is a tie-breaker for legacy
    second-resolution timestamps where multiple writes can collide.
    """
    candidates: list[dict] = []
    for row in rows:
        delta = target_ts - row["_ts"]
        if delta < 0 or delta > window_s:
            continue
        candidates.append(row)
    if not candidates:
        return None
    candidates.sort(key=lambda r: -r["_ts"])
    if content_hint and content_field and len(candidates) > 1:
        prefix = content_hint[:32]
        for row in candidates:
            val = row.get(content_field) or ""
            if val.startswith(prefix):
                return row
    return candidates[0]


def _load_finalizer_rows() -> list[dict]:
    import sqlite3
    from bantz.data.connection_pool import get_pool
    with get_pool().connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT id, timestamp, user_input, tool_used, tool_output,"
            " response, confidence, flagged, mode"
            " FROM hallucination_log ORDER BY timestamp"
        )]
    for r in rows:
        try:
            r["_ts"] = _ts_seconds(r["timestamp"])
        except Exception:
            r["_ts"] = 0.0
    return rows


def _load_route_rows() -> list[dict]:
    import sqlite3
    from bantz.data.connection_pool import get_pool
    with get_pool().connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT id, timestamp, en_input, route, tool_name, confidence,"
            " reasoning, source, attempt, outcome, error"
            " FROM route_log ORDER BY timestamp"
        )]
    for r in rows:
        try:
            r["_ts"] = _ts_seconds(r["timestamp"])
        except Exception:
            r["_ts"] = 0.0
    return rows


def _build_eval_row(
    msg: dict, fin: Optional[dict], rt: Optional[dict],
) -> dict:
    """Compose one paper1_eval-shaped dict from raw join inputs."""
    return {
        "message_id": msg["id"],
        "conversation_id": msg["conversation_id"],
        "message_ts": msg["created_at"],
        "message_tool": msg["tool_used"],
        "response_text": msg["content"],
        "finalizer_id": fin["id"] if fin else None,
        "finalizer_ts": fin["timestamp"] if fin else None,
        "finalizer_user_input": fin["user_input"] if fin else None,
        "finalizer_tool": fin["tool_used"] if fin else None,
        "tool_output": fin["tool_output"] if fin else None,
        "finalizer_confidence": fin["confidence"] if fin else None,
        "finalizer_flagged": fin["flagged"] if fin else None,
        "finalizer_mode": fin["mode"] if fin else None,
        "route_id": rt["id"] if rt else None,
        "route_ts": rt["timestamp"] if rt else None,
        "route_en_input": rt["en_input"] if rt else None,
        "route": rt["route"] if rt else None,
        "route_tool": rt["tool_name"] if rt else None,
        "route_confidence": rt["confidence"] if rt else None,
        "route_source": rt["source"] if rt else None,
        "route_attempt": rt["attempt"] if rt else None,
        "route_outcome": rt["outcome"] if rt else None,
    }


def fetch_eval_rows(
    *,
    limit: int | None = None,
    flagged_only: bool = False,
    tool_filter: str | None = None,
) -> Iterator[dict]:
    """Yield joined eval rows, oldest first.

    Args:
        limit: max rows to yield (None for unbounded).
        flagged_only: if True, only rows where ``finalizer_flagged = 1``.
        tool_filter: if set, restrict to that tool name (matches either
            ``message_tool`` or ``finalizer_tool``).
    """
    if not ensure_view():
        return

    import sqlite3
    from bantz.data.connection_pool import get_pool

    finalizer_rows = _load_finalizer_rows()
    route_rows = _load_route_rows()

    sql = (
        "SELECT id, conversation_id, role, content, tool_used, created_at"
        " FROM messages WHERE role = 'assistant' ORDER BY created_at"
    )
    yielded = 0
    with get_pool().connection() as conn:
        conn.row_factory = sqlite3.Row
        for raw in conn.execute(sql):
            msg = dict(raw)
            try:
                msg_ts = _ts_seconds(msg["created_at"])
            except Exception:
                continue

            fin = _nearest_preceding(
                msg_ts, finalizer_rows, FINALIZER_JOIN_WINDOW_S,
                content_hint=msg["content"], content_field="response",
            )
            rt = None
            if fin is not None:
                rt = _nearest_preceding(
                    fin["_ts"], route_rows, ROUTE_JOIN_WINDOW_S,
                    content_hint=fin.get("user_input") or "",
                    content_field="en_input",
                )

            if flagged_only and (fin is None or not fin.get("flagged")):
                continue
            if tool_filter is not None:
                ftool = fin["tool_used"] if fin else None
                if msg["tool_used"] != tool_filter and ftool != tool_filter:
                    continue

            yield _build_eval_row(msg, fin, rt)
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def stats() -> dict:
    """Return summary stats over the join for paper-1 monitoring."""
    if not ensure_view():
        return {"available": False}

    rows = list(fetch_eval_rows())
    by_mode: dict[str, int] = {}
    flagged = 0
    joined_fin = 0
    joined_route = 0
    for r in rows:
        if r["finalizer_id"] is not None:
            joined_fin += 1
            mode = r["finalizer_mode"]
            if mode:
                by_mode[mode] = by_mode.get(mode, 0) + 1
            if r["finalizer_flagged"]:
                flagged += 1
        if r["route_id"] is not None:
            joined_route += 1

    return {
        "available": True,
        "assistant_messages": len(rows),
        "joined_to_finalizer": joined_fin,
        "joined_to_route": joined_route,
        "flagged_finalizer_rows": flagged,
        "finalizer_mode_counts": by_mode,
    }
