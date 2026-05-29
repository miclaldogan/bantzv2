"""Bantz — Routing decision log (paper-1).

Captures every ``cot_route`` decision (fast-path, LLM attempt 1, fallback,
LLM attempt 2) to SQLite so that paper-1 can stratify hallucination rate
by routing confidence, route type, tool name, and attempt number.

The log is intentionally independent of the finalizer log; the eval view
in :mod:`bantz.core.eval_view` joins them by timestamp proximity.

Schema::

    route_log(
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp    TEXT NOT NULL,
      en_input     TEXT,
      route        TEXT,        -- 'tool' | 'planner' | 'chat' | NULL
      tool_name    TEXT,
      confidence   REAL,
      reasoning    TEXT,
      source       TEXT,        -- 'fast_reminder' | 'fast_chat' | 'cot' | 'cot_fallback' | 'cot_retry'
      attempt      INTEGER,     -- 0 for fast-paths, 1/2 for LLM attempts
      outcome      TEXT,        -- 'ok' | 'refusal' | 'low_confidence' | 'parse_error' | 'model_error'
      error        TEXT,
      thinking     TEXT         -- <thinking> block content if available
    )
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger("bantz.route_log")


def log_route(
    *,
    en_input: str,
    route: Optional[str],
    tool_name: Optional[str],
    confidence: Optional[float],
    reasoning: Optional[str],
    source: str,
    attempt: int,
    outcome: str,
    error: Optional[str] = None,
    thinking: Optional[str] = None,
) -> Optional[int]:
    """Write one route_log row. Returns the inserted row id, or None on failure.

    Best-effort: any exception is logged at DEBUG and swallowed. The routing
    pipeline must never fail because telemetry failed.
    """
    try:
        import sqlite3
        from bantz.core.memory import memory
        from bantz.data.connection_pool import get_pool

        if not memory._initialized:
            return None

        with get_pool().connection(write=True) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS route_log ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  timestamp TEXT NOT NULL,"
                "  en_input TEXT,"
                "  route TEXT,"
                "  tool_name TEXT,"
                "  confidence REAL,"
                "  reasoning TEXT,"
                "  source TEXT,"
                "  attempt INTEGER,"
                "  outcome TEXT,"
                "  error TEXT,"
                "  thinking TEXT"
                ")",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_route_log_ts"
                " ON route_log(timestamp)",
            )
            cur = conn.execute(
                "INSERT INTO route_log"
                "(timestamp, en_input, route, tool_name, confidence, reasoning,"
                " source, attempt, outcome, error, thinking)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now().isoformat(timespec="milliseconds"),
                    (en_input or "")[:500],
                    route,
                    tool_name,
                    confidence,
                    (reasoning or "")[:1000] if reasoning else None,
                    source,
                    attempt,
                    outcome,
                    (error or "")[:500] if error else None,
                    (thinking or "")[:4000] if thinking else None,
                ),
            )
            return cur.lastrowid
    except sqlite3.OperationalError as exc:
        log.debug("route_log write failed (db state): %s", exc)
    except Exception as exc:
        log.debug("route_log write failed: %s", exc)
    return None


def _extract_thinking(raw: str) -> str:
    """Return the inner content of the first <thinking>…</thinking> block, if any."""
    if not raw:
        return ""
    import re
    m = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else "")[:4000]
