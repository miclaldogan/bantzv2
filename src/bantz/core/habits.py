"""
Bantz v2 — Habit Engine

Mines usage patterns from the SQLite memory DB.
No new tables — queries the existing messages table.

Usage:
    from bantz.core.habits import habits
    top = habits.top_tools_for_segment("morning")
    should = habits.should_add_to_briefing("weather")
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from bantz.config import config

# Time segment boundaries (hour ranges)
_SEGMENTS: dict[str, tuple[int, int]] = {
    "late_night": (0, 6),
    "morning":    (6, 12),
    "afternoon":  (12, 17),
    "evening":    (17, 21),
    "night":      (21, 24),
}


class HabitEngine:
    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            db_path = config.db_path
            if not db_path.exists():
                raise RuntimeError(f"DB not found: {db_path}")
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # ── Core queries ──────────────────────────────────────────────────────

    def top_tools_for_segment(
        self, segment: str, n: int = 3, days: int = 14
    ) -> list[dict]:
        """
        Top N tools used in a time segment over the last `days` days.
        Returns: [{"tool": "weather", "count": 12}, ...]
        """
        conn = self._ensure_conn()
        hour_start, hour_end = _SEGMENTS.get(segment, (0, 24))
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        rows = conn.execute(
            """
            SELECT tool_used, COUNT(*) as cnt
            FROM messages
            WHERE role = 'assistant'
              AND tool_used IS NOT NULL
              AND tool_used != 'startup'
              AND created_at > ?
              AND CAST(strftime('%H', created_at) AS INTEGER) >= ?
              AND CAST(strftime('%H', created_at) AS INTEGER) < ?
            GROUP BY tool_used
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (cutoff, hour_start, hour_end, n),
        ).fetchall()

        return [{"tool": r["tool_used"], "count": r["cnt"]} for r in rows]

    def should_add_to_briefing(self, tool: str, days: int = 7) -> bool:
        """
        Returns True if `tool` was used in morning segment (06-12)
        on 3+ distinct days in the last `days` days.
        """
        conn = self._ensure_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        row = conn.execute(
            """
            SELECT COUNT(DISTINCT DATE(created_at)) as day_count
            FROM messages
            WHERE role = 'assistant'
              AND tool_used = ?
              AND created_at > ?
              AND CAST(strftime('%H', created_at) AS INTEGER) >= 6
              AND CAST(strftime('%H', created_at) AS INTEGER) < 12
            """,
            (tool, cutoff),
        ).fetchone()

        return (row["day_count"] or 0) >= 3

    def analyze(self, days: int = 14) -> dict:
        """
        Full pattern analysis across all segments.
        Returns: {
            "segments": {"morning": [...], "afternoon": [...], ...},
            "briefing_candidates": ["weather", "news"],
            "top_overall": [...],
            "total_interactions": int,
        }
        """
        conn = self._ensure_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Per-segment breakdown
        segments = {}
        for seg in ("morning", "afternoon", "evening", "night"):
            segments[seg] = self.top_tools_for_segment(seg, n=5, days=days)

        # Briefing candidates: tools used 3+ morning days
        all_tools = conn.execute(
            """
            SELECT DISTINCT tool_used FROM messages
            WHERE role = 'assistant'
              AND tool_used IS NOT NULL
              AND tool_used != 'startup'
              AND created_at > ?
            """,
            (cutoff,),
        ).fetchall()

        briefing_candidates = [
            r["tool_used"]
            for r in all_tools
            if self.should_add_to_briefing(r["tool_used"], days)
        ]

        # Top overall
        top_overall = conn.execute(
            """
            SELECT tool_used, COUNT(*) as cnt
            FROM messages
            WHERE role = 'assistant'
              AND tool_used IS NOT NULL
              AND tool_used != 'startup'
              AND created_at > ?
            GROUP BY tool_used
            ORDER BY cnt DESC
            LIMIT 10
            """,
            (cutoff,),
        ).fetchall()

        # Total interactions
        total = conn.execute(
            """
            SELECT COUNT(*) FROM messages
            WHERE role = 'user' AND created_at > ?
            """,
            (cutoff,),
        ).fetchone()[0]

        return {
            "segments": segments,
            "briefing_candidates": briefing_candidates,
            "top_overall": [
                {"tool": r["tool_used"], "count": r["cnt"]} for r in top_overall
            ],
            "total_interactions": total,
        }

    def recurring_patterns(self, days: int = 7) -> list[dict]:
        """
        Find tools used at the same hour on 3+ distinct days.
        Returns: [{"tool": "weather", "hour": 8, "days": 5}, ...]
        """
        conn = self._ensure_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        rows = conn.execute(
            """
            SELECT tool_used,
                   CAST(strftime('%H', created_at) AS INTEGER) as hour,
                   COUNT(DISTINCT DATE(created_at)) as day_count
            FROM messages
            WHERE role = 'assistant'
              AND tool_used IS NOT NULL
              AND tool_used != 'startup'
              AND created_at > ?
            GROUP BY tool_used, hour
            HAVING day_count >= 3
            ORDER BY day_count DESC
            """,
            (cutoff,),
        ).fetchall()

        return [
            {"tool": r["tool_used"], "hour": r["hour"], "days": r["day_count"]}
            for r in rows
        ]

    def status_line(self) -> str:
        """Short summary for --doctor."""
        try:
            data = self.analyze(days=7)
            total = data["total_interactions"]
            candidates = data["briefing_candidates"]
            if not total:
                return "insufficient data (no usage yet)"
            line = f"{total} interaction(s) (7 days)"
            if candidates:
                line += f"  |  briefing candidates: {', '.join(candidates)}"
            return line
        except Exception:
            return "analysis unavailable"


habits = HabitEngine()
