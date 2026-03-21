"""
Bantz — Mood State Machine (#135)

Drives TUI visual mood from system telemetry + AppDetector activity.
This module is **TUI-only** — it does NOT touch LLM prompts (see #169).

State machine:
    sleeping  (after midnight + IDLE 30min)
    chill     (CPU<20% + IDLE/BROWSING/ENTERTAINMENT)
    focused   (AppDetector → CODING | PRODUCTIVITY)
    busy      (CPU 50-80%)
    stressed  (CPU>80% or thermal_alert or observer errors in 5min)

Transition hysteresis: 10s sustained before mood actually changes.
This prevents rapid flickering on transient CPU spikes.

Mood history: SQLite rolling log (max 500 entries) for `bantz --mood-history`.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Mood enum
# ═══════════════════════════════════════════════════════════════════════════

class Mood(str, Enum):
    CHILL = "chill"
    FOCUSED = "focused"
    BUSY = "busy"
    STRESSED = "stressed"
    SLEEPING = "sleeping"


# ═══════════════════════════════════════════════════════════════════════════
# Mood display info
# ═══════════════════════════════════════════════════════════════════════════

MOOD_FACES: dict[Mood, str] = {
    Mood.CHILL:    "(◕‿◕)",
    Mood.FOCUSED:  "(•̀ᴗ•́)",
    Mood.BUSY:     "(⊙_⊙)",
    Mood.STRESSED: "(╥﹏╥)",
    Mood.SLEEPING: "(-.-)zzz",
}

MOOD_LABELS: dict[Mood, str] = {
    Mood.CHILL:    "chill",
    Mood.FOCUSED:  "focused",
    Mood.BUSY:     "busy",
    Mood.STRESSED: "stressed",
    Mood.SLEEPING: "sleeping",
}

MOOD_CSS_CLASS: dict[Mood, str] = {
    Mood.CHILL:    "mood-chill",
    Mood.FOCUSED:  "mood-focused",
    Mood.BUSY:     "mood-busy",
    Mood.STRESSED: "mood-stressed",
    Mood.SLEEPING: "mood-sleeping",
}

# All mood CSS classes for removal
ALL_MOOD_CLASSES = set(MOOD_CSS_CLASS.values())


# ═══════════════════════════════════════════════════════════════════════════
# Mood History — SQLite rolling log
# ═══════════════════════════════════════════════════════════════════════════

_MAX_HISTORY = 500  # rolling window — old entries pruned on insert


class MoodHistory:
    """SQLite-backed rolling mood transition log.

    Stores up to _MAX_HISTORY entries. Old entries are pruned automatically.
    No vector DB — simple, fast, queryable.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path
        self._initialized = False

    def init(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        from bantz.data.connection_pool import get_pool
        pool = get_pool(str(db_path))
        with pool.connection(write=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mood_history (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    mood       TEXT NOT NULL,
                    prev_mood  TEXT NOT NULL,
                    reason     TEXT NOT NULL DEFAULT '',
                    cpu_pct    REAL NOT NULL DEFAULT 0,
                    ram_pct    REAL NOT NULL DEFAULT 0,
                    activity   TEXT NOT NULL DEFAULT '',
                    timestamp  TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mood_ts
                    ON mood_history(timestamp)
            """)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def log_transition(
        self,
        mood: Mood,
        prev_mood: Mood,
        reason: str = "",
        cpu_pct: float = 0.0,
        ram_pct: float = 0.0,
        activity: str = "",
    ) -> None:
        """Record a mood transition."""
        if not self._initialized:
            return
        now = datetime.now().isoformat()
        from bantz.data.connection_pool import get_pool
        with get_pool().connection(write=True) as conn:
            conn.execute(
                """INSERT INTO mood_history
                       (mood, prev_mood, reason, cpu_pct, ram_pct, activity, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (mood.value, prev_mood.value, reason, cpu_pct, ram_pct, activity, now),
            )
            # Prune old entries beyond rolling window
            conn.execute(
                """DELETE FROM mood_history WHERE id NOT IN (
                       SELECT id FROM mood_history ORDER BY id DESC LIMIT ?
                   )""",
                (_MAX_HISTORY,),
            )

    def recent(self, hours: int = 24) -> list[dict]:
        """Get mood transitions from the last N hours."""
        if not self._initialized:
            return []
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        from bantz.data.connection_pool import get_pool
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT mood, prev_mood, reason, cpu_pct, ram_pct, activity, timestamp
                   FROM mood_history WHERE timestamp >= ? ORDER BY timestamp""",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]

    def count(self) -> int:
        """Total entries in the log."""
        if not self._initialized:
            return 0
        from bantz.data.connection_pool import get_pool
        with get_pool().connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM mood_history").fetchone()
            return row[0] if row else 0

    def summary_24h(self) -> dict[str, float]:
        """Time spent in each mood over the last 24 hours (in minutes)."""
        entries = self.recent(hours=24)
        if not entries:
            return {}
        durations: dict[str, float] = {}
        for i, e in enumerate(entries):
            ts = datetime.fromisoformat(e["timestamp"])
            if i + 1 < len(entries):
                next_ts = datetime.fromisoformat(entries[i + 1]["timestamp"])
            else:
                next_ts = datetime.now()
            delta_min = (next_ts - ts).total_seconds() / 60.0
            mood = e["mood"]
            durations[mood] = durations.get(mood, 0.0) + delta_min
        return durations


# ═══════════════════════════════════════════════════════════════════════════
# Mood State Machine
# ═══════════════════════════════════════════════════════════════════════════

_HYSTERESIS_SECONDS = 10.0  # sustain new mood for 10s before committing


class MoodStateMachine:
    """Computes Bantz's mood from telemetry + AppDetector.

    Design:
      - Reads TelemetryCollector.latest for CPU/RAM/thermal
      - Reads AppDetector.get_activity_category() for CODING/IDLE/etc.
      - Reads Observer.stats() for recent error count
      - Hysteresis: candidate must be sustained for 10s before becoming active
      - Logs transitions to MoodHistory (SQLite rolling log)
      - Does NOT affect LLM prompts — that's #169's job
    """

    def __init__(self) -> None:
        self._current: Mood = Mood.CHILL
        self._candidate: Mood = Mood.CHILL
        self._candidate_since: float = 0.0
        self._last_error_count: int = 0
        self._error_window_start: float = 0.0
        self._recent_errors: int = 0
        self._idle_since: float = time.monotonic()
        self.history = MoodHistory()

    @property
    def current(self) -> Mood:
        return self._current

    @property
    def face(self) -> str:
        return MOOD_FACES[self._current]

    @property
    def label(self) -> str:
        return MOOD_LABELS[self._current]

    @property
    def css_class(self) -> str:
        return MOOD_CSS_CLASS[self._current]

    def evaluate(
        self,
        cpu_pct: float = 0.0,
        ram_pct: float = 0.0,
        thermal_alert: bool = False,
        activity: str = "idle",
        observer_error_count: int = 0,
        hour: Optional[int] = None,
    ) -> Mood:
        """Evaluate mood from current system state.

        Call this every ~2s from the telemetry refresh cycle.
        Returns the active mood (may lag behind raw state due to hysteresis).
        """
        now = time.monotonic()
        if hour is None:
            hour = datetime.now().hour

        # Track error rate in 5-minute window
        if now - self._error_window_start > 300:
            self._error_window_start = now
            self._last_error_count = observer_error_count
            self._recent_errors = 0
        else:
            new_errors = observer_error_count - self._last_error_count
            if new_errors > 0:
                self._recent_errors += new_errors
                self._last_error_count = observer_error_count

        # Track idle time via activity
        if activity != "idle":
            self._idle_since = now
        idle_minutes = (now - self._idle_since) / 60.0

        # ── Determine raw mood ──────────────────────────────────────
        raw = self._compute_raw(
            cpu_pct, ram_pct, thermal_alert, activity,
            self._recent_errors, hour, idle_minutes,
        )

        # ── Hysteresis ──────────────────────────────────────────────
        if raw != self._candidate:
            self._candidate = raw
            self._candidate_since = now
        elif raw == self._current:
            # Already in this mood — reset candidate timer
            self._candidate_since = now

        # Commit if candidate sustained long enough
        if (
            raw != self._current
            and raw == self._candidate
            and (now - self._candidate_since) >= _HYSTERESIS_SECONDS
        ):
            prev = self._current
            self._current = raw
            reason = self._reason(
                cpu_pct, ram_pct, thermal_alert, activity,
                self._recent_errors, hour, idle_minutes,
            )
            log.info("Mood: %s → %s (%s)", prev.value, raw.value, reason)
            self.history.log_transition(
                mood=raw,
                prev_mood=prev,
                reason=reason,
                cpu_pct=cpu_pct,
                ram_pct=ram_pct,
                activity=activity,
            )

        return self._current

    def _compute_raw(
        self,
        cpu_pct: float,
        ram_pct: float,
        thermal_alert: bool,
        activity: str,
        recent_errors: int,
        hour: int,
        idle_minutes: float,
    ) -> Mood:
        """Pure function: inputs → mood (no side effects)."""
        # Priority order: stressed > sleeping > focused > busy > chill

        # STRESSED: CPU>80% OR thermal OR RAM>95% OR errors in 5min
        if cpu_pct > 80 or thermal_alert or ram_pct > 95 or recent_errors >= 3:
            return Mood.STRESSED

        # SLEEPING: after midnight & before 6 + idle 30min
        if (hour >= 0 and hour < 6) and idle_minutes >= 30:
            return Mood.SLEEPING

        # FOCUSED: user is coding or being productive (AppDetector)
        if activity in ("coding", "productivity"):
            return Mood.FOCUSED

        # BUSY: CPU 50-80%
        if cpu_pct >= 50:
            return Mood.BUSY

        # CHILL: everything else (low CPU, idle/browsing/entertainment)
        return Mood.CHILL

    def _reason(
        self,
        cpu_pct: float,
        ram_pct: float,
        thermal_alert: bool,
        activity: str,
        recent_errors: int,
        hour: int,
        idle_minutes: float,
    ) -> str:
        """Generate a human-readable reason for the mood."""
        parts = []
        if cpu_pct > 80:
            parts.append(f"CPU {cpu_pct:.0f}%")
        if ram_pct > 95:
            parts.append(f"RAM {ram_pct:.0f}%")
        if thermal_alert:
            parts.append("thermal")
        if recent_errors >= 3:
            parts.append(f"{recent_errors} errors")
        if activity in ("coding", "productivity"):
            parts.append(f"activity={activity}")
        if idle_minutes >= 30:
            parts.append(f"idle {idle_minutes:.0f}min")
        parts.append(f"hour={hour}")
        return ", ".join(parts) if parts else "baseline"


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

mood_machine = MoodStateMachine()
