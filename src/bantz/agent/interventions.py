"""
Bantz — Proactive intervention queue (#126).

Bridges RL engine, observer, and scheduler into a unified intervention
system with priority queue, rate limiting, focus/quiet modes, and
explainability labels.

Architecture:
    RL Engine  (Q > threshold)  ──┐
    Observer   (critical/warn)  ──┼──→ InterventionQueue ──→ TUI toast
    Scheduler  (reminders)      ──┤         │
    System     (maintenance)    ──┘    Rate limiter (max N/hour)
                                            │
                                ┌───────────┼──────────┐
                                ▼           ▼          ▼
                             [Accept]   [Dismiss]   [Never]
                             reward(+1) reward(-0.5) blacklist
                                            │
                                     Auto-dismiss (TTL)
                                     → mild penalty (-0.1)
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class InterventionType(str, Enum):
    """Category of proactive intervention."""
    ERROR_ALERT = "error_alert"
    ROUTINE = "routine"
    REMINDER = "reminder"
    MAINTENANCE = "maintenance"
    PROACTIVE = "proactive"  # #167: proactive engagement
    HEALTH = "health"  # #168: health & break interventions


class Priority(int, Enum):
    """Intervention priority level (higher = more important)."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class Outcome(str, Enum):
    """How the user responded to an intervention."""
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    NEVER = "never"
    AUTO_DISMISSED = "auto_dismissed"
    PENDING = "pending"


# ═══════════════════════════════════════════════════════════════════════════
# Intervention dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Intervention:
    """A single proactive intervention."""

    type: InterventionType
    priority: Priority
    title: str
    reason: str                          # Explainability: why suggested
    source: str                          # "rl_engine", "observer", "scheduler", "system"
    action: Optional[str] = None         # Action enum value for RL feedback
    state_key: Optional[str] = None      # State key for RL feedback
    ttl: float = 20.0                    # Seconds before auto-dismiss
    created_at: float = field(default_factory=time.time)
    outcome: Outcome = Outcome.PENDING
    responded_at: Optional[float] = None
    id: int = 0                          # DB id after logging

    @property
    def expired(self) -> bool:
        """True if the intervention has exceeded its TTL."""
        return time.time() - self.created_at > self.ttl

    @property
    def age(self) -> float:
        """Seconds since creation."""
        return time.time() - self.created_at

    @property
    def remaining_ttl(self) -> float:
        """Seconds until expiry (may be negative)."""
        return self.ttl - self.age


# ═══════════════════════════════════════════════════════════════════════════
# Intervention Log (SQLite persistence)
# ═══════════════════════════════════════════════════════════════════════════


class InterventionLog:
    """Persist intervention outcomes for RL training data."""

    _CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS intervention_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            type         TEXT    NOT NULL,
            priority     INTEGER NOT NULL,
            title        TEXT    NOT NULL,
            reason       TEXT    NOT NULL,
            source       TEXT    NOT NULL,
            action       TEXT,
            state_key    TEXT,
            outcome      TEXT    NOT NULL,
            ttl          REAL    NOT NULL,
            created_at   REAL    NOT NULL,
            responded_at REAL
        )
    """

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None

    def init(self, db_path: str | Any) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(self._CREATE_SQL)
        self._conn.commit()

    def record(self, iv: Intervention) -> None:
        """Log an intervention and its outcome."""
        if not self._conn:
            return
        cur = self._conn.execute(
            """INSERT INTO intervention_log
               (type, priority, title, reason, source, action,
                state_key, outcome, ttl, created_at, responded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                iv.type.value,
                iv.priority.value,
                iv.title,
                iv.reason,
                iv.source,
                iv.action,
                iv.state_key,
                iv.outcome.value,
                iv.ttl,
                iv.created_at,
                iv.responded_at,
            ),
        )
        iv.id = cur.lastrowid or 0
        self._conn.commit()

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent logged interventions."""
        if not self._conn:
            return []
        cur = self._conn.execute(
            "SELECT * FROM intervention_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def total(self) -> int:
        if not self._conn:
            return 0
        cur = self._conn.execute("SELECT COUNT(*) FROM intervention_log")
        return cur.fetchone()[0]

    def outcome_counts(self) -> dict[str, int]:
        """Breakdown of outcomes."""
        if not self._conn:
            return {}
        cur = self._conn.execute(
            "SELECT outcome, COUNT(*) FROM intervention_log GROUP BY outcome"
        )
        return dict(cur.fetchall())

    def acceptance_rate(self) -> float:
        """Fraction of interventions that were accepted (excludes pending)."""
        if not self._conn:
            return 0.0
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM intervention_log WHERE outcome = 'accepted'"
        )
        accepted = cur.fetchone()[0]
        cur2 = self._conn.execute(
            "SELECT COUNT(*) FROM intervention_log WHERE outcome != 'pending'"
        )
        total = cur2.fetchone()[0]
        return round(accepted / total, 3) if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "total": self.total(),
            "outcomes": self.outcome_counts(),
            "acceptance_rate": self.acceptance_rate(),
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# ═══════════════════════════════════════════════════════════════════════════
# Intervention Queue
# ═══════════════════════════════════════════════════════════════════════════


class InterventionQueue:
    """Priority queue with rate limiting, focus mode, quiet mode, and TTL.

    Usage:
        queue.init(db_path, rate_limit=3, default_ttl=20)
        queue.push(Intervention(...))
        iv = queue.pop()          # honours rate limits & modes
        queue.respond(Outcome.ACCEPTED)
        queue.expire_active()     # auto-dismiss on TTL
    """

    def __init__(self) -> None:
        self._queue: list[Intervention] = []
        self._lock = threading.Lock()
        self._log = InterventionLog()
        self._active: Optional[Intervention] = None

        # Rate limiting
        self._rate_limit = 3
        self._rate_window: deque[float] = deque()  # timestamps of shown interventions

        # Modes
        self._quiet = False
        self._focus = False

        # Config
        self._default_ttl = 20.0
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def init(
        self,
        db_path: str | Any,
        *,
        rate_limit: int = 3,
        default_ttl: float = 20.0,
    ) -> None:
        if self._initialized:
            return
        self._log.init(str(db_path))
        self._rate_limit = rate_limit
        self._default_ttl = default_ttl
        self._initialized = True
        log.info(
            "InterventionQueue initialized (rate=%d/h, ttl=%.0fs)",
            rate_limit,
            default_ttl,
        )

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ── Push / Pop ────────────────────────────────────────────────────

    def push(self, iv: Intervention) -> bool:
        """Add an intervention. Returns False if dropped by mode filters."""
        if not self._initialized:
            return False

        # Quiet mode: only CRITICAL passes
        if self._quiet and iv.priority < Priority.CRITICAL:
            log.debug("Quiet mode: dropping '%s'", iv.title)
            return False

        # Focus mode: only HIGH+ passes
        if self._focus and iv.priority < Priority.HIGH:
            log.debug("Focus mode: dropping '%s' (priority=%s)", iv.title, iv.priority.name)
            return False

        if iv.ttl <= 0:
            iv.ttl = self._default_ttl

        with self._lock:
            self._queue.append(iv)
            # Sort: highest priority first, then oldest first (FIFO within priority)
            self._queue.sort(key=lambda x: (-x.priority.value, x.created_at))

        log.debug("Queued intervention: '%s' [%s/%s]", iv.title, iv.type.value, iv.priority.name)
        return True

    def pop(self) -> Optional[Intervention]:
        """Get the next intervention, respecting rate limits and expiry.

        Returns None if nothing is ready or rate limit is hit.
        """
        if not self._initialized:
            return None

        now = time.time()

        # Clean rate window
        cutoff = now - 3600
        while self._rate_window and self._rate_window[0] < cutoff:
            self._rate_window.popleft()

        with self._lock:
            # Expire stale entries in queue
            expired = [iv for iv in self._queue if iv.expired]
            for exp in expired:
                exp.outcome = Outcome.AUTO_DISMISSED
                exp.responded_at = time.time()
                self._log.record(exp)
            self._queue = [iv for iv in self._queue if not iv.expired]

            if not self._queue:
                return None

            candidate = self._queue[0]

            # Rate limit (CRITICAL bypasses)
            if candidate.priority < Priority.CRITICAL:
                if len(self._rate_window) >= self._rate_limit:
                    return None

            iv = self._queue.pop(0)

        self._rate_window.append(now)
        self._active = iv
        return iv

    # ── Response handling ─────────────────────────────────────────────

    def respond(self, outcome: Outcome) -> Optional[Intervention]:
        """Record user response to the active intervention.

        Returns the resolved intervention for RL feedback.
        """
        if not self._active:
            return None

        self._active.outcome = outcome
        self._active.responded_at = time.time()
        self._log.record(self._active)

        result = self._active
        self._active = None
        return result

    def expire_active(self) -> Optional[Intervention]:
        """Auto-dismiss the active intervention (TTL expired).

        Returns the expired intervention for mild RL penalty.
        """
        if not self._active:
            return None

        self._active.outcome = Outcome.AUTO_DISMISSED
        self._active.responded_at = time.time()
        self._log.record(self._active)

        result = self._active
        self._active = None
        return result

    # ── Active state ──────────────────────────────────────────────────

    @property
    def active(self) -> Optional[Intervention]:
        return self._active

    @property
    def has_active(self) -> bool:
        return self._active is not None

    # ── Mode toggles ─────────────────────────────────────────────────

    @property
    def quiet(self) -> bool:
        return self._quiet

    def set_quiet(self, enabled: bool) -> None:
        self._quiet = enabled
        log.info("Quiet mode: %s", "ON" if enabled else "OFF")

    @property
    def focus(self) -> bool:
        return self._focus

    def set_focus(self, enabled: bool) -> None:
        """Toggle focus mode — only HIGH/CRITICAL interventions pass.

        When app detection (issue #127) is implemented, this will be
        triggered automatically when the user is in a fullscreen app,
        IDE coding session, or media playback.
        """
        self._focus = enabled
        log.info("Focus mode: %s", "ON" if enabled else "OFF")

    # ── Query ─────────────────────────────────────────────────────────

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def rate_remaining(self) -> int:
        """How many more interventions can be shown this hour."""
        now = time.time()
        cutoff = now - 3600
        while self._rate_window and self._rate_window[0] < cutoff:
            self._rate_window.popleft()
        return max(0, self._rate_limit - len(self._rate_window))

    def stats(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "pending": self.pending_count(),
            "rate_remaining": self.rate_remaining(),
            "rate_limit": self._rate_limit,
            "quiet_mode": self._quiet,
            "focus_mode": self._focus,
            "has_active": self._active is not None,
            "default_ttl": self._default_ttl,
            "log": self._log.stats(),
        }

    def status_line(self) -> str:
        s = self.stats()
        parts = [
            f"pending={s['pending']}",
            f"rate={s['rate_remaining']}/{s['rate_limit']}",
            f"ttl={s['default_ttl']:.0f}s",
        ]
        if s["quiet_mode"]:
            parts.append("QUIET")
        if s["focus_mode"]:
            parts.append("FOCUS")
        log_stats = s.get("log", {})
        total = log_stats.get("total", 0)
        if total:
            acc = log_stats.get("acceptance_rate", 0)
            parts.append(f"logged={total}")
            parts.append(f"accept={acc:.0%}")
        return " | ".join(parts)

    # ── Teardown ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Expire active + pending, close log DB."""
        if self._active:
            self.expire_active()
        with self._lock:
            for iv in self._queue:
                iv.outcome = Outcome.AUTO_DISMISSED
                iv.responded_at = time.time()
                self._log.record(iv)
            self._queue.clear()
        self._log.close()
        self._initialized = False
        log.info("InterventionQueue closed")


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build interventions from various sources
# ═══════════════════════════════════════════════════════════════════════════

# Mapping of RL Action → user-friendly English label + emoji
ACTION_LABELS: dict[str, tuple[str, str]] = {
    "launch_docker":     ("🐳", "Launch Docker dev environment?"),
    "open_workspace":    ("📂", "Open your recent workspace?"),
    "open_browser":      ("🌐", "Open frequently used sites?"),
    "focus_music":       ("🎵", "Start focus music?"),
    "run_maintenance":   ("🧹", "Run system maintenance?"),
    "prepare_briefing":  ("📋", "Prepare your daily briefing?"),
    "suggest_break":     ("☕", "Time for a break?"),
    "daily_review":      ("📊", "Generate your daily review?"),
    "proactive_chat":    ("💬", "Check in with you?"),
    "health_break":      ("🏥", "Take a health break?"),
}

# Source labels for explainability
SOURCE_LABELS: dict[str, str] = {
    "rl_engine":  "RL",
    "observer":   "Observer",
    "scheduler":  "Scheduler",
    "system":     "System",
}


def intervention_from_rl(
    action_value: str,
    state_key: str,
    reason: str = "",
    ttl: float = 0,
) -> Intervention:
    """Create a ROUTINE intervention from an RL engine suggestion."""
    emoji, label = ACTION_LABELS.get(action_value, ("💡", f"Suggestion: {action_value}"))
    return Intervention(
        type=InterventionType.ROUTINE,
        priority=Priority.MEDIUM,
        title=f"{emoji} {label}",
        reason=reason or "Learned routine pattern",
        source="rl_engine",
        action=action_value,
        state_key=state_key,
        ttl=ttl,  # 0 → queue will apply default
    )


def intervention_from_observer(
    raw_text: str,
    severity: str,
    analysis: str = "",
    ttl: float = 0,
) -> Intervention:
    """Create an ERROR_ALERT intervention from the observer."""
    sev_upper = severity.upper()
    if sev_upper == "CRITICAL":
        pri = Priority.CRITICAL
        icon = "🚨"
        title = f"{icon} Terminal error detected"
    elif sev_upper == "WARNING":
        pri = Priority.HIGH
        icon = "⚠️"
        title = f"{icon} Terminal warning"
    else:
        pri = Priority.MEDIUM
        icon = "ℹ️"
        title = f"{icon} Terminal notice"

    body = raw_text[:300]
    reason_parts = [f"stderr: {severity}"]
    if analysis:
        reason_parts.append(analysis[:200])

    return Intervention(
        type=InterventionType.ERROR_ALERT,
        priority=pri,
        title=title,
        reason=" — ".join(reason_parts),
        source="observer",
        action=None,
        state_key=None,
        ttl=ttl,
    )


def intervention_from_reminder(
    title: str,
    repeat: str = "none",
    ttl: float = 0,
) -> Intervention:
    """Create a REMINDER intervention from the scheduler."""
    repeat_tag = f" (repeats {repeat})" if repeat != "none" else ""
    return Intervention(
        type=InterventionType.REMINDER,
        priority=Priority.HIGH,
        title=f"⏰ Reminder: {title}{repeat_tag}",
        reason="Scheduled reminder",
        source="scheduler",
        action=None,
        state_key=None,
        ttl=ttl,
    )


def intervention_from_system(
    title: str,
    reason: str,
    priority: Priority = Priority.LOW,
    ttl: float = 0,
) -> Intervention:
    """Create a MAINTENANCE/system intervention."""
    return Intervention(
        type=InterventionType.MAINTENANCE,
        priority=priority,
        title=f"🔧 {title}",
        reason=reason,
        source="system",
        action=None,
        state_key=None,
        ttl=ttl,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

intervention_queue = InterventionQueue()
