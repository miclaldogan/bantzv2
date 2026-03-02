"""
Bantz v2 — Task Scheduler & Reminders (#61)
SQLite-backed persistent scheduler. Supports one-shot and recurring reminders.

DB table:
  reminders(id, title, fire_at, repeat, repeat_interval, created_at, fired, snoozed_until)

repeat modes: "none", "daily", "weekly", "weekdays", "custom"
repeat_interval: seconds (only used when repeat="custom")

Usage:
    from bantz.core.scheduler import scheduler

    # Add a one-shot reminder
    scheduler.add("call dentist", fire_at=datetime(2026, 3, 3, 15, 0))

    # Add a daily recurring reminder
    scheduler.add("check email", fire_at=datetime(2026, 3, 3, 9, 0), repeat="daily")

    # Check for due reminders (called periodically by app.py)
    due = scheduler.check_due()  # list[dict]

    # List upcoming reminders
    upcoming = scheduler.list_upcoming(limit=10)

    # Cancel a reminder
    scheduler.cancel(reminder_id)
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("bantz.scheduler")

_REPEAT_MODES = ("none", "daily", "weekly", "weekdays", "custom")


class Scheduler:
    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    # ── Init ──────────────────────────────────────────────────────────────

    def init(self, db_path: Path) -> None:
        """Call once at startup — uses the same DB as memory."""
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        log.debug("Scheduler initialized: %s", db_path)

    def _migrate(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                title            TEXT NOT NULL,
                fire_at          TEXT NOT NULL,
                repeat           TEXT NOT NULL DEFAULT 'none',
                repeat_interval  INTEGER DEFAULT 0,
                created_at       TEXT NOT NULL,
                fired            INTEGER NOT NULL DEFAULT 0,
                snoozed_until    TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reminders_fire
                ON reminders(fire_at, fired)
        """)

    # ── Public API ────────────────────────────────────────────────────────

    def add(
        self,
        title: str,
        fire_at: datetime,
        repeat: str = "none",
        repeat_interval: int = 0,
    ) -> int:
        """Create a new reminder. Returns the reminder ID."""
        if repeat not in _REPEAT_MODES:
            repeat = "none"

        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO reminders (title, fire_at, repeat, repeat_interval, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    title,
                    fire_at.isoformat(),
                    repeat,
                    repeat_interval,
                    datetime.now().isoformat(),
                ),
            )
            rid = cur.lastrowid
            log.info("Reminder #%d added: '%s' at %s (repeat=%s)", rid, title, fire_at, repeat)
            return rid

    def check_due(self) -> list[dict]:
        """Return all reminders that are due NOW and mark them fired.

        For recurring reminders, advance fire_at to the next occurrence instead
        of marking fired.
        """
        now = datetime.now()
        now_iso = now.isoformat()

        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM reminders
                   WHERE fired = 0
                     AND fire_at <= ?
                     AND (snoozed_until IS NULL OR snoozed_until <= ?)
                   ORDER BY fire_at""",
                (now_iso, now_iso),
            ).fetchall()

            due = []
            for row in rows:
                item = dict(row)
                due.append(item)

                repeat = item["repeat"]
                if repeat == "none":
                    self._conn.execute(
                        "UPDATE reminders SET fired = 1 WHERE id = ?",
                        (item["id"],),
                    )
                else:
                    # Advance to next occurrence
                    next_fire = self._next_occurrence(
                        datetime.fromisoformat(item["fire_at"]),
                        repeat,
                        item["repeat_interval"],
                    )
                    self._conn.execute(
                        "UPDATE reminders SET fire_at = ?, snoozed_until = NULL WHERE id = ?",
                        (next_fire.isoformat(), item["id"]),
                    )

            return due

    def list_upcoming(self, limit: int = 10) -> list[dict]:
        """Return upcoming (unfired) reminders sorted by fire_at."""
        rows = self._conn.execute(
            """SELECT * FROM reminders
               WHERE fired = 0
               ORDER BY fire_at
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_all(self, limit: int = 20) -> list[dict]:
        """Return all reminders (including fired), newest first."""
        rows = self._conn.execute(
            """SELECT * FROM reminders
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def cancel(self, reminder_id: int) -> bool:
        """Delete a reminder by ID. Returns True if found."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM reminders WHERE id = ?", (reminder_id,)
            )
            deleted = cur.rowcount > 0
            if deleted:
                log.info("Reminder #%d cancelled", reminder_id)
            return deleted

    def cancel_by_title(self, title: str) -> int:
        """Delete all reminders matching title (case-insensitive). Returns count."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM reminders WHERE lower(title) = lower(?)", (title,)
            )
            count = cur.rowcount
            if count:
                log.info("Cancelled %d reminder(s) matching '%s'", count, title)
            return count

    def snooze(self, reminder_id: int, minutes: int = 10) -> bool:
        """Snooze a reminder by N minutes."""
        until = datetime.now() + timedelta(minutes=minutes)
        with self._lock:
            cur = self._conn.execute(
                "UPDATE reminders SET snoozed_until = ?, fired = 0 WHERE id = ?",
                (until.isoformat(), reminder_id),
            )
            return cur.rowcount > 0

    def due_today(self) -> list[dict]:
        """Return reminders due today (for briefing)."""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        rows = self._conn.execute(
            """SELECT * FROM reminders
               WHERE fired = 0
                 AND fire_at >= ? AND fire_at < ?
               ORDER BY fire_at""",
            (today_start.isoformat(), today_end.isoformat()),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_active(self) -> int:
        """Count unfired reminders."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM reminders WHERE fired = 0"
        ).fetchone()
        return row[0] if row else 0

    def status_line(self) -> str:
        """Short status for --doctor."""
        if not self._conn:
            return "Scheduler: not initialized"
        n = self.count_active()
        return f"Scheduler: {n} active reminder(s)"

    # ── Format helpers ────────────────────────────────────────────────────

    def format_upcoming(self, limit: int = 10) -> str:
        """Human-readable list of upcoming reminders."""
        items = self.list_upcoming(limit)
        if not items:
            return "No upcoming reminders."

        lines = ["⏰ Upcoming reminders:"]
        for r in items:
            fire = datetime.fromisoformat(r["fire_at"])
            now = datetime.now()
            delta = fire - now

            if delta.total_seconds() < 0:
                time_str = "overdue"
            elif delta.days > 0:
                time_str = f"in {delta.days}d"
            elif delta.seconds >= 3600:
                time_str = f"in {delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
            else:
                time_str = f"in {delta.seconds // 60}m"

            repeat_tag = f" 🔁 {r['repeat']}" if r["repeat"] != "none" else ""
            lines.append(
                f"  #{r['id']} — {r['title']}  ⏱ {fire.strftime('%d %b %H:%M')} ({time_str}){repeat_tag}"
            )
        return "\n".join(lines)

    def format_due_today(self) -> Optional[str]:
        """Format reminders due today — used by briefing."""
        items = self.due_today()
        if not items:
            return None
        lines = []
        for r in items:
            fire = datetime.fromisoformat(r["fire_at"])
            repeat_tag = f" (🔁 {r['repeat']})" if r["repeat"] != "none" else ""
            lines.append(f"  • {fire.strftime('%H:%M')} — {r['title']}{repeat_tag}")
        return "⏰ Reminders today:\n" + "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _next_occurrence(
        current: datetime, repeat: str, interval: int
    ) -> datetime:
        """Calculate next fire time for a recurring reminder."""
        if repeat == "daily":
            return current + timedelta(days=1)
        elif repeat == "weekly":
            return current + timedelta(weeks=1)
        elif repeat == "weekdays":
            nxt = current + timedelta(days=1)
            while nxt.weekday() >= 5:  # skip Sat/Sun
                nxt += timedelta(days=1)
            return nxt
        elif repeat == "custom" and interval > 0:
            return current + timedelta(seconds=interval)
        else:
            return current + timedelta(days=1)  # fallback


scheduler = Scheduler()
