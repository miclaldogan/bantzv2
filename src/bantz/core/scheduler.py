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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from bantz.data.store import ReminderStore

log = logging.getLogger("bantz.scheduler")

_REPEAT_MODES = ("none", "daily", "weekly", "weekdays", "custom")


class Scheduler(ReminderStore):
    def __init__(self) -> None:
        self._initialized = False

    # ── Init ──────────────────────────────────────────────────────────────

    def init(self, db_path: Path) -> None:
        """Call once at startup — uses the same DB as memory."""
        from bantz.data.connection_pool import get_pool
        get_pool(str(db_path))
        self._migrate()
        self._initialized = True
        log.debug("Scheduler initialized: %s", db_path)

    def _migrate(self) -> None:
        from bantz.data.connection_pool import get_pool
        with get_pool().connection(write=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    title            TEXT NOT NULL,
                    fire_at          TEXT NOT NULL,
                    repeat           TEXT NOT NULL DEFAULT 'none',
                    repeat_interval  INTEGER DEFAULT 0,
                    created_at       TEXT NOT NULL,
                    fired            INTEGER NOT NULL DEFAULT 0,
                    snoozed_until    TEXT,
                    trigger_place    TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reminders_fire
                    ON reminders(fire_at, fired)
            """)
            # Migration: add trigger_place column to existing tables
            try:
                conn.execute("ALTER TABLE reminders ADD COLUMN trigger_place TEXT")
            except Exception:
                pass  # column already exists

    # ── Public API ────────────────────────────────────────────────────────

    def add(
        self,
        title: str,
        fire_at: datetime,
        repeat: str = "none",
        repeat_interval: int = 0,
        trigger_place: str | None = None,
    ) -> int:
        """Create a new reminder. Returns the reminder ID.

        If trigger_place is set, the reminder fires when the user enters
        that named place (from places.py) instead of at fire_at.
        fire_at is still stored as a fallback expiry.
        """
        if repeat not in _REPEAT_MODES:
            repeat = "none"

        from bantz.data.connection_pool import get_pool
        with get_pool().connection(write=True) as conn:
            cur = conn.execute(
                """INSERT INTO reminders
                       (title, fire_at, repeat, repeat_interval, created_at, trigger_place)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    title,
                    fire_at.isoformat(),
                    repeat,
                    repeat_interval,
                    datetime.now().isoformat(),
                    trigger_place,
                ),
            )
            rid = cur.lastrowid
            if trigger_place:
                log.info("Reminder #%d added: '%s' when at '%s' (repeat=%s)",
                         rid, title, trigger_place, repeat)
            else:
                log.info("Reminder #%d added: '%s' at %s (repeat=%s)",
                         rid, title, fire_at, repeat)
            return rid

    def check_due(self) -> list[dict]:
        """Return all reminders that are due NOW and mark them fired.

        For recurring reminders, advance fire_at to the next occurrence instead
        of marking fired.
        """
        now = datetime.now()
        now_iso = now.isoformat()

        from bantz.data.connection_pool import get_pool
        with get_pool().connection(write=True) as conn:
            rows = conn.execute(
                """SELECT * FROM reminders
                   WHERE fired = 0
                     AND fire_at <= ?
                     AND (snoozed_until IS NULL OR snoozed_until <= ?)
                   ORDER BY fire_at""",
                (now_iso, now_iso),
            ).fetchall()

            due = []
            one_offs = []
            repeating = []
            for row in rows:
                item = dict(row)
                due.append(item)

                repeat = item["repeat"]
                if repeat == "none":
                    one_offs.append((item["id"],))
                else:
                    # Advance to next occurrence
                    next_fire = self._next_occurrence(
                        datetime.fromisoformat(item["fire_at"]),
                        repeat,
                        item["repeat_interval"],
                    )
                    repeating.append((next_fire.isoformat(), item["id"]))

            # ⚡ Bolt: Replace N+1 queries with bulk operations
            if one_offs:
                conn.executemany(
                    "UPDATE reminders SET fired = 1 WHERE id = ?",
                    one_offs,
                )
            if repeating:
                conn.executemany(
                    "UPDATE reminders SET fire_at = ?, snoozed_until = NULL WHERE id = ?",
                    repeating,
                )

            return due

    def list_upcoming(self, limit: int = 10) -> list[dict]:
        """Return upcoming (unfired) reminders sorted by fire_at."""
        from bantz.data.connection_pool import get_pool
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT * FROM reminders
                   WHERE fired = 0
                   ORDER BY fire_at
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_all(self, limit: int = 20) -> list[dict]:
        """Return all reminders (including fired), newest first."""
        from bantz.data.connection_pool import get_pool
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT * FROM reminders
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def cancel(self, reminder_id: int) -> bool:
        """Delete a reminder by ID. Returns True if found."""
        from bantz.data.connection_pool import get_pool
        with get_pool().connection(write=True) as conn:
            cur = conn.execute(
                "DELETE FROM reminders WHERE id = ?", (reminder_id,)
            )
            deleted = cur.rowcount > 0
            if deleted:
                log.info("Reminder #%d cancelled", reminder_id)
            return deleted

    def cancel_by_title(self, title: str) -> int:
        """Delete all reminders matching title (case-insensitive). Returns count."""
        from bantz.data.connection_pool import get_pool
        with get_pool().connection(write=True) as conn:
            cur = conn.execute(
                "DELETE FROM reminders WHERE lower(title) = lower(?)", (title,)
            )
            count = cur.rowcount
            if count:
                log.info("Cancelled %d reminder(s) matching '%s'", count, title)
            return count

    def snooze(self, reminder_id: int, minutes: int = 10) -> bool:
        """Snooze a reminder by N minutes."""
        until = datetime.now() + timedelta(minutes=minutes)
        from bantz.data.connection_pool import get_pool
        with get_pool().connection(write=True) as conn:
            cur = conn.execute(
                "UPDATE reminders SET snoozed_until = ?, fired = 0 WHERE id = ?",
                (until.isoformat(), reminder_id),
            )
            return cur.rowcount > 0

    def due_today(self) -> list[dict]:
        """Return reminders due today (for briefing)."""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        from bantz.data.connection_pool import get_pool
        with get_pool().connection() as conn:
            rows = conn.execute(
                """SELECT * FROM reminders
                   WHERE fired = 0
                     AND fire_at >= ? AND fire_at < ?
                   ORDER BY fire_at""",
                (today_start.isoformat(), today_end.isoformat()),
            ).fetchall()
            return [dict(r) for r in rows]

    def check_place_due(self, place_key: str) -> list[dict]:
        """Return location-triggered reminders for the given place and mark fired.

        Called by places.py when the user enters a known place.
        """
        from bantz.data.connection_pool import get_pool
        with get_pool().connection(write=True) as conn:
            rows = conn.execute(
                """SELECT * FROM reminders
                   WHERE fired = 0
                     AND trigger_place = ?
                     AND (snoozed_until IS NULL OR snoozed_until <= ?)
                   ORDER BY created_at""",
                (place_key, datetime.now().isoformat()),
            ).fetchall()

            due = []
            one_offs = []
            for row in rows:
                item = dict(row)
                due.append(item)

                repeat = item["repeat"]
                if repeat == "none":
                    one_offs.append((item["id"],))
                # Recurring location reminders stay active

            # ⚡ Bolt: Replace N+1 queries with bulk operations
            if one_offs:
                conn.executemany(
                    "UPDATE reminders SET fired = 1 WHERE id = ?",
                    one_offs,
                )
            return due

    def count_active(self) -> int:
        """Count unfired reminders."""
        from bantz.data.connection_pool import get_pool
        with get_pool().connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM reminders WHERE fired = 0"
            ).fetchone()
            return row[0] if row else 0

    def status_line(self) -> str:
        """Short status for --doctor."""
        if not self._initialized:
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
            place = r.get("trigger_place")
            if place:
                lines.append(
                    f"  #{r['id']} — {r['title']}  📍 when at {place}{repeat_tag}"
                )
            else:
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
