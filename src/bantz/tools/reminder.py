"""
Bantz v2 — Reminder Tool (#61)
CRUD operations for reminders, powered by the Scheduler.

Supports:
  - add: create a new reminder (one-shot or recurring)
  - list: show upcoming reminders
  - cancel: remove a reminder by ID or title
  - snooze: snooze a fired reminder by N minutes

Natural-language time parsing is handled by _parse_reminder_time().
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry


class ReminderTool(BaseTool):
    name = "reminder"
    description = (
        "Create, list, cancel, or snooze reminders and timers. "
        "Use for: remind me, set a reminder, set a timer, my reminders, "
        "cancel reminder, snooze, alarm."
    )
    risk_level = "safe"

    async def execute(self, **kwargs: Any) -> ToolResult:
        from bantz.core.scheduler import scheduler

        action = kwargs.get("action", "add")
        intent = kwargs.get("intent", "")

        if action == "list":
            return self._list(scheduler)
        elif action == "cancel":
            return self._cancel(scheduler, kwargs)
        elif action == "snooze":
            return self._snooze(scheduler, kwargs)
        else:
            return self._add(scheduler, kwargs, intent)

    # ── Add ───────────────────────────────────────────────────────────────

    def _add(self, scheduler, kwargs: dict, intent: str) -> ToolResult:
        title = kwargs.get("title", "").strip()
        time_str = kwargs.get("time", "").strip()
        repeat = kwargs.get("repeat", "none")

        # Parse from natural language if we have intent text
        if intent and (not title or not time_str):
            parsed_title, parsed_time, parsed_repeat = _parse_reminder_intent(intent)
            if not title:
                title = parsed_title
            if not time_str:
                time_str = parsed_time
            if repeat == "none" and parsed_repeat != "none":
                repeat = parsed_repeat

        if not title:
            title = "Reminder"

        # Resolve the fire time
        fire_at = _resolve_time(time_str) if time_str else None
        if not fire_at:
            # Default: 1 hour from now
            fire_at = datetime.now() + timedelta(hours=1)

        rid = scheduler.add(title, fire_at, repeat=repeat)
        time_display = fire_at.strftime("%d %b %H:%M")
        repeat_info = f" (repeats: {repeat})" if repeat != "none" else ""

        return ToolResult(
            success=True,
            output=f"⏰ Reminder #{rid} set: \"{title}\" at {time_display}{repeat_info}",
            data={"id": rid, "title": title, "fire_at": fire_at.isoformat(), "repeat": repeat},
        )

    # ── List ──────────────────────────────────────────────────────────────

    def _list(self, scheduler) -> ToolResult:
        text = scheduler.format_upcoming(limit=10)
        return ToolResult(success=True, output=text)

    # ── Cancel ────────────────────────────────────────────────────────────

    def _cancel(self, scheduler, kwargs: dict) -> ToolResult:
        rid = kwargs.get("id")
        title = kwargs.get("title", "")

        if rid:
            try:
                rid = int(rid)
            except (ValueError, TypeError):
                return ToolResult(success=False, output="", error="Invalid reminder ID")
            ok = scheduler.cancel(rid)
            if ok:
                return ToolResult(success=True, output=f"✓ Reminder #{rid} cancelled.")
            return ToolResult(success=False, output="", error=f"Reminder #{rid} not found.")

        if title:
            count = scheduler.cancel_by_title(title)
            if count:
                return ToolResult(
                    success=True,
                    output=f"✓ Cancelled {count} reminder(s) matching \"{title}\".",
                )
            return ToolResult(
                success=False, output="",
                error=f"No reminders found matching \"{title}\".",
            )

        return ToolResult(success=False, output="", error="Specify a reminder ID or title to cancel.")

    # ── Snooze ────────────────────────────────────────────────────────────

    def _snooze(self, scheduler, kwargs: dict) -> ToolResult:
        rid = kwargs.get("id")
        minutes = int(kwargs.get("minutes", 10))

        if not rid:
            return ToolResult(success=False, output="", error="Specify a reminder ID to snooze.")

        try:
            rid = int(rid)
        except (ValueError, TypeError):
            return ToolResult(success=False, output="", error="Invalid reminder ID")

        ok = scheduler.snooze(rid, minutes=minutes)
        if ok:
            return ToolResult(success=True, output=f"⏰ Reminder #{rid} snoozed for {minutes} minutes.")
        return ToolResult(success=False, output="", error=f"Reminder #{rid} not found.")


# ── Natural language parsing ──────────────────────────────────────────────────

def _parse_reminder_intent(text: str) -> tuple[str, str, str]:
    """
    Extract title, time, and repeat mode from natural language.
    Returns (title, time_str, repeat).

    Examples:
        "remind me to call dentist at 3pm" → ("call dentist", "3pm", "none")
        "remind me every day at 9am to check email" → ("check email", "9am", "daily")
        "set a timer for 30 minutes" → ("Timer", "30 minutes", "none")
        "remind me in 2 hours to buy groceries" → ("buy groceries", "2 hours", "none")
    """
    t = text.strip()
    repeat = "none"

    # Detect repeat mode
    if re.search(r"\bevery\s*day\b", t, re.IGNORECASE):
        repeat = "daily"
    elif re.search(r"\bevery\s*week\b", t, re.IGNORECASE):
        repeat = "weekly"
    elif re.search(r"\b(?:weekday|workday)s?\b", t, re.IGNORECASE):
        repeat = "weekdays"

    # Strip common prefixes
    cleaned = re.sub(
        r"^(?:remind\s+me\s+)?(?:to\s+)?(?:every\s*(?:day|week)\s+)?",
        "", t, flags=re.IGNORECASE,
    ).strip()

    # Extract "at HH:MM" or "at Xpm/am"
    time_str = ""
    time_match = re.search(
        r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
        cleaned, re.IGNORECASE,
    )
    if time_match:
        time_str = time_match.group(1).strip()
        cleaned = cleaned[:time_match.start()] + cleaned[time_match.end():]

    # Extract "in X minutes/hours"
    if not time_str:
        dur_match = re.search(
            r"\bin\s+(\d+)\s*(minute|min|hour|hr|second|sec)s?\b",
            cleaned, re.IGNORECASE,
        )
        if dur_match:
            time_str = f"{dur_match.group(1)} {dur_match.group(2)}"
            cleaned = cleaned[:dur_match.start()] + cleaned[dur_match.end():]

    # Extract "for X minutes" (timer style)
    if not time_str:
        timer_match = re.search(
            r"\bfor\s+(\d+)\s*(minute|min|hour|hr|second|sec)s?\b",
            cleaned, re.IGNORECASE,
        )
        if timer_match:
            time_str = f"{timer_match.group(1)} {timer_match.group(2)}"
            cleaned = cleaned[:timer_match.start()] + cleaned[timer_match.end():]

    # Extract "tomorrow at HH:MM"
    if not time_str:
        tmrw_match = re.search(
            r"\btomorrow\s+(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
            cleaned, re.IGNORECASE,
        )
        if tmrw_match:
            time_str = f"tomorrow {tmrw_match.group(1)}"
            cleaned = cleaned[:tmrw_match.start()] + cleaned[tmrw_match.end():]

    # Clean up title
    title = re.sub(r"\s+", " ", cleaned).strip()
    title = re.sub(r"^(?:to\s+|that\s+|about\s+)", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"^(?:set\s+a?\s*timer|set\s+a?\s*reminder)\s*", "", title, flags=re.IGNORECASE).strip()
    title = title.rstrip(".,!? ")

    if not title or len(title) < 2:
        title = "Reminder"

    return title, time_str, repeat


def _resolve_time(time_str: str) -> datetime | None:
    """
    Resolve a time string to an absolute datetime.

    Supports:
      - "3pm", "15:00", "3:30pm"
      - "30 minutes", "2 hours"
      - "tomorrow 3pm"
    """
    now = datetime.now()
    t = time_str.strip().lower()

    # "X minutes/hours/seconds" → relative
    dur_match = re.match(r"(\d+)\s*(minute|min|hour|hr|second|sec)s?$", t)
    if dur_match:
        amount = int(dur_match.group(1))
        unit = dur_match.group(2)
        if unit.startswith("hour") or unit.startswith("hr"):
            return now + timedelta(hours=amount)
        elif unit.startswith("sec"):
            return now + timedelta(seconds=amount)
        else:
            return now + timedelta(minutes=amount)

    # "tomorrow HH:MM" or "tomorrow Xpm"
    tmrw_match = re.match(r"tomorrow\s+(.+)$", t)
    if tmrw_match:
        base = now + timedelta(days=1)
        parsed = _parse_clock_time(tmrw_match.group(1))
        if parsed:
            return base.replace(hour=parsed[0], minute=parsed[1], second=0, microsecond=0)
        return base.replace(hour=9, minute=0, second=0, microsecond=0)

    # Absolute clock time: "3pm", "15:00", "3:30pm"
    parsed = _parse_clock_time(t)
    if parsed:
        result = now.replace(hour=parsed[0], minute=parsed[1], second=0, microsecond=0)
        # If the time is already past today, schedule for tomorrow
        if result <= now:
            result += timedelta(days=1)
        return result

    return None


def _parse_clock_time(s: str) -> tuple[int, int] | None:
    """Parse "3pm", "15:00", "3:30pm" → (hour, minute)."""
    s = s.strip()

    # "3:30pm" or "3:30 pm"
    m = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)?$", s, re.IGNORECASE)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if m.group(3):
            if m.group(3).lower() == "pm" and h < 12:
                h += 12
            elif m.group(3).lower() == "am" and h == 12:
                h = 0
        return (h, mn)

    # "3pm" or "3 pm"
    m = re.match(r"(\d{1,2})\s*(am|pm)$", s, re.IGNORECASE)
    if m:
        h = int(m.group(1))
        if m.group(2).lower() == "pm" and h < 12:
            h += 12
        elif m.group(2).lower() == "am" and h == 12:
            h = 0
        return (h, 0)

    # "15:00"
    m = re.match(r"(\d{1,2}):(\d{2})$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    return None


# Register
reminder_tool = ReminderTool()
registry.register(reminder_tool)
