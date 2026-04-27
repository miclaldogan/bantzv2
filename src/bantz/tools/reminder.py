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
        "Manage reminders and timers: create, list, cancel, snooze. "
        "Params: action (add|list|cancel|snooze), intent (str) = what to remind, "
        "id (str) = reminder ID for cancel/snooze. "
        "Supports time-based and location-based triggers. "
        "Use for: 'remind me to X', 'set a timer', 'list my reminders'."
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

    @staticmethod
    def _bridge_to_job_scheduler(
        title: str,
        fire_at: datetime,
        repeat: str = "none",
    ) -> str | None:
        """Forward reminder to APScheduler-based job_scheduler (#128).

        Returns the APScheduler job ID if successful, None otherwise.
        The old core/scheduler.py remains the source of truth for CRUD;
        this bridge ensures APScheduler also fires the notification.
        """
        try:
            from bantz.agent.job_scheduler import job_scheduler
            if not job_scheduler._started:
                return None
            return job_scheduler.add_reminder(title, fire_at, repeat=repeat)
        except Exception:
            return None

    # ── Add ───────────────────────────────────────────────────────────────

    def _add(self, scheduler, kwargs: dict, intent: str) -> ToolResult:
        title = kwargs.get("title", "").strip()
        time_str = kwargs.get("time", "").strip()
        repeat = kwargs.get("repeat", "none")
        place = kwargs.get("place", "").strip()

        # Parse from natural language if we have intent text
        if intent and (not title or (not time_str and not place)):
            parsed_title, parsed_time, parsed_repeat, parsed_place = _parse_reminder_intent(intent)
            if not title:
                title = parsed_title
            if not time_str and not place:
                time_str = parsed_time
                place = parsed_place
            if repeat == "none" and parsed_repeat != "none":
                repeat = parsed_repeat

        # Also try parsing time from raw title/intent if time_str still empty
        if not time_str and not place:
            raw_text = intent or title or ""
            _, fallback_time, _, _ = _parse_reminder_intent(raw_text)
            if fallback_time:
                time_str = fallback_time

        if not title:
            title = "Reminder"

        # ── Location-based reminder ──
        if place:
            place_key = _resolve_place(place)
            if not place_key:
                return ToolResult(
                    success=False, output="",
                    error=f"Unknown place: \"{place}\". Save it first with 'save here as {place}'.",
                )
            # Use a far-future fire_at as fallback expiry
            fire_at = datetime.now() + timedelta(days=365)
            rid = scheduler.add(title, fire_at, repeat=repeat, trigger_place=place_key)
            # Store reminder triple in KnowledgeGraph
            _store_reminder_kg(rid, title, trigger_place=place_key, repeat=repeat)
            # Bridge to APScheduler (#128)
            ReminderTool._bridge_to_job_scheduler(title, fire_at, repeat)
            return ToolResult(
                success=True,
                output=f"📍 Reminder #{rid} set: \"{title}\" — when you arrive at {place}",
                data={"id": rid, "title": title, "trigger_place": place_key, "repeat": repeat},
            )

        # ── Time-based reminder ──
        fire_at = _resolve_time(time_str) if time_str else None
        if not fire_at:
            # Last resort: default to 5 minutes from now (NOT 1 hour)
            fire_at = datetime.now() + timedelta(minutes=5)

        rid = scheduler.add(title, fire_at, repeat=repeat)
        time_display = fire_at.strftime("%d %b %H:%M")
        repeat_info = f" (repeats: {repeat})" if repeat != "none" else ""

        # Store reminder triple in KnowledgeGraph
        _store_reminder_kg(rid, title, fire_at=fire_at, repeat=repeat)
        # Bridge to APScheduler (#128)
        ReminderTool._bridge_to_job_scheduler(title, fire_at, repeat)

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

def _parse_reminder_intent(text: str) -> tuple[str, str, str, str]:
    """
    Extract title, time, repeat mode, and place from natural language.
    Returns (title, time_str, repeat, place).

    Examples:
        "remind me to call dentist at 3pm" → ("call dentist", "3pm", "none", "")
        "remind me every day at 9am to check email" → ("check email", "9am", "daily", "")
        "remind me to buy milk when I'm at the market" → ("buy milk", "", "none", "market")
        "when I get to university remind me about office hours" → ("about office hours", "", "none", "university")
    """
    t = text.strip()
    repeat = "none"
    place = ""

    # Detect repeat mode
    if re.search(r"\bevery\s*day\b", t, re.IGNORECASE):
        repeat = "daily"
    elif re.search(r"\bevery\s*week\b", t, re.IGNORECASE):
        repeat = "weekly"
    elif re.search(r"\b(?:weekday|workday)s?\b", t, re.IGNORECASE):
        repeat = "weekdays"

    # ── Detect location triggers ──
    # "when at X", "when I'm at X", "when I get to X", "when I arrive at X",
    # "when I'm near X", "when at the X"
    place_patterns = [
        r"when\s+(?:i(?:'m|\s+am))?\s*(?:at|near|around)\s+(?:the\s+)?(.+?)(?:\s+remind|\s+tell|\s*$)",
        r"when\s+(?:i\s+)?(?:get|arrive|go)\s+(?:to|at)\s+(?:the\s+)?(.+?)(?:\s+remind|\s*$|\s*,)",
    ]
    for pat in place_patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            place = m.group(1).strip().rstrip(".,!? ")
            t = t[:m.start()] + t[m.end():]
            break

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

    # Extract "a minute/an hour/a few minutes" (no digit)
    if not time_str:
        a_dur_match = re.search(
            r"\b(?:a|an|one)\s+(minute|min|hour|hr|second|sec)(?:\s+later|\s+from\s+now)?\b",
            cleaned, re.IGNORECASE,
        )
        if a_dur_match:
            time_str = f"1 {a_dur_match.group(1)}"
            cleaned = cleaned[:a_dur_match.start()] + cleaned[a_dur_match.end():]

    # Extract "X minutes/hours later" or "X min later"
    if not time_str:
        later_match = re.search(
            r"\b(\d+)\s*(minute|min|hour|hr|second|sec)s?\s+later\b",
            cleaned, re.IGNORECASE,
        )
        if later_match:
            time_str = f"{later_match.group(1)} {later_match.group(2)}"
            cleaned = cleaned[:later_match.start()] + cleaned[later_match.end():]

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

    return title, time_str, repeat, place


def _resolve_time(time_str: str) -> datetime | None:
    """
    Resolve a time string to an absolute datetime.

    Supports:
      - "3pm", "15:00", "3:30pm"
      - "30 minutes", "2 hours", "1 minute"
      - "a minute", "an hour" (article-based durations)
      - "tomorrow 3pm"
    """
    now = datetime.now()
    t = time_str.strip().lower()

    # "a minute", "an hour", "one minute" → map article to 1
    art_match = re.match(r"(?:a|an|one)\s+(minute|min|hour|hr|second|sec)s?$", t)
    if art_match:
        unit = art_match.group(1)
        if unit.startswith("hour") or unit.startswith("hr"):
            return now + timedelta(hours=1)
        elif unit.startswith("sec"):
            return now + timedelta(seconds=1)
        else:
            return now + timedelta(minutes=1)

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_place(name: str) -> str | None:
    """Resolve a place name to its key from the places database.

    Returns the place key if found, None otherwise.
    """
    try:
        from bantz.core.places import places
        all_places = places.all_places()
        name_lower = name.lower().strip()

        # Exact key match
        if name_lower in all_places:
            return name_lower

        # Label match (case-insensitive)
        for key, place in all_places.items():
            label = place.get("label", key).lower()
            if label == name_lower or name_lower in label or label in name_lower:
                return key

        return None
    except Exception:
        return None


def _store_reminder_kg(
    rid: int,
    title: str,
    fire_at: datetime | None = None,
    trigger_place: str | None = None,
    repeat: str = "none",
) -> None:
    """Store reminder as a KnowledgeGraph triple via MemPalace bridge."""
    try:
        from bantz.memory.bridge import palace_bridge
        if not palace_bridge or not palace_bridge.enabled:
            return

        kg = palace_bridge.kg
        if kg is None:
            return


        # Build a descriptive triple: User → set_reminder → title (with metadata)
        obj_parts = [title]
        if fire_at:
            obj_parts.append(f"at {fire_at.isoformat()}")
        if trigger_place:
            obj_parts.append(f"place:{trigger_place}")
        if repeat != "none":
            obj_parts.append(f"repeat:{repeat}")

        kg.add_triple(
            subject="User",
            predicate="set_reminder",
            obj=" | ".join(obj_parts),
        )
    except Exception:
        pass
