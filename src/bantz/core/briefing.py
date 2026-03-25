"""
Bantz v2 — Daily Briefing
Parallel API calls: calendar + classroom + gmail + weather + schedule.
Any service failure is isolated — others still show.

Overnight poll integration (#132):
    When overnight poll data is cached in KV store, the briefing prefers
    that data over live API calls (zero extra API cost at wake-up time).
    Auth errors are reported gracefully (Rec #4).

Usage:
    from bantz.core.briefing import briefing
    text = await briefing.generate()
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from bantz.core.schedule import schedule
from bantz.core.time_context import time_ctx
from bantz.core.profile import profile as _profile
from bantz.core.habits import habits as _habits

log = logging.getLogger(__name__)


class Briefing:

    async def generate(self) -> str:
        now = datetime.now()
        tc = time_ctx.snapshot()
        sections = _profile.briefing_sections  # user-selected sections

        # ── Check for cached overnight poll data (#132) ───────────────────
        overnight = self._read_overnight_cache()

        # Fire all external calls in parallel (skip disabled sections)
        # If overnight cache has data for a source, use it instead of live call
        weather_coro = self._get_weather() if "weather" in sections else asyncio.sleep(0)

        if overnight.get("calendar") and "calendar" in sections:
            calendar_coro = self._wrap(self._calendar_from_cache(overnight["calendar"]))
        else:
            calendar_coro = self._get_calendar(now) if "calendar" in sections else asyncio.sleep(0)

        if overnight.get("gmail") and "mail" in sections:
            gmail_coro = self._wrap(self._gmail_from_cache(overnight["gmail"]))
        else:
            gmail_coro = self._get_gmail() if "mail" in sections else asyncio.sleep(0)

        if overnight.get("classroom") and "classroom" in sections:
            classroom_coro = self._wrap(self._classroom_from_cache(overnight["classroom"]))
        else:
            classroom_coro = self._get_classroom() if "classroom" in sections else asyncio.sleep(0)

        results = await asyncio.gather(
            weather_coro,
            calendar_coro,
            gmail_coro,
            classroom_coro,
            return_exceptions=True,
        )
        weather_str, calendar_str, gmail_str, classroom_str = [
            r if isinstance(r, str) else None for r in results
        ]

        # Schedule is local — no network, no failure
        schedule_str = self._get_schedule(now) if "schedule" in sections else None
        next_class_str = await self._get_next_class(now) if "schedule" in sections else None
        habit_str = self._get_habit_hint(now) if "habits" in sections else None
        reminder_str = self._get_reminders()

        # Clear overnight cache after consumption
        if overnight:
            self._clear_overnight_cache()

        return self._format(
            tc=tc,
            now=now,
            weather=weather_str,
            calendar=calendar_str,
            gmail=gmail_str,
            classroom=classroom_str,
            schedule=schedule_str,
            next_class=next_class_str,
            habits=habit_str,
            reminders=reminder_str,
        )

    # ── Overnight cache readers (#132) ────────────────────────────────────

    @staticmethod
    async def _wrap(value):
        """Wrap a sync return value as an awaitable for asyncio.gather."""
        return value

    def _read_overnight_cache(self) -> dict:
        """Read overnight poll data from KV store (if available)."""
        try:
            from bantz.agent.workflows.overnight_poll import read_overnight_data
            data = read_overnight_data()
            if data.get("last_poll"):
                return data
        except Exception:
            pass
        return {}

    def _clear_overnight_cache(self) -> None:
        """Clear overnight data after briefing consumes it."""
        try:
            from bantz.agent.workflows.overnight_poll import clear_overnight_data
            clear_overnight_data()
        except Exception:
            pass

    def _gmail_from_cache(self, data: dict) -> Optional[str]:
        """Format Gmail data from overnight cache.
        Rec #4: Gracefully reports auth errors."""
        status = data.get("status", "")
        if status == "auth_error":
            return ("I couldn't check your email overnight — "
                    "Google revoked my access. Please re-authenticate Gmail.")
        if status != "ok":
            return None
        d = data.get("data", {})
        unread = d.get("unread", 0)
        urgent_count = d.get("urgent_count", 0)
        if unread == 0:
            return "Inbox is clean"
        parts = [f"{unread} unread emails"]
        if urgent_count:
            urgent = d.get("urgent", [])
            subjects = [u.get("subject", "?")[:60] for u in urgent[:3]]
            parts.append(f"🚨 {urgent_count} urgent: " + ", ".join(subjects))
        return "  ".join(parts)

    def _calendar_from_cache(self, data: dict) -> Optional[str]:
        """Format Calendar data from overnight cache.
        Rec #4: Gracefully reports auth errors."""
        status = data.get("status", "")
        if status == "auth_error":
            return ("I don't know your schedule today — "
                    "Google calendar access expired overnight. "
                    "Please re-authenticate.")
        if status != "ok":
            return None
        d = data.get("data", {})
        events = d.get("events", [])
        if not events:
            return "No events on the calendar today"
        lines = []
        for ev in events[:3]:
            loc = f" @ {ev['location']}" if ev.get("location") else ""
            lines.append(f"  {ev.get('start', '')}  {ev.get('summary', '?')}{loc}")
        return "\n    ".join(lines)

    def _classroom_from_cache(self, data: dict) -> Optional[str]:
        """Format Classroom data from overnight cache.
        Rec #4: Gracefully reports auth errors."""
        status = data.get("status", "")
        if status == "auth_error":
            return ("I couldn't check Classroom overnight — "
                    "please re-authenticate your school account.")
        if status != "ok":
            return None
        d = data.get("data", {})
        due_today = d.get("due_today", [])
        overdue = d.get("overdue", [])
        due_tomorrow = d.get("due_tomorrow", [])
        if not due_today and not overdue and not due_tomorrow:
            return None
        parts = []
        if overdue:
            parts.append(f"⚠️ {len(overdue)} overdue")
        if due_today:
            titles = [a.get("title", "?") for a in due_today[:2]]
            parts.append(f"Due today: {', '.join(titles)}")
        if due_tomorrow:
            titles = [a.get("title", "?") for a in due_tomorrow[:2]]
            parts.append(f"Due tomorrow: {', '.join(titles)}")
        return "  ".join(parts)

    # ── Formatters ────────────────────────────────────────────────────────

    def _format(
        self, tc: dict, now: datetime,
        weather: Optional[str],
        calendar: Optional[str],
        gmail: Optional[str],
        classroom: Optional[str],
        schedule: Optional[str],
        next_class: Optional[str],
        habits: Optional[str] = None,
        reminders: Optional[str] = None,
    ) -> str:
        lines = []

        # Header — Broadcaster style
        time_str = tc["time_str"]
        date_str = now.strftime("%A, %d %B %Y")
        lines.append(f"🦌 Briefing at {time_str} — {date_str}")
        lines.append("")

        # Schedule (always show if configured)
        if schedule:
            lines.append(schedule)
        if next_class:
            lines.append(next_class)
        if schedule or next_class:
            lines.append("")

        # Weather
        if weather:
            lines.append(f"🌤  {weather}")

        # Calendar
        if calendar:
            lines.append(f"📅  {calendar}")

        # Gmail
        if gmail:
            lines.append(f"📬  {gmail}")

        # Classroom — upcoming assignments only
        if classroom:
            lines.append(f"📚  {classroom}")

        # Habit-based hint
        if habits:
            lines.append(f"🔁  {habits}")
        # Reminders due today
        if reminders:
            lines.append(f"{reminders}")
        # Footer if everything failed
        if not any([weather, calendar, gmail, classroom]):
            lines.append("(Services not responding — check your internet connection)")

        return "\n".join(lines).strip()

    # ── Data fetchers (each isolated) ─────────────────────────────────────

    async def _get_weather(self) -> Optional[str]:
        try:
            from bantz.tools.weather import WeatherTool
            w = WeatherTool()
            result = await w.execute(city="")
            if result.success:
                # Extract first line only: "📍 Samsun  🌡 8°C — Light rain"
                lines = result.output.splitlines()
                # Combine location + temp lines
                short = "  ".join(line.strip() for line in lines[:2] if line.strip())
                return short
        except Exception:
            pass
        return None

    async def _get_calendar(self, now: datetime) -> Optional[str]:
        try:
            from bantz.tools.calendar import CalendarTool
            c = CalendarTool()
            result = await c.execute(action="today")
            if result.success:
                output = result.output.strip()
                if "no events" in output.lower():
                    return "No events on the calendar today"
                # Strip "Today:" header, keep event lines
                lines = [line for line in output.splitlines() if line.strip() and "Today:" not in line]
                return "\n    ".join(lines[:3])  # max 3 events
        except Exception:
            pass
        return None

    async def _get_gmail(self) -> Optional[str]:
        try:
            from bantz.tools.gmail import GmailTool
            g = GmailTool()
            result = await g.execute(action="count")
            if result.success:
                count = result.data.get("count", 0)
                if count == 0:
                    return "Inbox is clean"
                return f"{count} unread emails"
        except Exception:
            pass
        return None

    async def _get_classroom(self) -> Optional[str]:
        try:
            from bantz.tools.classroom import ClassroomTool
            c = ClassroomTool()
            result = await c.execute(action="due_today")
            if result.success:
                output = result.output.strip()
                if "no assignments" in output.lower():
                    # Check tomorrow too
                    result2 = await c.execute(action="assignments")
                    if result2.success and "🟡 Tomorrow" in result2.output:
                        tmrw_lines = [
                            line.strip() for line in result2.output.splitlines()
                            if "Tomorrow" in line
                        ]
                        return "Due tomorrow: " + ", ".join(
                            line.replace("🟡 Tomorrow:", "").strip() for line in tmrw_lines[:2]
                        )
                    return None  # No urgent assignments → skip from briefing
                return output.replace("Due today:\n", "Due today: ")
        except Exception:
            pass
        return None

    def _get_schedule(self, now: datetime) -> Optional[str]:
        if not schedule.is_configured():
            return None
        return schedule.format_today(now)

    async def _get_next_class(self, now: datetime) -> Optional[str]:
        if not schedule.is_configured():
            return None
        text = schedule.format_next(now)
        # Append travel hint if places are configured
        try:
            from bantz.core.places import places as _places
            if _places.is_configured():
                cls = schedule.next_class(now)
                if cls and cls.get("starts_today") and cls.get("location"):
                    hint = await _places.travel_hint(
                        cls["location"], cls["starts_in_minutes"]
                    )
                    if hint:
                        text += f"\n  🚶 {hint}"
        except Exception:
            pass
        return text

    def _get_habit_hint(self, now: datetime) -> Optional[str]:
        """Show tools the user habitually uses at this time of day."""
        try:
            hour = now.hour
            if 6 <= hour < 12:
                segment = "morning"
            elif 12 <= hour < 17:
                segment = "afternoon"
            elif 17 <= hour < 21:
                segment = "evening"
            else:
                segment = "night"

            top = _habits.top_tools_for_segment(segment, n=3, days=14)
            if not top:
                return None

            names = ", ".join(t["tool"] for t in top)
            return f"Frequently used at this hour: {names}"
        except Exception:
            return None

    def _get_reminders(self) -> Optional[str]:
        """Show reminders due today — from Scheduler (#61)."""
        try:
            from bantz.core.scheduler import scheduler
            return scheduler.format_due_today()
        except Exception:
            return None


briefing = Briefing()