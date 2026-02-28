"""
Bantz v2 — Daily Briefing
Parallel API calls: calendar + classroom + gmail + weather + schedule.
Any service failure is isolated — others still show.

Usage:
    from bantz.core.briefing import briefing
    text = await briefing.generate()
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from bantz.core.schedule import schedule
from bantz.core.time_context import time_ctx
from bantz.core.profile import profile as _profile
from bantz.core.habits import habits as _habits


class Briefing:

    async def generate(self) -> str:
        now = datetime.now()
        tc = time_ctx.snapshot()
        sections = _profile.briefing_sections  # user-selected sections

        # Fire all external calls in parallel (skip disabled sections)
        weather_coro = self._get_weather() if "weather" in sections else asyncio.sleep(0)
        calendar_coro = self._get_calendar(now) if "calendar" in sections else asyncio.sleep(0)
        gmail_coro = self._get_gmail() if "mail" in sections else asyncio.sleep(0)
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
        )

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
                short = "  ".join(l.strip() for l in lines[:2] if l.strip())
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
                # Strip "Bugün:" header, keep event lines
                lines = [l for l in output.splitlines() if l.strip() and "Bugün:" not in l]
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
                if "ödev yok" in output.lower() or "no assignments" in output.lower():
                    # Check tomorrow too
                    result2 = await c.execute(action="assignments")
                    if result2.success and ("🟡 Tomorrow" in result2.output or "🟡 Yarın" in result2.output):
                        yarın_lines = [
                            l.strip() for l in result2.output.splitlines()
                            if "Tomorrow" in l or "Yarın" in l
                        ]
                        return "Due tomorrow: " + ", ".join(
                            l.replace("🟡 Tomorrow:", "").replace("🟡 Yarın:", "").strip() for l in yarın_lines[:2]
                        )
                    return None  # No urgent assignments → skip from briefing
                return output.replace("Due today:\n", "Due today: ").replace("Bugün teslim:\n", "Due today: ")
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
                segment = "sabah"
            elif 12 <= hour < 17:
                segment = "oglen"
            elif 17 <= hour < 21:
                segment = "aksam"
            else:
                segment = "gece_gec"

            top = _habits.top_tools_for_segment(segment, n=3, days=14)
            if not top:
                return None

            names = ", ".join(t["tool"] for t in top)
            return f"Frequently used at this hour: {names}"
        except Exception:
            return None


briefing = Briefing()