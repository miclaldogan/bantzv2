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


class Briefing:

    async def generate(self) -> str:
        now = datetime.now()
        tc = time_ctx.snapshot()

        # Fire all external calls in parallel
        results = await asyncio.gather(
            self._get_weather(),
            self._get_calendar(now),
            self._get_gmail(),
            self._get_classroom(),
            return_exceptions=True,
        )
        weather_str, calendar_str, gmail_str, classroom_str = [
            r if isinstance(r, str) else None for r in results
        ]

        # Schedule is local — no network, no failure
        schedule_str = self._get_schedule(now)
        next_class_str = self._get_next_class(now)

        return self._format(
            tc=tc,
            now=now,
            weather=weather_str,
            calendar=calendar_str,
            gmail=gmail_str,
            classroom=classroom_str,
            schedule=schedule_str,
            next_class=next_class_str,
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
    ) -> str:
        lines = []

        # Header
        greeting = tc["greeting"]
        if _profile.is_configured():
            greeting = f"{greeting} {_profile.get('name')}!"
        else:
            greeting = f"{greeting}!"
        time_str = tc["time_str"]
        date_str = now.strftime("%A, %d %B %Y")
        lines.append(f"{greeting} 🕐 {time_str}  {date_str}")
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

        # Footer if everything failed
        if not any([weather, calendar, gmail, classroom]):
            lines.append("(Servisler yanıt vermedi — internet bağlantısını kontrol et)")

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
                if "etkinlik yok" in output.lower():
                    return "Bugün takvimde etkinlik yok"
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
                    return "Gelen kutu temiz"
                return f"{count} okunmamış mail"
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
                if "ödev yok" in output.lower():
                    # Check tomorrow too
                    result2 = await c.execute(action="assignments")
                    if result2.success and "🟡 Yarın" in result2.output:
                        yarın_lines = [
                            l.strip() for l in result2.output.splitlines()
                            if "Yarın" in l
                        ]
                        return "Yarın teslim: " + ", ".join(
                            l.replace("🟡 Yarın:", "").strip() for l in yarın_lines[:2]
                        )
                    return None  # No urgent assignments → skip from briefing
                return output.replace("Bugün teslim:\n", "Bugün teslim: ")
        except Exception:
            pass
        return None

    def _get_schedule(self, now: datetime) -> Optional[str]:
        if not schedule.is_configured():
            return None
        return schedule.format_today(now)

    def _get_next_class(self, now: datetime) -> Optional[str]:
        if not schedule.is_configured():
            return None
        return schedule.format_next(now)


briefing = Briefing()