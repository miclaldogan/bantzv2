"""
Bantz v2 — Broadcaster Greeting

Context-aware theatrical greeting on app launch.
Bantz is "The Broadcaster" — a 1930s radio showman who treats the terminal as his studio.
Combines absence awareness, time-of-day, and live service summaries.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional

from bantz.core.time_context import get_segment
from bantz.core.profile import profile


class Butler:

    async def greet(self, session_info: dict[str, Any]) -> str:
        """
        Compose a butler-style greeting:
        1. Time-of-day + absence-aware opener
        2. Live data summaries (mail, calendar, classroom)
        3. Return 2-4 natural sentences
        """
        now = datetime.now()
        seg = get_segment(now.hour)

        opener = self._opener(
            absence_label=session_info["absence_label"],
            absence_hours=session_info["absence_hours"],
            is_first=session_info.get("is_first", False),
            segment=seg,
        )

        # Fire live data fetchers in parallel (each isolated)
        summaries = await asyncio.gather(
            self._mail_summary(),
            self._calendar_summary(now),
            self._classroom_summary(),
            return_exceptions=True,
        )
        mail_str, cal_str, class_str = [
            r if isinstance(r, str) else None for r in summaries
        ]

        return self._format(opener, mail_str, cal_str, class_str)

    # ── Opener generation ─────────────────────────────────────────────────────

    def _opener(
        self,
        absence_label: str,
        absence_hours: float,
        is_first: bool,
        segment: str,
    ) -> str:
        if is_first:
            return (
                "Signal strong, coffee bitter, and I'm here. "
                "Greetings, old friend — broadcast begins."
            )

        if absence_hours < 1:
            return "Oh, you again! Haven't even left the stage."

        if absence_hours < 6:
            return "Welcome back, friend. Your signal comes in clear on this frequency."

        if absence_hours < 20:
            return "Old friend, you're back! The studio was quiet without you."

        if absence_hours < 30:
            if segment == "morning":
                return (
                    "Good morning, friend! Hope the dreams were good — "
                    "reality's here waiting."
                )
            return "You drew the curtain last night, but the show goes on."

        if absence_hours < 72:
            return (
                "Old friend! The studio's been dark for a few days. "
                "But the signal never cut out."
            )

        if absence_hours < 168:
            return (
                "Friend, you've been off the air for a week! "
                "The frequencies were looking for you."
            )

        return (
            "Dear listener... it's been quiet on this channel for a long time. "
            "But Bantz is always here, always on air."
        )

    @staticmethod
    def _time_greeting(segment: str) -> str:
        return {
            "late_night": "Good night",
            "morning":    "Good morning",
            "afternoon":  "Good afternoon",
            "evening":    "Good evening",
            "night":      "Good night",
        }.get(segment, "Hello")

    # ── Formatters ────────────────────────────────────────────────────────────

    def _format(
        self,
        opener: str,
        mail: Optional[str],
        calendar: Optional[str],
        classroom: Optional[str],
    ) -> str:
        parts = [opener]

        details: list[str] = []
        if calendar:
            details.append(calendar)
        if mail:
            details.append(mail)
        if classroom:
            details.append(classroom)

        if details:
            parts.append(" ".join(details))

        return " ".join(parts)

    # ── Live data fetchers (isolated, never crash) ────────────────────────────

    async def _mail_summary(self) -> Optional[str]:
        """Count unread messages if Gmail is configured."""
        try:
            from bantz.auth.token_store import token_store
            status = token_store.status()
            if status.get("gmail") != "ok":
                return None

            from bantz.tools.gmail import GmailTool
            g = GmailTool()
            result = await g.execute(action="filter", q="is:unread", max_results=10)
            if not result.success:
                return None
            output = result.output.strip()
            if not output or "no results" in output.lower() or "not found" in output.lower():
                return None
            lines = [l for l in output.splitlines() if l.strip()]
            count = len(lines)
            if count == 0:
                return None
            if count == 1:
                return "1 unread message waiting in your inbox."
            return f"{count} unread messages in your inbox."
        except Exception:
            return None

    async def _calendar_summary(self, now: datetime) -> Optional[str]:
        """Get next calendar event if configured."""
        try:
            from bantz.auth.token_store import token_store
            status = token_store.status()
            if status.get("calendar") != "ok":
                return None

            from bantz.tools.calendar import CalendarTool
            c = CalendarTool()
            result = await c.execute(action="today")
            if not result.success:
                return None
            output = result.output.strip()
            if not output or "no events" in output.lower() or "empty" in output.lower():
                return None
            import re
            for line in output.splitlines():
                stripped = line.strip()
                if stripped and not stripped.endswith(":") and len(stripped) > 5:
                    cleaned = re.sub(r"\s*\[id:[^\]]+\]", "", stripped).strip()
                    if cleaned:
                        return f"On today's agenda: {cleaned}."
            return None
        except Exception:
            return None

    async def _classroom_summary(self) -> Optional[str]:
        """Get upcoming assignment deadlines if configured."""
        try:
            from bantz.auth.token_store import token_store
            status = token_store.status()
            if status.get("classroom") != "ok":
                return None

            from bantz.tools.classroom import ClassroomTool
            c = ClassroomTool()
            result = await c.execute(action="upcoming")
            if not result.success:
                return None
            output = result.output.strip()
            if not output or "no assignments" in output.lower() or "not found" in output.lower():
                return None
            lines = [l for l in output.splitlines() if l.strip()]
            count = len(lines)
            if count == 0:
                return None
            return f"{count} assignment(s) due soon."
        except Exception:
            return None


butler = Butler()
