"""
Bantz v2 — Butler Greeting

Context-aware proactive greeting on app launch.
Bantz is an Operations Director — direct, specific, data-driven.
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
        Compose a proactive greeting:
        1. Time-of-day + absence-aware opener
        2. Live data summaries (mail count, next event, schedule)
        3. Return 2-4 natural sentences — direct and specific
        """
        now = datetime.now()
        seg = get_segment(now.hour)

        opener = self._opener(
            absence_label=session_info["absence_label"],
            absence_hours=session_info["absence_hours"],
            is_first=session_info.get("is_first", False),
            segment=seg,
        )

        # Fire live data fetchers in parallel (each with individual timeout)
        async def _with_timeout(coro, secs=5):
            try:
                return await asyncio.wait_for(coro, timeout=secs)
            except (asyncio.TimeoutError, Exception):
                return None

        summaries = await asyncio.gather(
            _with_timeout(self._mail_summary()),
            _with_timeout(self._calendar_summary(now)),
            _with_timeout(self._classroom_summary()),
            _with_timeout(self._schedule_summary(now)),
            return_exceptions=True,
        )
        mail_str, cal_str, class_str, sched_str = [
            r if isinstance(r, str) else None for r in summaries
        ]

        return self._format(opener, mail_str, cal_str, class_str, sched_str)

    # ── Opener generation ─────────────────────────────────────────────────

    def _opener(
        self,
        absence_label: str,
        absence_hours: float,
        is_first: bool,
        segment: str,
    ) -> str:
        greeting = self._time_greeting(segment)

        if is_first:
            return f"{greeting}, ma'am. Bantz here — ready to go."

        if absence_hours < 1:
            return f"Welcome back, ma'am."

        if absence_hours < 6:
            return f"{greeting}, ma'am. Been a few hours."

        if absence_hours < 20:
            return f"{greeting}, boss. Here's what's going on."

        if absence_hours < 30:
            if segment == "sabah":
                return "Good morning, ma'am. Let me catch you up."
            return f"{greeting}, ma'am. Here's your status."

        if absence_hours < 72:
            return f"{greeting}, boss. It's been a couple days — here's the rundown."

        if absence_hours < 168:
            return f"{greeting}, ma'am. Been about a week. Let me brief you."

        return f"{greeting}, boss. Long time no see. Here's what you've missed."

    @staticmethod
    def _time_greeting(segment: str) -> str:
        return {
            "gece_erken": "Good evening",
            "sabah": "Good morning",
            "oglen": "Good afternoon",
            "aksam": "Good evening",
            "gece_gec": "Good evening",
        }.get(segment, "Hello")

    # ── Formatters ────────────────────────────────────────────────────────

    def _format(
        self,
        opener: str,
        mail: Optional[str],
        calendar: Optional[str],
        classroom: Optional[str],
        schedule: Optional[str],
    ) -> str:
        parts = [opener]

        details: list[str] = []
        if schedule:
            details.append(schedule)
        if calendar:
            details.append(calendar)
        if mail:
            details.append(mail)
        if classroom:
            details.append(classroom)

        if details:
            parts.append(" ".join(details))
        elif not any([mail, calendar, classroom, schedule]):
            parts.append("All clear — nothing urgent right now.")

        return " ".join(parts)

    # ── Live data fetchers (isolated, never crash) ────────────────────────

    async def _mail_summary(self) -> Optional[str]:
        """Count unread messages if Gmail is configured."""
        try:
            from bantz.auth.token_store import token_store
            status = token_store.status()
            if status.get("gmail") != "ok":
                return None

            from bantz.tools.gmail import GmailTool
            g = GmailTool()
            result = await g.execute(action="count")
            if not result.success:
                return None
            count = result.data.get("count", 0)
            if count == 0:
                return "Inbox is clean."
            if count == 1:
                return "You have 1 unread email."
            return f"You have {count} unread emails."
        except Exception:
            return None

    async def _calendar_summary(self, now: datetime) -> Optional[str]:
        """Get today's calendar events if configured."""
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
            data = result.data or {}
            count = data.get("count", 0)
            if count == 0:
                return "Calendar is clear today."
            events = data.get("events", [])
            if events:
                first = events[0]
                name = first.get("summary", "event")
                time = first.get("start_local", "")
                if count == 1:
                    return f"You have 1 event today: {name} at {time}."
                return f"You have {count} events today — first one: {name} at {time}."
            return f"You have {count} events today."
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
            result = await c.execute(action="due_today")
            if not result.success:
                return None
            output = result.output.strip()
            if not output or "no" in output.lower() or output.strip() == "":
                return None
            lines = [l for l in output.splitlines() if l.strip()]
            count = len(lines)
            if count == 0:
                return None
            if count == 1:
                return "1 assignment due."
            return f"{count} assignments due."
        except Exception:
            return None

    async def _schedule_summary(self, now: datetime) -> Optional[str]:
        """Get today's class schedule."""
        try:
            from bantz.core.schedule import schedule as _sched
            if not _sched.is_configured():
                return None
            classes = _sched.today(now)
            if not classes:
                return "No classes today."
            count = len(classes)
            first = classes[0]
            name = first.get("name", "class")
            time = first.get("time", "")
            if count == 1:
                return f"You have 1 class today: {name} at {time}."
            return f"You have {count} classes today — first: {name} at {time}."
        except Exception:
            return None


butler = Butler()
