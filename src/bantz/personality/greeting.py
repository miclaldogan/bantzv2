"""
Bantz v3 — Boot Greeting + Morning Briefing

Called on TUI startup. Absence-aware, time-aware.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from bantz.core.time_context import get_segment


class Greeting:
    """Generates context-aware boot greeting."""

    async def boot(self, session_info: dict) -> str:
        """Full boot greeting with live data summaries."""
        now = datetime.now()
        seg = get_segment(now.hour)
        opener = self._opener(
            absence_hours=session_info.get("absence_hours", 0),
            is_first=session_info.get("is_first", False),
            segment=seg,
        )
        # Parallel live data fetch
        summaries = await asyncio.gather(
            self._mail_summary(),
            self._calendar_summary(now),
            return_exceptions=True,
        )
        mail_str, cal_str = [r if isinstance(r, str) else None for r in summaries]

        parts = [opener]
        if cal_str:
            parts.append(cal_str)
        if mail_str:
            parts.append(mail_str)
        return " ".join(parts)

    def _opener(self, absence_hours: float, is_first: bool, segment: str) -> str:
        if is_first:
            return (
                "Signal strong, coffee bitter, and I'm here. "
                "Greetings, old friend — broadcast begins."
            )
        if absence_hours < 1:
            return "Oh, you again! Haven't even left the stage."
        if absence_hours < 6:
            return "Welcome back, friend. Signal coming in clear."
        if absence_hours < 20:
            return "Old friend, you're back! Studio was quiet without you."
        if absence_hours < 30:
            if segment == "morning":
                return "Good morning, friend! Hope the dreams were good — reality's here waiting."
            return "You drew the curtain last night, but the show goes on."
        if absence_hours < 72:
            return "Old friend! Studio's been dark a few days. Signal never cut out."
        if absence_hours < 168:
            return "Friend, you've been off the air for a week! Frequencies were looking for you."
        return "Dear listener... it's been quiet on this channel. But Bantz is always here, always on air."

    async def _mail_summary(self) -> Optional[str]:
        try:
            from bantz.auth.token_store import token_store
            if token_store.status().get("gmail") != "ok":
                return None
            from bantz.tools.gmail import GmailTool
            result = await GmailTool().execute(action="count")
            if result.success:
                count = result.data.get("count", 0)
                if count > 0:
                    return f"{count} unread mail(s) in your inbox."
        except Exception:
            pass
        return None

    async def _calendar_summary(self, now: datetime) -> Optional[str]:
        try:
            from bantz.auth.token_store import token_store
            if token_store.status().get("calendar") != "ok":
                return None
            from bantz.tools.calendar import CalendarTool
            result = await CalendarTool().execute(action="today")
            if result.success:
                output = result.output.strip()
                if "no events" not in output.lower() and output:
                    import re
                    for line in output.splitlines():
                        stripped = line.strip()
                        if stripped and not stripped.endswith(":") and len(stripped) > 5:
                            cleaned = re.sub(r"\s*\[id:[^\]]+\]", "", stripped).strip()
                            if cleaned:
                                return f"On today's agenda: {cleaned}."
        except Exception:
            pass
        return None


greeting = Greeting()
