"""
Bantz v2 — Time Context
Pure Python time-of-day awareness. No LLM, no API.
Gives Bantz context about when it is so responses feel natural.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

Segment = Literal["late_night", "morning", "afternoon", "evening", "night"]


def get_segment(hour: int | None = None) -> Segment:
    """Return the time-of-day segment for a given hour (0-23)."""
    if hour is None:
        hour = datetime.now().hour
    if 0 <= hour < 6:
        return "late_night"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


_GREETINGS: dict[Segment, str] = {
    "late_night": "Working late tonight",
    "morning":    "Good morning",
    "afternoon":  "Good afternoon",
    "evening":    "Good evening",
    "night":      "Good night",
}

_SEGMENT_EN: dict[Segment, str] = {
    "late_night": "late night",
    "morning":    "morning",
    "afternoon":  "afternoon",
    "evening":    "evening",
    "night":      "night",
}


class TimeContext:
    """
    Usage:
        from bantz.core.time_context import time_ctx
        ctx = time_ctx.snapshot()
        print(ctx["greeting"])       # "Good morning"
        print(ctx["prompt_hint"])    # injected into LLM prompts
    """

    def snapshot(self) -> dict:
        """Return a dict with all time context fields."""
        now = datetime.now()
        seg = get_segment(now.hour)

        return {
            "hour":         now.hour,
            "minute":       now.minute,
            "segment":      seg,
            "segment_en":   _SEGMENT_EN[seg],
            "greeting":     _GREETINGS[seg],
            "time_str":     now.strftime("%H:%M"),
            "date_str":     now.strftime("%A, %d %B %Y"),
            "prompt_hint":  self._prompt_hint(seg, now),
        }

    def _prompt_hint(self, seg: Segment, now: datetime) -> str:
        """Short string injected into LLM system prompts."""
        return (
            f"Current time: {now.strftime('%H:%M')} ({_SEGMENT_EN[seg]}), "
            f"{now.strftime('%A %d %B %Y')}."
        )

    def greeting_line(self) -> str:
        """Ready-to-use greeting for startup or 'hello' responses."""
        now = datetime.now()
        seg = get_segment(now.hour)
        greeting = _GREETINGS[seg]
        return f"{greeting}! It's {now.strftime('%H:%M')}."


# Singleton
time_ctx = TimeContext()
