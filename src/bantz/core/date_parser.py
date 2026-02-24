"""
Bantz v3 — Date Parser
Resolves relative and named dates from English natural language.

Usage:
    from bantz.core.date_parser import resolve_date
    dt = resolve_date("tomorrow classes")       # → tomorrow
    dt = resolve_date("thursday meeting")       # → next Thursday
    dt = resolve_date("yesterday summary")      # → yesterday
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional


# English weekday names → Python weekday index (Monday=0 .. Sunday=6)
_EN_WEEKDAYS: dict[str, int] = {
    "monday":    0,
    "tuesday":   1,
    "wednesday": 2,
    "thursday":  3,
    "friday":    4,
    "saturday":  5,
    "sunday":    6,
    "mon":       0,
    "tue":       1,
    "wed":       2,
    "thu":       3,
    "fri":       4,
    "sat":       5,
    "sun":       6,
}

# Relative date tokens
_RELATIVE: list[tuple[str, int]] = [
    ("day after tomorrow", 2),
    ("day before yesterday", -2),
    ("tomorrow",  1),
    ("yesterday", -1),
    ("today",     0),
]

# Week-based references
_WEEK_REFS: list[tuple[str, str]] = [
    ("next week", "next"),
    ("last week", "last"),
    ("this week", "this"),
]


def resolve_date(text: str, now: datetime | None = None) -> Optional[datetime]:
    """
    Extract and resolve a date reference from English text.
    Returns a datetime (date at 00:00) or None if no date found.

    Priority: explicit relative ("tomorrow") > named weekday ("thursday") > week ref.
    """
    now = now or datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    t = text.lower().strip()

    # 1. Relative dates (ordered longest-first to avoid partial matches)
    for token, delta_days in _RELATIVE:
        if token in t:
            return today + timedelta(days=delta_days)

    # 2. Named weekdays — resolve to next occurrence (today if today is that day)
    for day_name, weekday_idx in _EN_WEEKDAYS.items():
        if re.search(rf"\b{day_name}\b", t):
            return _next_weekday(today, weekday_idx)

    # 3. Week-based references
    for token, direction in _WEEK_REFS:
        if token in t:
            return _week_start(today, direction)

    # 4. ISO date in text (2025-01-15)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", t)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            pass

    return None


def _next_weekday(today: datetime, target_weekday: int) -> datetime:
    """
    Return the next occurrence of target_weekday from today.
    If today IS that weekday, return today.
    """
    current = today.weekday()
    delta = (target_weekday - current) % 7
    return today + timedelta(days=delta)


def _week_start(today: datetime, direction: str) -> datetime:
    """
    Return the Monday of 'this', 'next', or 'last' week.
    """
    monday = today - timedelta(days=today.weekday())
    if direction == "next":
        return monday + timedelta(weeks=1)
    elif direction == "last":
        return monday - timedelta(weeks=1)
    return monday  # "this"
