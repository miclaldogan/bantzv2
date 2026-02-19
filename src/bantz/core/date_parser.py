"""
Bantz v2 — Turkish Date Parser
Resolves relative and named dates from Turkish natural language.

Usage:
    from bantz.core.date_parser import resolve_date
    dt = resolve_date("yarın derslerim")       # → tomorrow
    dt = resolve_date("perşembe toplantım var") # → next Thursday
    dt = resolve_date("dün ne yaptık")          # → yesterday
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional


# Turkish weekday names → Python weekday index (Monday=0 .. Sunday=6)
_TR_WEEKDAYS: dict[str, int] = {
    "pazartesi": 0,
    "salı":      1,
    "salı":      1,
    "çarşamba":  2,
    "perşembe":  3,
    "cuma":      4,
    "cumartesi": 5,
    "pazar":     6,
}

# Relative date tokens
_RELATIVE: list[tuple[str, int]] = [
    ("öbür gün",    2),
    ("önceki gün", -2),
    ("evvelsi gün",-2),
    ("yarın",       1),
    ("dün",        -1),
    ("bugün",       0),
]

# Week-based references
_WEEK_REFS: list[tuple[str, str]] = [
    ("gelecek hafta", "next"),
    ("haftaya",       "next"),
    ("geçen hafta",   "last"),
    ("bu hafta",      "this"),
]


def resolve_date(text: str, now: datetime | None = None) -> Optional[datetime]:
    """
    Extract and resolve a date reference from Turkish text.
    Returns a datetime (date at 00:00) or None if no date found.

    Priority: explicit relative ("yarın") > named weekday ("perşembe") > week ref.
    """
    now = now or datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    t = text.lower().strip()

    # 1. Relative dates (ordered longest-first to avoid partial matches)
    for token, delta_days in _RELATIVE:
        if token in t:
            return today + timedelta(days=delta_days)

    # 2. Named weekdays — resolve to next occurrence (today if today is that day)
    for day_name, weekday_idx in _TR_WEEKDAYS.items():
        if day_name in t:
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
