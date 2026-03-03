"""
Bantz v3 — Time Parser
Resolves natural-language time expressions to HH:MM format.

Handles:
  "5pm"         → "17:00"
  "5:30 PM"     → "17:30"
  "3 pm"        → "15:00"
  "at 14"       → "14:00"
  "at 9"        → "09:00"
  "noon"        → "12:00"
  "midnight"    → "00:00"
  "morning"     → "09:00"
  "afternoon"   → "14:00"
  "evening"     → "19:00"
  "15:00"       → "15:00"  (passthrough)

Usage:
    from bantz.core.time_parser import resolve_time
    t = resolve_time("meeting at 5pm")       # → "17:00"
    t = resolve_time("lunch at noon")        # → "12:00"
    t = resolve_time("15:30")                # → "15:30"
"""
from __future__ import annotations

import re
from typing import Optional


_NAMED_TIMES = {
    "noon": "12:00",
    "midday": "12:00",
    "midnight": "00:00",
    "morning": "09:00",
    "afternoon": "14:00",
    "evening": "19:00",
    "night": "21:00",
}

# Pattern: "5pm", "5:30pm", "5 pm", "5:30 pm", "5:30PM", etc.
_AMPM_RE = re.compile(
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)\b",
)

# Pattern: "at 14", "at 9", "at 5" — bare hour after "at"
_AT_HOUR_RE = re.compile(
    r"\bat\s+(\d{1,2})(?::(\d{2}))?\b(?!\s*(?:am|pm))",
    re.IGNORECASE,
)

# Pattern: already in HH:MM format
_HHMM_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def resolve_time(text: str) -> Optional[str]:
    """
    Extract and resolve a time reference from natural language text.
    Returns "HH:MM" string or None if no time found.
    """
    t = text.lower().strip()

    # 1. Named times
    for name, hhmm in _NAMED_TIMES.items():
        if name in t:
            return hhmm

    # 2. AM/PM format — "5pm", "5:30 PM", "11am"
    m = _AMPM_RE.search(text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        period = m.group(3).lower()
        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    # 3. "at N" — bare hour
    m = _AT_HOUR_RE.search(text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        # Heuristic: if hour < 7 and no AM/PM, assume PM (e.g. "at 5" → 17:00)
        if hour < 7:
            hour += 12
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    # 4. Already HH:MM
    m = _HHMM_RE.search(text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    return None


def normalize_time(time_str: str) -> str:
    """
    Normalize a time string to HH:MM. Accepts:
      "5pm", "5:30PM", "17:00", "9", "noon", etc.
    Falls back to the input if unparseable.
    """
    resolved = resolve_time(time_str)
    if resolved:
        return resolved

    # Try bare number
    time_str = time_str.strip()
    if time_str.isdigit():
        h = int(time_str)
        if 0 <= h <= 23:
            return f"{h:02d}:00"

    # Already well-formed?
    if re.match(r"^\d{1,2}:\d{2}$", time_str):
        return time_str

    return time_str
