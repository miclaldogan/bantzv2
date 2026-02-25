"""
Bantz v2 — University Schedule
Reads weekly timetable from ~/.local/share/bantz/schedule.json.
No DB needed — static JSON, fast read.

Schedule format:
{
  "monday": [
    {
      "name": "Machine Learning",
      "time": "10:00",
      "duration": 90,
      "location": "Architecture B2",
      "type": "lab"        // optional: "lecture" | "lab" | "seminar"
    }
  ],
  "tuesday": [...],
  ...
}
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


SCHEDULE_PATH = Path.home() / ".local" / "share" / "bantz" / "schedule.json"

DAYS_EN = [
    "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday",
]

DAYS_DISPLAY = {
    "monday":    "Monday",
    "tuesday":   "Tuesday",
    "wednesday": "Wednesday",
    "thursday":  "Thursday",
    "friday":    "Friday",
    "saturday":  "Saturday",
    "sunday":    "Sunday",
}

# Keep DAYS_TR as alias for backwards compatibility with existing schedules
DAYS_TR = DAYS_DISPLAY

TYPE_EMOJI = {
    "lab":      "🔬",
    "seminar":  "📢",
    "lecture":  "📖",
    "":         "📚",
}


class Schedule:
    def __init__(self) -> None:
        self._data: dict = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if SCHEDULE_PATH.exists():
            try:
                self._data = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        self._loaded = True

    def _day_key(self, dt: datetime) -> str:
        return DAYS_EN[dt.weekday()]

    def _parse_time(self, time_str: str, ref_date: datetime) -> datetime:
        """Parse "HH:MM" into a datetime on ref_date."""
        h, m = map(int, time_str.split(":"))
        return ref_date.replace(hour=h, minute=m, second=0, microsecond=0)

    # ── Public API ────────────────────────────────────────────────────────

    def today(self, now: datetime | None = None) -> list[dict]:
        """Return today's classes, sorted by time."""
        self._load()
        now = now or datetime.now()
        key = self._day_key(now)
        classes = self._data.get(key, [])
        return sorted(classes, key=lambda c: c.get("time", ""))

    def next_class(self, now: datetime | None = None) -> Optional[dict]:
        """
        Return the next upcoming class (today or coming days).
        Adds 'starts_in_minutes' and 'starts_today' to result.
        """
        self._load()
        now = now or datetime.now()

        # Look ahead up to 7 days
        for day_offset in range(7):
            check_dt = now + timedelta(days=day_offset)
            key = self._day_key(check_dt)
            classes = sorted(
                self._data.get(key, []),
                key=lambda c: c.get("time", ""),
            )
            for cls in classes:
                class_dt = self._parse_time(cls["time"], check_dt)
                # Skip classes that already ended (add duration)
                duration = cls.get("duration", 60)
                class_end = class_dt + timedelta(minutes=duration)
                if class_end > now:
                    delta_minutes = int((class_dt - now).total_seconds() / 60)
                    return {
                        **cls,
                        "starts_in_minutes": delta_minutes,
                        "starts_today": day_offset == 0,
                        "day_name": DAYS_DISPLAY[key],
                    }
        return None

    def format_today(self, now: datetime | None = None) -> str:
        """Return formatted string of today's classes."""
        now = now or datetime.now()
        classes = self.today(now)
        if not classes:
            day_name = DAYS_DISPLAY[self._day_key(now)]
            return f"No classes on {day_name}."

        lines = []
        for cls in classes:
            emoji = TYPE_EMOJI.get(cls.get("type", ""), "📚")
            name = cls.get("name", "")
            time = cls.get("time", "")
            duration = cls.get("duration", 60)
            location = cls.get("location", "")
            instructor = cls.get("instructor", "")
            end_h = int(time.split(":")[0]) * 60 + int(time.split(":")[1]) + duration
            end_str = f"{end_h // 60:02d}:{end_h % 60:02d}"
            loc_str = f"  📍 {location}" if location else ""
            instr_str = f"  ({instructor})" if instructor else ""
            lines.append(f"  {emoji} {time}–{end_str}  {name}{instr_str}{loc_str}")

        day_name = DAYS_DISPLAY[self._day_key(now)]
        return f"{day_name} classes:\n" + "\n".join(lines)

    def format_for_date(self, target: datetime) -> str:
        """Return formatted string of classes for an arbitrary date."""
        return self.format_today(target)

    def format_week(self, start: datetime | None = None) -> str:
        """Return formatted string for an entire week starting from Monday."""
        self._load()
        start = start or datetime.now()
        monday = start - timedelta(days=start.weekday())

        sections: list[str] = []
        total = 0
        for i in range(7):
            day_dt = monday + timedelta(days=i)
            key = self._day_key(day_dt)
            classes = sorted(
                self._data.get(key, []),
                key=lambda c: c.get("time", ""),
            )
            if not classes:
                continue
            total += len(classes)
            day_name = DAYS_DISPLAY[key]
            lines = []
            for cls in classes:
                emoji = TYPE_EMOJI.get(cls.get("type", ""), "📚")
                name = cls.get("name", "")
                time = cls.get("time", "")
                lines.append(f"    {emoji} {time}  {name}")
            sections.append(f"  {day_name}:\n" + "\n".join(lines))

        if not sections:
            return "No classes this week."
        header = f"Weekly schedule ({total} class(es)):"
        return header + "\n" + "\n".join(sections)

    def format_next(self, now: datetime | None = None) -> str:
        """Return formatted string of next upcoming class."""
        now = now or datetime.now()
        cls = self.next_class(now)
        if not cls:
            return "No classes this week."

        emoji = TYPE_EMOJI.get(cls.get("type", ""), "📚")
        name = cls.get("name", "")
        time = cls.get("time", "")
        location = cls.get("location", "")
        instructor = cls.get("instructor", "")
        loc_str = f"  📍 {location}" if location else ""
        instr_str = f" ({instructor})" if instructor else ""
        mins = cls["starts_in_minutes"]

        if cls["starts_today"]:
            if mins <= 0:
                when = "in progress now"
            elif mins < 60:
                when = f"in {mins} min"
            else:
                h, m = divmod(mins, 60)
                when = f"in {h}h {m}m" if m else f"in {h}h"
        else:
            when = f"{cls['day_name']} {time}"

        return f"Next class: {emoji} {name}{instr_str}  {time}{loc_str}\n  ⏰ {when}"

    def is_configured(self) -> bool:
        return SCHEDULE_PATH.exists()

    @staticmethod
    def setup_path() -> Path:
        return SCHEDULE_PATH


# Singleton
schedule = Schedule()
