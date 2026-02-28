"""
Bantz v2 — User Profile

Stores user identity and preferences in ~/.local/share/bantz/profile.json.
Used by brain.py for personalized prompts, briefing.py for section filtering,
time_context.py / butler.py for named greetings.

Schema:
    {
        "name": "Iclal",
        "university": "OMU",
        "department": "Computer Engineering",
        "year": 2,
        "pronoun": "casual",
        "tone": "casual",
        "preferences": {
            "briefing_sections": ["schedule", "weather", "mail", "calendar", "classroom"],
            "news_sources": ["hn", "google"],
            "response_style": "casual"
        }
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_PROFILE_PATH = Path.home() / ".local" / "share" / "bantz" / "profile.json"

_DEFAULT_PREFERENCES: dict[str, Any] = {
    "briefing_sections": ["schedule", "weather", "mail", "calendar", "classroom"],
    "news_sources": ["hn", "google"],
    "response_style": "casual",
}

_DEFAULTS: dict[str, Any] = {
    "name": "",
    "university": "",
    "department": "",
    "year": 0,
    "pronoun": "casual",
    "tone": "casual",
    "preferences": dict(_DEFAULT_PREFERENCES),
}

ALL_BRIEFING_SECTIONS = ["schedule", "weather", "mail", "calendar", "classroom", "habits"]
ALL_NEWS_SOURCES = ["hn", "google"]


class Profile:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if _PROFILE_PATH.exists():
            try:
                self._data = json.loads(_PROFILE_PATH.read_text("utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def save(self, data: dict[str, Any]) -> None:
        _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged = {**_DEFAULTS, **data}
        # Merge preferences sub-dict properly
        merged["preferences"] = {
            **_DEFAULT_PREFERENCES,
            **data.get("preferences", {}),
        }
        self._data = merged
        _PROFILE_PATH.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reload(self) -> None:
        self._load()

    # ── Query ─────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = "") -> Any:
        return self._data.get(key, default)

    def is_configured(self) -> bool:
        return bool(self._data.get("name"))

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def preferences(self) -> dict[str, Any]:
        return self._data.get("preferences", dict(_DEFAULT_PREFERENCES))

    @property
    def response_style(self) -> str:
        return self.preferences.get("response_style", "casual")

    @property
    def briefing_sections(self) -> list[str]:
        return self.preferences.get("briefing_sections", ALL_BRIEFING_SECTIONS)

    @property
    def news_sources(self) -> list[str]:
        return self.preferences.get("news_sources", ALL_NEWS_SOURCES)

    @property
    def path(self) -> Path:
        return _PROFILE_PATH

    # ── Prompt helpers ────────────────────────────────────────────────────

    def prompt_hint(self) -> str:
        """One-line hint injected into brain system prompts."""
        if not self.is_configured():
            return ""
        name = self._data["name"]
        style = self.response_style
        # Preferred address: explicit field > pronoun-based > default
        address = self._data.get("preferred_address", "")
        pronoun = self._data.get("pronoun", "casual")
        if not address:
            if pronoun in ("siz", "formal", "ma'am", "madam"):
                address = "ma'am"
            else:
                address = "boss"
        parts: list[str] = [
            f"User's name is {name}. Address them as '{address}'."
        ]
        uni = self._data.get("university")
        dept = self._data.get("department")
        year = self._data.get("year")
        if uni:
            edu = uni
            if dept:
                edu += f" — {dept}"
            if year:
                edu += f", year {year}"
            parts.append(f"Studies at {edu}.")
        tone = self._data.get("tone", "casual")
        if style == "formal" or tone == "formal":
            parts.append("Use a professional, respectful tone.")
        else:
            parts.append("Keep it casual and friendly — like an old friend.")
        return " ".join(parts)

    def status_line(self) -> str:
        """Short status for --doctor output."""
        if not self.is_configured():
            return "not configured  → bantz --setup profile"
        name = self._data.get("name", "?")
        style = self.response_style
        return f"{name} ({style})"


profile = Profile()
