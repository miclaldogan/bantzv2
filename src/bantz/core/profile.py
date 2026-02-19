"""
Bantz v2 — User Profile

Stores user preferences in ~/.local/share/bantz/profile.json.
Used by brain.py for personalized system prompts and briefing.py for greetings.

Schema:
    {
        "name": "Ali",
        "university": "OMÜ",
        "department": "Bilgisayar Mühendisliği",
        "year": 2,
        "pronoun": "sen",       # sen | siz
        "tone": "samimi"        # samimi | resmi
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_PROFILE_PATH = Path.home() / ".local" / "share" / "bantz" / "profile.json"

_DEFAULTS: dict[str, Any] = {
    "name": "",
    "university": "",
    "department": "",
    "year": 0,
    "pronoun": "sen",
    "tone": "samimi",
}


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
        self._data = {**_DEFAULTS, **data}
        _PROFILE_PATH.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Query ─────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = "") -> Any:
        return self._data.get(key, default)

    def is_configured(self) -> bool:
        return bool(self._data.get("name"))

    @property
    def path(self) -> Path:
        return _PROFILE_PATH

    # ── Prompt helpers ────────────────────────────────────────────────────

    def prompt_hint(self) -> str:
        """One-line hint injected into brain system prompts."""
        if not self.is_configured():
            return ""
        parts: list[str] = [f"Kullanıcının adı {self._data['name']} ama ismini kullanma — 'dostum', 'eski arkadaşım' de."]
        uni = self._data.get("university")
        dept = self._data.get("department")
        year = self._data.get("year")
        if uni:
            edu = uni
            if dept:
                edu += f" {dept}"
            if year:
                edu += f" {year}. sınıf"
            parts.append(f"Eğitim: {edu}.")
        parts.append("Kullanıcıya sen diye hitap et, eski arkadaş gibi samimi ol.")
        return " ".join(parts)

    def status_line(self) -> str:
        """Short status for --doctor output."""
        if not self.is_configured():
            return "yapılandırılmamış  → bantz --setup profile"
        name = self._data.get("name", "?")
        tone = self._data.get("tone", "samimi")
        return f"{name} ({tone})"


profile = Profile()
