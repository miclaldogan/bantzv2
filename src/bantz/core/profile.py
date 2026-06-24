"""
Bantz v2 — User Profile

Stores user identity and preferences via DAL (SQLite), with JSON fallback.
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
from typing import TYPE_CHECKING, Any

from bantz.core.secure_io import secure_write_text

if TYPE_CHECKING:
    from bantz.data.store import ProfileStore as _ProfileStore


_PROFILE_PATH = Path.home() / ".local" / "share" / "bantz" / "profile.json"

_DEFAULT_PREFERENCES: dict[str, Any] = {
    "briefing_sections": ["schedule", "weather", "mail", "calendar", "classroom"],
    "news_sources": ["hn", "google"],
    "response_style": "casual",
    # Personal-preference schema consumed by the vision executor handoff
    # ("bana biraz AC/DC aç" = play me some AC/DC → favorite song,
    # preferred service, browser).
    "music": {
        "favorite_artists": [],          # e.g. ["AC/DC", "Metallica"]
        "artist_favorites": {},          # e.g. {"AC/DC": "Thunderstruck"}
        "preferred_service": "yt_music",  # "yt_music" | "spotify"
    },
    "apps": {
        "browser": "firefox",
        "music_player": "yt_music",
    },
}

_SERVICE_URLS: dict[str, str] = {
    "yt_music": "https://music.youtube.com",
    "spotify": "https://open.spotify.com",
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
        self._store: _ProfileStore | None = None
        self._load()

    def bind_store(self, store: _ProfileStore) -> None:
        """Bind a DAL store for persistence (called by DataLayer)."""
        self._store = store
        self._load()  # reload from new store

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._store:
            self._data = self._store.load()
            return
        # JSON fallback
        if _PROFILE_PATH.exists():
            try:
                self._data = json.loads(_PROFILE_PATH.read_text("utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def save(self, data: dict[str, Any]) -> None:
        merged = {**_DEFAULTS, **data}
        # Merge preferences sub-dict properly
        merged["preferences"] = {
            **_DEFAULT_PREFERENCES,
            **data.get("preferences", {}),
        }
        self._data = merged
        if self._store:
            self._store.save(self._data)
            return
        # JSON fallback
        _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        secure_write_text(
            _PROFILE_PATH,
            json.dumps(self._data, ensure_ascii=False, indent=2),
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

    def get_relevant(self, intent: str, **kwargs: Any) -> dict[str, Any]:
        """Pre-fetch the preference slice relevant to *intent*.

        Used by the routing layer before handing a goal to the vision
        executor, so goal strings carry resolved values, never placeholders.

        >>> profile.get_relevant("play_music", artist="AC/DC")
        {'artist': 'AC/DC', 'song': 'Thunderstruck',
         'service': 'yt_music', 'service_url': 'https://music.youtube.com',
         'browser': 'firefox'}
        """
        prefs = self.preferences
        music = {**_DEFAULT_PREFERENCES["music"], **prefs.get("music", {})}
        apps = {**_DEFAULT_PREFERENCES["apps"], **prefs.get("apps", {})}

        if intent == "play_music":
            artist = kwargs.get("artist") or ""
            favorites = music.get("artist_favorites", {})
            # Normalized artist match: case- and punctuation-insensitive,
            # so "acdc" and "ac dc" both resolve "AC/DC".
            def _norm(s: str) -> str:
                return "".join(c for c in s.lower() if c.isalnum())
            song = ""
            for known, fav in favorites.items():
                if _norm(known) == _norm(artist):
                    artist, song = known, fav
                    break
            service = music.get("preferred_service", "yt_music")
            return {
                "artist": artist,
                "song": song,
                "service": service,
                "service_url": _SERVICE_URLS.get(service, _SERVICE_URLS["yt_music"]),
                "browser": apps.get("browser", "firefox"),
            }
        if intent == "open_app":
            return {"browser": apps.get("browser", "firefox"), **apps}
        return {}

    @property
    def path(self) -> Path:
        if self._store:
            return self._store.path
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
