"""
Bantz v3 — JSON File Store Implementations

Backward-compatible JSON adapters for profile, places, schedule, and session.
These read/write the same files as v2, ensuring seamless migration.

    store = JSONProfileStore(base_dir)
    data  = store.load()           # reads ~/.local/share/bantz/profile.json
    store.save({"name": "Iclal"})  # writes it back
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bantz.data.store import PlaceStore, ProfileStore, ScheduleStore, SessionStore
import os

log = logging.getLogger("bantz.data.json")


# ── helpers ───────────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict:
    """Read a JSON file, returning {} on any error."""
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            log.warning("Failed to read %s — returning empty dict", path)
    return {}


def _write_json(path: Path, data: dict | list) -> None:
    """Write JSON securely with 0o600 permissions."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    content = json.dumps(data, ensure_ascii=False, indent=2)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    path.chmod(0o600)


# ━━ Profile ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class JSONProfileStore(ProfileStore):
    """Reads / writes ``~/.local/share/bantz/profile.json``."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or (Path.home() / ".local" / "share" / "bantz")
        self._path = self._base / "profile.json"

    def load(self) -> dict[str, Any]:
        return _read_json(self._path)

    def save(self, data: dict[str, Any]) -> None:
        _write_json(self._path, data)

    def exists(self) -> bool:
        return self._path.exists()

    @property
    def path(self) -> Path:
        return self._path


# ━━ Places ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class JSONPlaceStore(PlaceStore):
    """Reads / writes ``~/.local/share/bantz/places.json``."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or (Path.home() / ".local" / "share" / "bantz")
        self._path = self._base / "places.json"

    def load_all(self) -> dict[str, dict]:
        return _read_json(self._path)

    def save_all(self, data: dict[str, dict]) -> None:
        _write_json(self._path, data)

    def exists(self) -> bool:
        return self._path.exists()

    @property
    def path(self) -> Path:
        return self._path


# ━━ Schedule ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class JSONScheduleStore(ScheduleStore):
    """Reads / writes ``~/.local/share/bantz/schedule.json``."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or (Path.home() / ".local" / "share" / "bantz")
        self._path = self._base / "schedule.json"

    def load(self) -> dict[str, list[dict]]:
        return _read_json(self._path)

    def save(self, data: dict[str, list[dict]]) -> None:
        _write_json(self._path, data)

    def exists(self) -> bool:
        return self._path.exists()

    @property
    def path(self) -> Path:
        return self._path


# ━━ Session ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class JSONSessionStore(SessionStore):
    """Reads / writes ``~/.local/share/bantz/session.json``."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or (Path.home() / ".local" / "share" / "bantz")
        self._path = self._base / "session.json"

    def load(self) -> dict:
        return _read_json(self._path)

    def save(self, data: dict) -> None:
        _write_json(self._path, data)

    @property
    def path(self) -> Path:
        return self._path
