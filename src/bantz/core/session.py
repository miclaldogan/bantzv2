"""
Bantz v2 — Session Tracker

Tracks launch timestamps for absence-aware greetings.
Stores data via DAL (SQLite), with JSON fallback for backward compat.

Schema:
    {
        "last_seen": "2026-02-19T16:30:00",
        "session_count": 47
    }
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bantz.data.store import SessionStore


_SESSION_PATH = Path.home() / ".local" / "share" / "bantz" / "session.json"


class SessionTracker:

    def __init__(self) -> None:
        self._store: SessionStore | None = None

    def bind_store(self, store: SessionStore) -> None:
        """Bind a DAL store for persistence (called by DataLayer)."""
        self._store = store

    def on_launch(self) -> dict[str, Any]:
        """
        Read last_seen, compute absence, update to now.
        Returns dict with absence_hours, absence_label, session_count, last_seen.
        """
        data = self._load()
        now = datetime.now()

        last_seen_str = data.get("last_seen")
        session_count = data.get("session_count", 0)

        last_seen: datetime | None = None
        absence_hours = 0.0
        if last_seen_str:
            try:
                last_seen = datetime.fromisoformat(last_seen_str)
                absence_hours = (now - last_seen).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

        session_count += 1

        # Persist updated values
        self._save({
            "last_seen": now.isoformat(timespec="seconds"),
            "session_count": session_count,
        })

        return {
            "absence_hours": absence_hours,
            "absence_label": self.absence_label(absence_hours),
            "session_count": session_count,
            "last_seen": last_seen,
            "is_first": last_seen is None,
        }

    @staticmethod
    def absence_label(hours: float) -> str:
        if hours < 1:
            return "a few minutes"
        if hours < 6:
            return "a few hours"
        if hours < 20:
            return "earlier today"
        if hours < 30:
            return "last night"
        if hours < 72:
            return "a few days"
        if hours < 168:
            return "about a week"
        return "a long time"

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._store:
            return self._store.load()
        # JSON fallback (pre-migration or standalone usage)
        if _SESSION_PATH.exists():
            try:
                return json.loads(_SESSION_PATH.read_text("utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self, data: dict) -> None:
        if self._store:
            self._store.save(data)
            return
        # JSON fallback
        _SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SESSION_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def path(self) -> Path:
        if self._store:
            return self._store.path
        return _SESSION_PATH


session_tracker = SessionTracker()
