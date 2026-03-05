"""
Bantz v3 — Abstract Store Interfaces

Every data backend implements one or more of these contracts.
Brain and services never touch sqlite3/json directly — they go through these.

Hierarchy:
    ConversationStore — messages & sessions      (SQLite)
    ReminderStore     — scheduled reminders      (SQLite)
    ProfileStore      — user identity/prefs      (JSON → SQLite later)
    PlaceStore        — named GPS locations      (JSON → SQLite later)
    ScheduleStore     — weekly timetable         (JSON → SQLite later)
    SessionStore      — launch tracking          (JSON → SQLite later)
    GraphStore        — knowledge graph          (Neo4j)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ━━ Conversation / Message Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ConversationStore(ABC):
    """Conversation & message persistence (SQLite, etc.)."""

    @abstractmethod
    def init(self, db_path: Path) -> None:
        """Open / create the database at *db_path*."""

    @abstractmethod
    def new_session(self) -> int:
        """Start a new conversation. Returns session id."""

    @abstractmethod
    def resume_session(self, session_id: int) -> bool:
        """Resume an existing session. Returns False if not found."""

    @property
    @abstractmethod
    def session_id(self) -> Optional[int]:
        """Current active session id, or None."""

    @abstractmethod
    def add(self, role: str, content: str, tool_used: Optional[str] = None) -> int:
        """Persist a message. Returns message id."""

    @abstractmethod
    def context(self, n: int = 12) -> list[dict]:
        """Last *n* messages as ``[{role, content}]`` for LLM context."""

    @abstractmethod
    def last_n(self, n: int = 20) -> list[dict]:
        """Last *n* messages with full metadata."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across all conversations."""

    @abstractmethod
    def search_by_date(self, date: datetime, limit: int = 20) -> list[dict]:
        """Messages from a specific date."""

    @abstractmethod
    def conversation_list(self, limit: int = 20) -> list[dict]:
        """Recent conversations with message counts."""

    @abstractmethod
    def stats(self) -> dict:
        """DB statistics for diagnostics."""

    @abstractmethod
    def prune(self, keep_days: int = 90) -> int:
        """Delete old conversations. Returns count deleted."""

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""


# ━━ Reminder Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ReminderStore(ABC):
    """Reminder & task persistence (SQLite, etc.)."""

    @abstractmethod
    def init(self, db_path: Path) -> None:
        """Open / create the database at *db_path*."""

    @abstractmethod
    def add(
        self,
        title: str,
        fire_at: datetime,
        repeat: str = "none",
        repeat_interval: int = 0,
        trigger_place: Optional[str] = None,
    ) -> int:
        """Create a reminder. Returns reminder id."""

    @abstractmethod
    def check_due(self) -> list[dict]:
        """Return and advance all due reminders."""

    @abstractmethod
    def list_upcoming(self, limit: int = 10) -> list[dict]:
        """Upcoming (unfired) reminders sorted by fire_at."""

    @abstractmethod
    def list_all(self, limit: int = 20) -> list[dict]:
        """All reminders (including fired), newest first."""

    @abstractmethod
    def cancel(self, reminder_id: int) -> bool:
        """Delete by id. Returns True if found."""

    @abstractmethod
    def cancel_by_title(self, title: str) -> int:
        """Delete all matching title. Returns count."""

    @abstractmethod
    def snooze(self, reminder_id: int, minutes: int = 10) -> bool:
        """Snooze a reminder. Returns True if found."""

    @abstractmethod
    def due_today(self) -> list[dict]:
        """Reminders due today (for briefing)."""

    @abstractmethod
    def check_place_due(self, place_key: str) -> list[dict]:
        """Location-triggered reminders for a place."""

    @abstractmethod
    def count_active(self) -> int:
        """Count unfired reminders."""


# ━━ Profile Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ProfileStore(ABC):
    """User profile persistence."""

    @abstractmethod
    def load(self) -> dict[str, Any]:
        """Read the full profile dict."""

    @abstractmethod
    def save(self, data: dict[str, Any]) -> None:
        """Write the profile dict."""

    @abstractmethod
    def exists(self) -> bool:
        """Does a saved profile exist?"""

    @property
    @abstractmethod
    def path(self) -> Path:
        """Filesystem path (for diagnostics / --doctor)."""


# ━━ Place Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PlaceStore(ABC):
    """Named-places persistence."""

    @abstractmethod
    def load_all(self) -> dict[str, dict]:
        """Read all places {key: {label, lat, lon, radius}}."""

    @abstractmethod
    def save_all(self, data: dict[str, dict]) -> None:
        """Write all places."""

    @abstractmethod
    def exists(self) -> bool:
        """Does a saved places file exist?"""

    @property
    @abstractmethod
    def path(self) -> Path:
        """Filesystem path."""


# ━━ Schedule Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ScheduleStore(ABC):
    """Weekly timetable persistence."""

    @abstractmethod
    def load(self) -> dict[str, list[dict]]:
        """Read weekly schedule {monday: [{name, time, ...}], ...}."""

    @abstractmethod
    def save(self, data: dict[str, list[dict]]) -> None:
        """Write weekly schedule."""

    @abstractmethod
    def exists(self) -> bool:
        """Does a saved schedule exist?"""

    @property
    @abstractmethod
    def path(self) -> Path:
        """Filesystem path."""


# ━━ Session Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SessionStore(ABC):
    """Launch-tracking session persistence (absence detection)."""

    @abstractmethod
    def load(self) -> dict:
        """Read session data {last_seen, session_count}."""

    @abstractmethod
    def save(self, data: dict) -> None:
        """Write session data."""

    @property
    @abstractmethod
    def path(self) -> Path:
        """Filesystem path."""


# ━━ Graph Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GraphStore(ABC):
    """Knowledge-graph persistence (Neo4j or similar)."""

    @abstractmethod
    async def init(self) -> bool:
        """Connect to graph DB. Returns True on success."""

    @abstractmethod
    async def close(self) -> None:
        """Disconnect from graph DB."""

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Is the graph store connected and usable?"""

    @abstractmethod
    async def extract_and_store(
        self,
        user_msg: str,
        assistant_msg: str,
        tool_used: Optional[str] = None,
        tool_data: Optional[dict] = None,
    ) -> None:
        """Extract entities from an exchange and persist them."""

    @abstractmethod
    async def context_for(self, user_msg: str) -> str:
        """Return graph context string relevant to the user message."""
