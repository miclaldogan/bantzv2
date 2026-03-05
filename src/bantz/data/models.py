"""
Bantz v3 — Data Models

Pydantic models for all persistent entities in the Bantz ecosystem.
These replace the raw dicts that were passed around in v2.

Usage:
    from bantz.data.models import Message, Reminder, Place, UserProfile

    msg = Message(role="user", content="hello")
    place = Place(key="dorm", label="Yurt", lat=41.27, lon=36.33)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message in a conversation."""

    id: int = 0
    conversation_id: int = 0
    role: str  # "user" | "assistant" | "system"
    content: str
    tool_used: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"from_attributes": True}


class Conversation(BaseModel):
    """A conversation session with metadata."""

    id: int = 0
    started_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    message_count: int = 0
    first_message: str = ""


class Reminder(BaseModel):
    """A scheduled reminder or recurring task."""

    id: int = 0
    title: str
    fire_at: datetime
    repeat: str = "none"  # "none" | "daily" | "weekly" | "weekdays" | "custom"
    repeat_interval: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    fired: bool = False
    snoozed_until: Optional[datetime] = None
    trigger_place: Optional[str] = None


class Place(BaseModel):
    """A named geographic location with geofence radius."""

    key: str
    label: str
    lat: float = 0.0
    lon: float = 0.0
    radius: float = 100.0


class ScheduleEntry(BaseModel):
    """A class/event in the weekly university timetable."""

    name: str
    time: str  # "HH:MM"
    duration: int = 60
    location: str = ""
    type: str = ""  # "lecture" | "lab" | "seminar"


class UserProfile(BaseModel):
    """User identity and preference settings."""

    name: str = ""
    university: str = ""
    department: str = ""
    year: int = 0
    pronoun: str = "casual"
    tone: str = "casual"
    preferred_address: str = ""
    preferences: dict[str, Any] = Field(default_factory=dict)


class SessionInfo(BaseModel):
    """Launch-tracking data for absence-aware greetings."""

    last_seen: Optional[datetime] = None
    session_count: int = 0
    absence_hours: float = 0.0
    absence_label: str = ""
    is_first: bool = False
