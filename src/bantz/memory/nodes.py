"""
Bantz v3 — Node Schema

Typed node definitions for Neo4j.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class PersonNode:
    name: str
    relationship_type: str = "contact"  # contact | professor | friend | family
    importance: int = 5                  # 1-10
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    email: str = ""


@dataclass
class EventNode:
    title: str
    date: str
    attendees: list[str] = field(default_factory=list)
    outcome: str = ""
    location: str = ""


@dataclass
class TaskNode:
    description: str
    status: str = "open"     # open | in_progress | done | cancelled
    priority: str = "medium" # low | medium | high | urgent
    deadline: str = ""
    owner: str = ""


@dataclass
class TopicNode:
    name: str
    description: str = ""


@dataclass
class DecisionNode:
    what: str
    context: str = ""
    when: str = field(default_factory=lambda: datetime.now().isoformat())
    topic: str = ""


@dataclass
class LocationNode:
    name: str
    lat: float = 0.0
    lon: float = 0.0
    visit_frequency: int = 0


@dataclass
class DocumentNode:
    title: str
    path: str = ""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)


@dataclass
class ReminderNode:
    message: str
    trigger_type: str = "time"   # time | location
    trigger_value: str = ""
    status: str = "pending"


@dataclass
class CommitmentNode:
    description: str
    made_by: str = ""
    made_to: str = ""
    deadline: str = ""
    status: str = "open"
