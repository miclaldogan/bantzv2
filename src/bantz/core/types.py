"""
Bantz — Core type definitions (hotfix #228)

Houses ``BrainResult`` and other shared response types so that
**no module needs to import brain.py** just to reference the
orchestrator's return value.

This file is intentionally dependency-light — it only uses stdlib
types so it can be safely imported from anywhere in the package tree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class Attachment:
    """File attachment produced by a tool (e.g. a screenshot / daguerreotype).

    Carried by BrainResult so Telegram (and future interfaces) can deliver
    the bytes directly to the user without writing anything to disk.
    """

    type: str               # "image" | "document"
    data: bytes             # raw bytes (JPEG for screenshots)
    caption: str = ""       # butler's description
    filename: str = "bantz_attachment"
    mime_type: str = "image/jpeg"


@dataclass
class BrainResult:
    """Standard response payload returned by the Brain orchestrator.

    Also used by ``routing_engine`` for internal-tool results, and
    by interface layers (TUI, Telegram) to drive rendering.
    """

    response: str
    tool_used: str | None
    tool_result: Any | None = None   # ToolResult — kept as Any to avoid coupling
    needs_confirm: bool = False
    pending_command: str = ""
    pending_tool: str = ""
    pending_args: dict = field(default_factory=dict)
    stream: AsyncIterator[str] | None = None
    attachments: list = field(default_factory=list)  # list[Attachment]
