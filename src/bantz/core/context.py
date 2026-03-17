"""
Bantz — BantzContext: request-scoped data carrier (#224)

A plain dataclass that travels through every stage of the brain pipeline
(translation → routing → tool-execution → finalisation → response).

By **carrying** all intermediate data the context object:
  - eliminates the need for modules to import each other just to pass data,
  - breaks the circular-import chains that plague the current God-Object,
  - gives each future sub-module a clean, typed contract.

This module is intentionally dependency-free — it imports nothing from
the ``bantz`` package so it can be imported from anywhere without risk.

Closes #224 (Part 1-A of #218).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BantzContext:
    """Immutable-ish carrier for one request→response cycle.

    Every field has a safe default so callers only need to set the
    fields relevant to their pipeline stage.  Later stages enrich
    the same instance as it flows downstream.

    Fields are grouped by pipeline phase — keep this ordering when
    adding new ones.
    """

    # ── 1. Input ──────────────────────────────────────────────────────
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_input: str = ""
    is_remote: bool = False          # True for Telegram / API callers
    confirmed: bool = False          # second pass after destructive-confirm

    # ── 2. Translation ────────────────────────────────────────────────
    en_input: str = ""               # English-translated input (same as user_input when EN)
    source_lang: str = "en"          # ISO-639-1 code detected by bridge

    # ── 3. Time context ──────────────────────────────────────────────
    time_context: dict[str, Any] = field(default_factory=dict)
    # snapshot of time_ctx: {time_segment, day_name, location, prompt_hint, …}

    # ── 4. RLHF / feedback ───────────────────────────────────────────
    feedback: str | None = None      # "positive" | "negative" | None
    feedback_hint: str = ""          # one-shot system-prompt injection

    # ── 5. Memory & persona hints ─────────────────────────────────────
    graph_context: str = ""          # graph memory context string
    vector_context: str = ""         # semantic-search past messages
    deep_memory: str = ""            # spontaneous deep memory recall
    memory_combined: str = ""        # budget-trimmed unified memory (#211)
    desktop_context: str = ""        # AppDetector workspace snapshot
    persona_state: str = ""          # persona builder output
    formality_hint: str = ""         # bonding-meter hint
    style_hint: str = ""             # response_style + pronoun instruction
    profile_hint: str = ""           # user profile summary

    # ── 6. Routing / intent ───────────────────────────────────────────
    route: str = "chat"              # "chat" | "tool"
    tool_name: str | None = None     # resolved tool name (or internal _tts_stop etc.)
    tool_args: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "safe"         # "safe" | "moderate" | "destructive"
    is_quick_route: bool = False     # True when resolved by regex quick_route

    # ── 7. Execution ──────────────────────────────────────────────────
    tool_success: bool | None = None # None = not executed yet
    tool_output: str = ""            # raw ToolResult.output
    tool_data: dict[str, Any] = field(default_factory=dict)  # ToolResult.data

    # ── 8. Finalisation / response ────────────────────────────────────
    response: str = ""               # final assistant response text
    is_streaming: bool = False       # True when response delivered via stream
    needs_confirm: bool = False      # waiting for destructive-op confirm
    pending_command: str = ""        # command shown in confirm prompt
    pending_tool: str = ""           # tool held for confirm
    pending_args: dict[str, Any] = field(default_factory=dict)

    # ── 9. Timestamps ─────────────────────────────────────────────────
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    # ── helpers ───────────────────────────────────────────────────────

    @property
    def is_tool_call(self) -> bool:
        """True when routing selected a tool (not plain chat)."""
        return self.route == "tool" and bool(self.tool_name)

    @property
    def has_memory(self) -> bool:
        """True when any memory context was injected."""
        return bool(
            self.memory_combined
            or self.graph_context or self.vector_context or self.deep_memory
        )

    @property
    def elapsed_ms(self) -> float | None:
        """Wall-clock milliseconds from creation to completion."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.created_at) * 1000

    def mark_complete(self) -> None:
        """Stamp ``completed_at`` — call at end of pipeline."""
        self.completed_at = time.time()

    def as_log_dict(self) -> dict[str, Any]:
        """Compact dict suitable for structured logging / telemetry."""
        return {
            "sid": self.session_id,
            "input": self.user_input[:80],
            "lang": self.source_lang,
            "route": self.route,
            "tool": self.tool_name,
            "risk": self.risk_level,
            "ok": self.tool_success,
            "stream": self.is_streaming,
            "ms": round(self.elapsed_ms, 1) if self.elapsed_ms is not None else None,
        }
