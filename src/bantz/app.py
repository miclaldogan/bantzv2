"""
Bantz v2/v3 — Textual UI (backward-compatibility shim)

The TUI has moved to bantz.interface.tui.app.
This file re-exports for backward compatibility.
"""
from bantz.interface.tui.app import BantzApp, run  # noqa: F401
from bantz.interface.tui.panels.system import SystemStatus  # noqa: F401
from bantz.interface.tui.panels.chat import ChatLog, ThinkingLabel  # noqa: F401

__all__ = ["BantzApp", "run", "SystemStatus", "ChatLog", "ThinkingLabel"]