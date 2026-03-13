"""
Bantz — Notification Manager (#225)

Centralises all UI / desktop / sound notification side-effects that were
previously scattered across ``brain.py``.

Extracted surface:
  - ``toast_callback`` — TUI callback hook (set by the app on mount)
  - ``notify_toast()`` — push a toast to TUI → App.current → notify-send
  - ``push_toast()``   — public convenience wrapper

The module is **intentionally thin**: it just routes a (title, reason, type)
triple to the best available output channel.  No LLM, no routing, no memory.

Backward compatibility:
  ``brain.py`` re-exports ``_toast_callback`` and ``_notify_toast`` at
  module level so that existing callers (``app.py``, tests) keep working
  until they migrate to ``from bantz.core.notification_manager import …``.

Closes #225 (Part 2-A of #218).
"""
from __future__ import annotations

import logging

log = logging.getLogger("bantz.core.notifications")

# ── TUI callback hook ────────────────────────────────────────────────
# Set by the TUI app on mount:
#   ``notification_manager.toast_callback = app._on_brain_toast``
toast_callback = None


def notify_toast(title: str, reason: str = "", toast_type: str = "info") -> None:
    """Push a toast notification to the best available channel.

    Resolution order:
      1. ``toast_callback`` (set by TUI on mount)
      2. ``App.current.push_toast`` (Textual's thread-safe call)
      3. ``notify-send`` via ``bantz.agent.notifier`` (desktop fallback)
      4. Silent no-op

    Never raises — all exceptions are swallowed so the main pipeline
    is never interrupted by a notification failure.
    """
    if toast_callback:
        try:
            toast_callback(title, reason, toast_type)
            return
        except Exception:
            pass

    # Fallback: try App.current directly
    try:
        from textual.app import App as _App
        app = _App.current
        if app and hasattr(app, "push_toast"):
            app.call_from_thread(app.push_toast, title, reason, toast_type)
            return
    except Exception:
        pass

    # Fallback: desktop notification via notify-send (#153)
    try:
        from bantz.agent.notifier import notifier
        if notifier.enabled:
            notifier.send(f"Bantz: {title}", reason or "")
    except Exception:
        pass


def push_toast(title: str, reason: str = "", toast_type: str = "info") -> None:
    """Public convenience alias for :func:`notify_toast`."""
    notify_toast(title, reason, toast_type)
