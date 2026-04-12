"""
Bantz — Continuous Awareness Pipeline (#325)

Collects ambient desktop context every N seconds:
  • Active window title + process name (via xdotool + psutil)
  • Clipboard contents (via xclip)
  • Screenshot path (via grim, saved but not sent to VLM)

The collected state is kept in a rolling deque of the last 20 snapshots
and exposed via two clean methods:

  • ``get_current_context()``     → formatted string for system prompt injection
  • ``get_screenshot_for_vlm()``  → latest screenshot path (only when explicitly requested)

All subprocess calls are best-effort: any failure is logged at DEBUG level
and silently skipped — awareness must never crash Bantz.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger("bantz.awareness")

_SCREENSHOT_PATH = "/tmp/bantz_awareness_latest.png"


# ═══════════════════════════════════════════════════════════════════════════
# State dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AwarenessState:
    """A single point-in-time snapshot of the user's desktop context."""

    active_window_title: str = ""
    active_window_process: str = ""
    clipboard_text: str = ""
    screenshot_path: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════
# Collector
# ═══════════════════════════════════════════════════════════════════════════

class AwarenessCollector:
    """Background asyncio task that periodically captures desktop state.

    Usage::

        collector = AwarenessCollector(interval_s=15)
        asyncio.create_task(collector.run())

        # Later:
        ctx = collector.get_current_context()

    The ``run()`` coroutine loops forever until cancelled.  It is designed
    to be launched with ``asyncio.create_task()`` so it runs concurrently
    with the rest of Bantz without blocking the event loop.  All
    subprocess I/O is dispatched to the default executor so the loop is
    never blocked.
    """

    BUFFER_SIZE: int = 20

    def __init__(self, interval_s: float = 15.0) -> None:
        self.interval_s = interval_s
        self._buffer: deque[AwarenessState] = deque(maxlen=self.BUFFER_SIZE)
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────

    def get_current_context(self) -> str:
        """Return a formatted summary of the latest awareness state.

        Returns an empty string when no state has been collected yet.

        Example output::

            [Awareness] Active window: VS Code (code) | Clipboard: "def foo():" | Screenshot: /tmp/bantz_awareness_latest.png
        """
        if not self._buffer:
            return ""
        state = self._buffer[-1]
        parts: list[str] = []

        if state.active_window_title or state.active_window_process:
            win = state.active_window_title or "(unknown)"
            proc = f" ({state.active_window_process})" if state.active_window_process else ""
            parts.append(f"Active window: {win}{proc}")

        if state.clipboard_text:
            snippet = state.clipboard_text[:120].replace("\n", " ")
            parts.append(f'Clipboard: "{snippet}"')

        if state.screenshot_path:
            parts.append(f"Screenshot: {state.screenshot_path}")

        if not parts:
            return ""

        return "[Awareness] " + " | ".join(parts)

    def get_screenshot_for_vlm(self) -> Optional[str]:
        """Return the latest screenshot path, or None if none captured yet."""
        if not self._buffer:
            return None
        path = self._buffer[-1].screenshot_path
        return path if path else None

    @property
    def latest(self) -> Optional[AwarenessState]:
        """The most recently collected state, or None."""
        return self._buffer[-1] if self._buffer else None

    # ── Background task ────────────────────────────────────────────────

    async def run(self) -> None:
        """Long-running coroutine — collect state every ``interval_s`` seconds.

        Cancel the task to stop collection gracefully.
        """
        self._running = True
        log.info("AwarenessCollector: started (interval=%.0fs)", self.interval_s)
        try:
            while True:
                await self._collect()
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            log.info("AwarenessCollector: stopped")
            self._running = False

    # ── Internal helpers ───────────────────────────────────────────────

    async def _collect(self) -> None:
        """Capture one snapshot and append it to the rolling buffer."""
        loop = asyncio.get_running_loop()
        state = AwarenessState()

        # Run all subprocess calls concurrently in the executor
        title, proc, clipboard, screenshot_ok = await asyncio.gather(
            loop.run_in_executor(None, _get_active_window_title),
            loop.run_in_executor(None, _get_active_window_process),
            loop.run_in_executor(None, _get_clipboard),
            loop.run_in_executor(None, _capture_screenshot),
            return_exceptions=True,
        )

        state.active_window_title = title if isinstance(title, str) else ""
        state.active_window_process = proc if isinstance(proc, str) else ""
        state.clipboard_text = clipboard if isinstance(clipboard, str) else ""
        state.screenshot_path = _SCREENSHOT_PATH if screenshot_ok is True else ""
        state.timestamp = datetime.now()

        self._buffer.append(state)
        log.debug(
            "AwarenessCollector: collected — window=%r proc=%r clip_len=%d screenshot=%s",
            state.active_window_title[:40],
            state.active_window_process,
            len(state.clipboard_text),
            "ok" if state.screenshot_path else "none",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Subprocess helpers — all return empty/False on any error
# ═══════════════════════════════════════════════════════════════════════════

def _get_active_window_title() -> str:
    """Return the active window title via xdotool, or '' on failure."""
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log.debug("AwarenessCollector: xdotool title failed: %s", exc)
    return ""


def _get_active_window_process() -> str:
    """Return the process name of the active window, or '' on failure."""
    try:
        # Get the window ID first
        wid_result = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, timeout=3,
        )
        if wid_result.returncode != 0 or not wid_result.stdout.strip():
            return ""
        wid = wid_result.stdout.strip()

        # Get the PID for that window
        pid_result = subprocess.run(
            ["xdotool", "getwindowpid", wid],
            capture_output=True, text=True, timeout=3,
        )
        if pid_result.returncode != 0 or not pid_result.stdout.strip():
            return ""
        pid = int(pid_result.stdout.strip())

        # Resolve PID → process name via psutil
        import psutil
        try:
            return psutil.Process(pid).name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
        log.debug("AwarenessCollector: xdotool pid/process failed: %s", exc)
    return ""


def _get_clipboard() -> str:
    """Return clipboard text via xclip, or '' on failure / empty clipboard."""
    try:
        result = subprocess.run(
            ["xclip", "-o", "-selection", "clipboard"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log.debug("AwarenessCollector: xclip failed: %s", exc)
    return ""


def _capture_screenshot() -> bool:
    """Capture a screenshot to ``_SCREENSHOT_PATH`` via grim.

    Returns True on success, False on any failure.
    """
    try:
        result = subprocess.run(
            ["grim", _SCREENSHOT_PATH],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return True
        log.debug("AwarenessCollector: grim exited %d: %s", result.returncode, result.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log.debug("AwarenessCollector: grim failed: %s", exc)
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton — imported by brain.py and ghost_loop.py
# ═══════════════════════════════════════════════════════════════════════════

awareness_collector = AwarenessCollector()
"""The global awareness collector instance.  ``brain.py`` calls
``asyncio.create_task(awareness_collector.run())`` at startup when
``BANTZ_AWARENESS_ENABLED=true``."""
