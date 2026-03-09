"""
Bantz — Desktop notification dispatcher (#153).

Zero-dependency Linux desktop notifications via ``notify-send``.
Integrates with the Intervention Queue (#126) and AppDetector (#127)
to deliver desktop notifications when the TUI is not the active window.

Architecture:
    InterventionQueue.pop()
      → _process_interventions() (TUI app.py)
        → notifier.dispatch(intervention)
          → Is TUI the active window? (AppDetector)
            YES → skip (already visible in chat)
            NO  → fire notify-send subprocess
          → Focus/quiet mode respected (matching TUI behaviour)

Priority → urgency mapping:
    CRITICAL → critical (sticky)
    HIGH     → critical (10 s)
    MEDIUM   → normal   (5 s)
    LOW      → low      (3 s)

Usage:
    from bantz.agent.notifier import notifier

    notifier.init()
    notifier.dispatch(intervention)     # smart: checks active window
    notifier.send("Title", "Body")      # manual override
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from bantz.agent.interventions import Intervention

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Priority → notify-send urgency / timeout mapping
# ═══════════════════════════════════════════════════════════════════════════

_URGENCY_MAP: dict[str, tuple[str, int]] = {
    # priority.value → (urgency, expire_ms)
    "critical": ("critical", 0),       # sticky
    "high":     ("critical", 10_000),
    "medium":   ("normal",   5_000),
    "low":      ("low",      3_000),
}

_DEFAULT_URGENCY = ("normal", 5_000)

# TUI window identifiers for active-window check
_TUI_IDENTIFIERS = {"bantz", "python", "python3"}


# ═══════════════════════════════════════════════════════════════════════════
# Notifier
# ═══════════════════════════════════════════════════════════════════════════


class Notifier:
    """Desktop notification dispatcher using ``notify-send``.

    - Detects ``notify-send`` availability at init.
    - Respects focus/quiet mode from the intervention queue.
    - Skips desktop notification when TUI is the active window.
    - Falls back to silent no-op if ``notify-send`` is missing.
    """

    def __init__(self) -> None:
        self._binary: Optional[str] = None
        self._available = False
        self._initialized = False
        self._enabled = True
        self._icon: str = ""
        self._sound = False
        self._sent_count = 0
        self._skipped_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────────

    def init(
        self,
        *,
        enabled: bool = True,
        icon: str = "",
        sound: bool = False,
    ) -> None:
        """Initialize the notifier.

        Detects ``notify-send`` and configures options.
        Safe to call multiple times (idempotent).
        """
        if self._initialized:
            return

        self._enabled = enabled
        self._icon = icon
        self._sound = sound

        # Check for notify-send binary
        self._binary = shutil.which("notify-send")
        self._available = self._binary is not None

        if not self._available:
            log.warning(
                "notify-send not found — desktop notifications disabled. "
                "Install: sudo apt install libnotify-bin"
            )

        self._initialized = True
        log.info(
            "Notifier initialized: available=%s enabled=%s",
            self._available,
            self._enabled,
        )

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def available(self) -> bool:
        return self._available

    @property
    def enabled(self) -> bool:
        return self._enabled and self._available

    # ── Active window check ───────────────────────────────────────────

    def _is_tui_active(self) -> bool:
        """Check if the TUI is the currently focused window (via AppDetector).

        Returns False if AppDetector is not initialized or not enabled.
        """
        try:
            from bantz.agent.app_detector import app_detector

            if not app_detector.initialized:
                return False

            win = app_detector.get_active_window()
            if not win:
                return False

            name_lower = win.name.lower()
            title_lower = win.title.lower()

            # Check if TUI-like window is active
            # TUI runs as "python -m bantz" so active window is usually
            # the terminal with "bantz" in the title
            if "bantz" in title_lower:
                return True

            # Also check if it's a terminal running bantz
            if name_lower in _TUI_IDENTIFIERS:
                return "bantz" in title_lower

            return False
        except Exception:
            return False

    # ── Focus / quiet filtering ───────────────────────────────────────

    def _should_skip(self, intervention: "Intervention") -> tuple[bool, str]:
        """Determine if this notification should be skipped.

        Returns (should_skip, reason).
        """
        if not self.enabled:
            return True, "notifier disabled"

        if not self._initialized:
            return True, "not initialized"

        # If TUI is the active window, user already sees the toast
        if self._is_tui_active():
            return True, "tui_active"

        # Check focus mode — only CRITICAL passes through
        try:
            from bantz.agent.interventions import intervention_queue, Priority

            if intervention_queue.initialized:
                if intervention_queue.focus and intervention.priority != Priority.CRITICAL:
                    return True, "focus_mode"
                if intervention_queue.quiet and intervention.priority != Priority.CRITICAL:
                    return True, "quiet_mode"
        except Exception:
            pass

        return False, ""

    # ── Send notification ─────────────────────────────────────────────

    def send(
        self,
        title: str,
        body: str = "",
        *,
        urgency: str = "normal",
        expire_ms: int = 5000,
    ) -> bool:
        """Send a desktop notification directly.

        Returns True if notification was sent successfully.
        """
        if not self.enabled or not self._binary:
            return False

        cmd = [
            self._binary,
            "--app-name", "Bantz",
            "--urgency", urgency,
        ]

        if expire_ms > 0:
            cmd.extend(["--expire-time", str(expire_ms)])

        if self._icon:
            cmd.extend(["--icon", self._icon])

        cmd.extend([title, body])

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                timeout=3,
            )
            self._sent_count += 1
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            log.debug("notify-send failed: %s", exc)
            return False

    # ── Dispatch from intervention ────────────────────────────────────

    def dispatch(self, intervention: "Intervention") -> bool:
        """Smart dispatch: check focus/quiet/active-window, then notify.

        This is the main entry point called from TUI's
        ``_process_interventions()`` — it decides whether to send a
        desktop notification based on context.

        Returns True if notification was actually sent.
        """
        should_skip, reason = self._should_skip(intervention)
        if should_skip:
            self._skipped_count += 1
            log.debug("Desktop notification skipped: %s", reason)
            return False

        # Map priority to urgency
        priority_val = intervention.priority.value if hasattr(intervention.priority, "value") else str(intervention.priority)
        urgency, expire_ms = _URGENCY_MAP.get(priority_val, _DEFAULT_URGENCY)

        # Build notification content
        title = f"Bantz: {intervention.title}"
        body = intervention.reason or ""

        return self.send(title, body, urgency=urgency, expire_ms=expire_ms)

    # ── Stats / Doctor ────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "initialized": self._initialized,
            "available": self._available,
            "enabled": self._enabled,
            "icon": self._icon,
            "sound": self._sound,
            "sent": self._sent_count,
            "skipped": self._skipped_count,
        }

    def status_line(self) -> str:
        state = "active" if self.enabled else ("unavailable" if not self._available else "disabled")
        return (
            f"state={state} "
            f"sent={self._sent_count} "
            f"skipped={self._skipped_count}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

notifier = Notifier()
