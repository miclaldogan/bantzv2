"""
Bantz — Non-blocking toast notifications (#137).

Custom toast widget system integrated with InterventionQueue for
proactive alerts.  Toasts never steal focus from chat input.

Architecture:
    InterventionQueue ──→ ToastContainer ──→ ToastWidget (×3 max)
                               │                    │
                         dismiss_top()         auto-dismiss
                         accept_top()          (TTL-based)
                               │
                    ToastAccepted / ToastDismissed / ToastExpired
                               │
                        RL feedback (reward/penalty)

Toast types:
    INFO     cyan    5 s auto-dismiss    system notifications
    SUCCESS  green   3 s auto-dismiss    completion notices
    WARNING  amber  10 s auto-dismiss    maintenance alerts
    ERROR    red     ∞  manual dismiss   critical errors
    ACTION   blue    ∞  accept/dismiss   RL suggestions, reminders
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from textual.app import ComposeResult
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static

log = logging.getLogger("bantz.toast")


# ═══════════════════════════════════════════════════════════════════════════
# Enums & constants
# ═══════════════════════════════════════════════════════════════════════════


class ToastType(Enum):
    """Visual category for a toast notification."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    ACTION = "action"


# Auto-dismiss durations (seconds).  None → manual dismiss only.
TOAST_AUTO_DISMISS: dict[ToastType, float | None] = {
    ToastType.INFO: 5.0,
    ToastType.SUCCESS: 3.0,
    ToastType.WARNING: 10.0,
    ToastType.ERROR: None,
    ToastType.ACTION: None,
}

TOAST_ICONS: dict[ToastType, str] = {
    ToastType.INFO: "ℹ",
    ToastType.SUCCESS: "✓",
    ToastType.WARNING: "⚠",
    ToastType.ERROR: "✗",
    ToastType.ACTION: "◆",
}


# ═══════════════════════════════════════════════════════════════════════════
# Generic toast payload (for non-intervention toasts)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ToastData:
    """Simple toast payload for system / brain notifications.

    Compatible with Intervention's attribute interface so ToastWidget
    can render both types uniformly via ``getattr``.
    """
    title: str
    reason: str = ""
    ttl: float = 0.0
    source: str = "system"
    action: str | None = None
    state_key: str | None = None
    type: Any = None  # InterventionType compat — left None for generics

    @property
    def remaining_ttl(self) -> float:
        return self.ttl

    @property
    def expired(self) -> bool:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Textual Messages
# ═══════════════════════════════════════════════════════════════════════════


class ToastAccepted(Message):
    """Fired when the user accepts an action toast."""
    def __init__(self, intervention: Any) -> None:
        super().__init__()
        self.intervention = intervention


class ToastDismissed(Message):
    """Fired when the user explicitly dismisses a toast."""
    def __init__(self, intervention: Any) -> None:
        super().__init__()
        self.intervention = intervention


class ToastExpired(Message):
    """Fired when a toast auto-dismisses after its TTL."""
    def __init__(self, intervention: Any) -> None:
        super().__init__()
        self.intervention = intervention


# ═══════════════════════════════════════════════════════════════════════════
# ToastWidget  —  a single non-blocking notification
# ═══════════════════════════════════════════════════════════════════════════


class ToastWidget(Static):
    """A single non-blocking toast notification.

    CRITICAL: ``can_focus = False`` — keyboard stays on chat input at
    all times.  The user interacts with toasts via app-level bindings
    (Escape to dismiss) or text commands (accept/dismiss/never).
    """

    can_focus = False

    def __init__(
        self,
        intervention: Any,
        toast_type: ToastType = ToastType.INFO,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.intervention = intervention
        self.toast_type = toast_type
        self._dismiss_timer: Timer | None = None
        self.add_class(f"toast--{toast_type.value}")

    def on_mount(self) -> None:
        """Start auto-dismiss timer and slide-in animation."""
        # ── Determine TTL ──
        ttl = TOAST_AUTO_DISMISS.get(self.toast_type)
        iv_ttl = getattr(self.intervention, "remaining_ttl", 0)
        if iv_ttl and iv_ttl > 0:
            if ttl is None or iv_ttl < ttl:
                ttl = iv_ttl

        if ttl is not None and ttl > 0:
            self._dismiss_timer = self.set_timer(ttl, self._auto_expire)

        # ── Slide-in animation ──
        self.add_class("toast-enter")
        self.set_timer(0.05, self._animate_in)

        self._render_content()

    def _animate_in(self) -> None:
        """Remove entrance class so CSS transition triggers the slide."""
        self.remove_class("toast-enter")

    def _render_content(self) -> None:
        """Build rich-markup content for this toast."""
        icon = TOAST_ICONS.get(self.toast_type, "•")
        title = getattr(self.intervention, "title", str(self.intervention))
        reason = getattr(self.intervention, "reason", "")

        parts: list[str] = [f"{icon} {title}"]
        if reason:
            parts.append(f"  [dim]{reason}[/dim]")

        if self.toast_type == ToastType.ACTION:
            ttl_val = getattr(self.intervention, "remaining_ttl", 0)
            ttl_str = f" ({int(ttl_val)}s)" if ttl_val and ttl_val > 0 else ""
            parts.append(
                f"  [bold cyan]\\[Accept][/] [bold red]\\[Dismiss][/]{ttl_str}"
            )
        elif self.toast_type == ToastType.ERROR:
            parts.append("  [bold red]\\[Esc: Dismiss][/]")

        self.update("\n".join(parts))

    # ── User actions ──────────────────────────────────────────────────

    def _auto_expire(self) -> None:
        """Called when the auto-dismiss timer fires."""
        self.post_message(ToastExpired(self.intervention))
        self.remove()

    def dismiss(self) -> None:
        """Manually dismiss this toast."""
        if self._dismiss_timer:
            self._dismiss_timer.stop()
        self.post_message(ToastDismissed(self.intervention))
        self.remove()

    def accept(self) -> None:
        """Accept this toast (for ACTION type)."""
        if self._dismiss_timer:
            self._dismiss_timer.stop()
        self.post_message(ToastAccepted(self.intervention))
        self.remove()


# ═══════════════════════════════════════════════════════════════════════════
# ToastContainer  —  manages visible + overflow toasts
# ═══════════════════════════════════════════════════════════════════════════


class ToastContainer(Widget):
    """Non-focus-stealing container for up to MAX_VISIBLE toasts.

    Overflow is queued and promoted automatically when a visible toast
    is dismissed / accepted / expired.  Sits on the ``toast`` layer so
    it floats above chat content without disrupting layout.
    """

    can_focus = False
    MAX_VISIBLE = 3

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._overflow: list[tuple[Any, ToastType]] = []

    def compose(self) -> ComposeResult:
        yield from ()  # empty — toasts are mounted dynamically

    # ── Properties ────────────────────────────────────────────────────

    @property
    def has_toasts(self) -> bool:
        return len(self.query("ToastWidget")) > 0

    @property
    def toast_count(self) -> int:
        return len(self.query("ToastWidget"))

    @property
    def top_toast(self) -> ToastWidget | None:
        """Most-recently added (last) toast widget."""
        toasts = self.query("ToastWidget")
        return toasts.last() if toasts else None

    @property
    def has_action_toast(self) -> bool:
        """Whether any visible toast is actionable."""
        for t in self.query("ToastWidget"):
            if t.toast_type == ToastType.ACTION:
                return True
        return False

    # ── Public API ────────────────────────────────────────────────────

    def push_toast(
        self,
        intervention: Any,
        toast_type: ToastType | None = None,
    ) -> None:
        """Add a new toast notification.

        Determines ``ToastType`` from the intervention if not specified.
        Queues overflow if ``MAX_VISIBLE`` is reached.
        """
        if toast_type is None:
            toast_type = self._infer_type(intervention)

        if self.toast_count >= self.MAX_VISIBLE:
            self._overflow.append((intervention, toast_type))
            log.debug("Toast overflow queued (%d)", len(self._overflow))
            return

        widget = ToastWidget(intervention, toast_type)
        self.mount(widget)
        log.debug(
            "Toast mounted: %s [%s]",
            getattr(intervention, "title", "?"),
            toast_type.value,
        )

    def dismiss_top(self) -> None:
        """Dismiss the most recent (top) toast."""
        toast = self.top_toast
        if toast:
            toast.dismiss()
            self._promote_overflow()

    def accept_top(self) -> None:
        """Accept the most recent ACTION toast."""
        toast = self.top_toast
        if toast and toast.toast_type == ToastType.ACTION:
            toast.accept()
            self._promote_overflow()

    def clear_all(self) -> None:
        """Dismiss every visible toast and clear the overflow queue."""
        for toast in self.query("ToastWidget"):
            toast.dismiss()
        self._overflow.clear()

    def remove_by_intervention(self, iv: Any) -> None:
        """Remove the toast displaying a specific intervention object.

        Used when the user responds via text chat (accept / dismiss / never)
        so the corresponding toast is cleaned up too.
        """
        for toast in self.query("ToastWidget"):
            if toast.intervention is iv:
                if toast._dismiss_timer:
                    toast._dismiss_timer.stop()
                toast.remove()
                self._promote_overflow()
                return

    # ── Internal ──────────────────────────────────────────────────────

    def _promote_overflow(self) -> None:
        """Mount the next queued toast if there is room."""
        if self._overflow and self.toast_count < self.MAX_VISIBLE:
            iv, tt = self._overflow.pop(0)
            self.push_toast(iv, tt)

    def on_toast_expired(self, event: ToastExpired) -> None:
        self._promote_overflow()

    def on_toast_dismissed(self, event: ToastDismissed) -> None:
        self._promote_overflow()

    def on_toast_accepted(self, event: ToastAccepted) -> None:
        self._promote_overflow()

    # ── Type inference ────────────────────────────────────────────────

    @staticmethod
    def _infer_type(intervention: Any) -> ToastType:
        """Map ``Intervention.type`` to ``ToastType``."""
        iv_type = getattr(intervention, "type", None)
        if iv_type is None:
            return ToastType.INFO

        type_val = iv_type.value if hasattr(iv_type, "value") else str(iv_type)

        if type_val == "error_alert":
            return ToastType.ERROR

        # Anything with an RL action is actionable
        action = getattr(intervention, "action", None)
        if action:
            return ToastType.ACTION

        if type_val == "reminder":
            return ToastType.ACTION

        if type_val == "maintenance":
            return ToastType.WARNING

        return ToastType.INFO
