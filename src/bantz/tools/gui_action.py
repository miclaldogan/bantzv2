"""
Bantz v3 — GUI Action Tool (#123)

Unified tool for natural-language GUI interactions.
Bridges the Navigator pipeline (AT-SPI → Cache → VLM → fallback) with
the Input Control backend to execute commands like:

    "click the Send button in Firefox"
    "type 'hello world' into the search bar in Chrome"
    "open a new tab in Firefox"

Registered as ``gui_action`` in the tool registry.
"""
from __future__ import annotations

import logging
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tools.gui_action")


class GUIActionTool(BaseTool):
    """One-shot GUI interaction: navigate to element → perform action.

    Actions:
        click          — find element → click its center
        double_click   — find element → double-click
        right_click    — find element → right-click
        type           — find element → click → type text
        focus          — focus application window (no element needed)
        navigate       — just find element, return coordinates
        stats          — navigation analytics

    Required args:
        app   — application name (e.g., "firefox", "vscode")

    Optional args:
        label  — element description (e.g., "search bar", "send button")
        text   — text to type (for action="type")
        role   — AT-SPI role filter (e.g., "push button")
    """

    name = "gui_action"
    description = (
        "Navigate to a UI element in any application and perform an action "
        "(click, double_click, right_click, type, focus). Uses AT-SPI, "
        "spatial cache, and VLM vision in order of speed."
    )
    risk_level = "moderate"

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "click")
        app = kwargs.get("app", "")
        label = kwargs.get("label", "")
        text = kwargs.get("text", "")
        role = kwargs.get("role")

        # ── Stats ────────────────────────────────────────────
        if action == "stats":
            return self._stats(app or None)

        # ── Input validation ─────────────────────────────────
        if not app:
            return ToolResult(
                success=False,
                output="Application name is required.",
                error="missing_app",
            )

        if action not in ("focus",) and not label:
            return ToolResult(
                success=False,
                output="Element label is required for this action.",
                error="missing_label",
            )

        # ── Check input control is enabled for actions that need it
        if action not in ("navigate", "stats"):
            from bantz.config import config
            if not config.input_control_enabled:
                # Allow navigate-only even when input control is off
                if action != "focus":
                    return ToolResult(
                        success=False,
                        output="Input control is disabled. Enable BANTZ_INPUT_CONTROL_ENABLED first.",
                        error="input_disabled",
                    )

        # ── Execute ──────────────────────────────────────────
        from bantz.vision.navigator import navigator

        if action == "navigate":
            # Just find, don't act
            nav = await navigator.navigate_to(app, label, role_filter=role)
            if nav.found:
                cx, cy = nav.center
                return ToolResult(
                    success=True,
                    output=(
                        f"Found '{label}' in {app} at ({cx}, {cy}) "
                        f"via {nav.method} (conf={nav.confidence:.2f}, {nav.latency_ms:.0f}ms)"
                    ),
                    data=nav.to_dict(),
                )
            return ToolResult(
                success=False,
                output=f"Could not find '{label}' in {app}.",
                data=nav.to_dict(),
                error="not_found",
            )

        # Actions that interact with the UI
        result = await navigator.execute_action(
            action, app, label, text=text, role_filter=role,
        )

        if result.success:
            return ToolResult(
                success=True,
                output=result.message,
                data=result.to_dict(),
            )

        return ToolResult(
            success=False,
            output=result.error or f"Failed to {action} '{label}' in {app}.",
            data=result.to_dict(),
            error=result.error,
        )

    def _stats(self, app: str | None) -> ToolResult:
        """Navigation analytics."""
        try:
            from bantz.vision.navigator import navigator
            stats = navigator.analytics.app_stats(app)
            if not stats.get("methods"):
                return ToolResult(
                    success=True,
                    output="No navigation data yet.",
                    data=stats,
                )
            lines = [f"Navigation stats{f' for {app}' if app else ''}:"]
            for m in stats["methods"]:
                rate = m["successes"] / max(m["attempts"], 1) * 100
                lines.append(
                    f"  {m['method']}: {m['successes']}/{m['attempts']} "
                    f"({rate:.0f}%) avg={m['avg_latency_ms']:.0f}ms"
                )
            lines.append(f"Total: {stats['total_attempts']} attempts")
            return ToolResult(
                success=True,
                output="\n".join(lines),
                data=stats,
            )
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), error=str(exc))


# ── Auto-register ─────────────────────────────────────────────────────────────

_tool = GUIActionTool()
registry.register(_tool)
