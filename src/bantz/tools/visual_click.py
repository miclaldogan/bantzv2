"""Bantz v3 - VisualClickTool (#185 Phase 1: The Butler's Eyes)

Exposes the unified navigation pipeline (Cache -> AT-SPI -> VLM) to the
LLM as a callable tool.  The LLM can now say "click the Send button"
and Bantz will find it on screen and physically click it.

Supported actions: click, double_click, right_click, hover

Safety: moderate (physically moves the mouse and performs clicks).
PyAutoGUI FAILSAFE remains active -- slam mouse to corner (0,0) to abort.
"""
from __future__ import annotations

import logging
from typing import Any

from bantz.config import config
from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tools.visual_click")

# Action -> input_control function name mapping
_ACTION_MAP = {
    "click": "click",
    "double_click": "double_click",
    "right_click": "right_click",
    "hover": "move_to",
}


class VisualClickTool(BaseTool):
    name = "visual_click"
    description = (
        "Click, double-click, right-click, or hover over ANY visible UI "
        "element on the user's screen.  Call this tool whenever the user "
        "asks you to interact with a graphical interface — buttons, menus, "
        "icons, links, tabs, text fields, or anything else they can see.  "
        "You only need to describe the element in plain language (e.g. "
        "'the Send button', 'File menu', 'Terminal tab'); the tool will "
        "find it automatically via accessibility tree and screen vision.  "
        "Optionally pass an 'app' hint (e.g. 'firefox') to narrow the "
        "search.  Works on any visible application window.  "
        "EXAMPLE: If user says 'click the terminal', call visual_click "
        "with target='terminal'. Do NOT attempt this via bash or shell."
    )
    risk_level = "moderate"

    async def execute(
        self,
        target: str = "",
        action: str = "click",
        app: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Find *target* on screen and perform *action*.

        Args:
            target: Human description of the UI element (e.g., "Send button").
            action: One of click, double_click, right_click, hover.
            app:    Optional application name hint (e.g., "firefox").
        """
        # ── Gate checks ───────────────────────────────────────────────
        if not config.input_control_enabled:
            return ToolResult(
                success=False, output="",
                error="Vision and input control are disabled. "
                      "Enable BANTZ_INPUT_CONTROL_ENABLED in .env.",
            )

        if not target:
            return ToolResult(
                success=False, output="",
                error="No target specified. Tell me what to click.",
            )

        action = action.lower().strip()
        if action not in _ACTION_MAP:
            return ToolResult(
                success=False, output="",
                error=f"Unknown action '{action}'. Supported: {', '.join(_ACTION_MAP)}.",
            )

        # ── Navigate (Cache -> AT-SPI -> VLM) ────────────────────────
        try:
            from bantz.vision.navigator import navigator

            nav_result = await navigator.navigate_to(
                app_name=app,
                element_label=target,
            )
        except Exception as exc:
            log.error("visual_click navigate_to failed: %s", exc, exc_info=True)
            return ToolResult(
                success=False,
                output=(
                    f"I attempted to scan the screen for '{target}' but the "
                    f"vision system reported an error, ma'am: {exc}"
                ),
                error=f"Navigation pipeline error: {exc}",
            )

        if not nav_result.found:
            return ToolResult(
                success=False,
                output=(
                    f"I peered through the glass pane but could not locate "
                    f"'{target}', ma'am. Perhaps it is obscured."
                ),
                data=nav_result.to_dict(),
            )

        # ── Execute action at found coordinates ──────────────────────
        cx, cy = nav_result.center

        try:
            import bantz.tools.input_control as ic
            fn_name = _ACTION_MAP[action]
            fn = getattr(ic, fn_name)
            await fn(cx, cy)
        except Exception as exc:
            log.error("visual_click %s(%d, %d) failed: %s", action, cx, cy, exc)
            return ToolResult(
                success=False,
                output=f"I found '{target}' but my hand slipped, ma'am: {exc}",
                error=f"Input action failed: {exc}",
                data=nav_result.to_dict(),
            )

        action_past = {
            "click": "clicked",
            "double_click": "double-clicked",
            "right_click": "right-clicked",
            "hover": "hovered over",
        }
        verb = action_past.get(action, f"{action}ed")

        log.info(
            "visual_click: %s '%s' at (%d, %d) via %s (conf=%.2f, %.0fms)",
            verb, target, cx, cy,
            nav_result.method, nav_result.confidence, nav_result.latency_ms,
        )

        return ToolResult(
            success=True,
            output=(
                f"Located and {verb} the '{target}' on the glass pane, ma'am. "
                f"Method: {nav_result.method}"
            ),
            data=nav_result.to_dict(),
        )


registry.register(VisualClickTool())
