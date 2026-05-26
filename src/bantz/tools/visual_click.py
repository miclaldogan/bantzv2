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
        "Click any visible UI element on screen by describing it in plain language. "
        "Params: target (str) = what to click (e.g. 'Send button', 'File menu', 'Terminal'); "
        "app (str, optional) = narrow the search to a specific app. "
        "Supports: click, double_click, right_click, hover. "
        "The tool finds the element automatically via accessibility tree + screen vision. "
        "Use for: 'click X', 'press the Y button', 'open the Z menu', 'click terminal'. "
        "Do NOT use shell or bash for GUI clicking. "
        "Example: 'click the terminal' → visual_click(target='Terminal', action='click')."
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

        # ── Step 1: Take a screenshot for context and VLM fallback ──────
        _screenshot_b64: str = ""
        try:
            from bantz.vision.screenshot import capture_base64
            _screenshot_b64 = await capture_base64() or ""
        except Exception as _ss_exc:
            log.debug("visual_click: pre-screenshot failed: %s", _ss_exc)

        # ── Step 2: Navigate (Cache -> AT-SPI -> VLM) ────────────────
        try:
            from bantz.vision.navigator import navigator

            nav_result = await navigator.navigate_to(
                app_name=app,
                element_label=target,
                screenshot_b64=_screenshot_b64 or None,
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
            # Describe what IS visible when element not found
            _visible_desc = await _describe_visible(target, _screenshot_b64)
            return ToolResult(
                success=False,
                output=_visible_desc,
                error=f"Could not find '{target}' via AT-SPI or VLM.",
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


async def _describe_visible(target: str, screenshot_b64: str) -> str:
    """Return a plain-text description of what's visible on screen.

    Called when the element cannot be found, so the user knows what Bantz
    sees and can correct the request. Uses VLM if enabled, AT-SPI list
    otherwise, and falls back to a generic message.
    """
    # Try AT-SPI apps list first (fast, no GPU)
    try:
        from bantz.tools.accessibility import list_applications
        apps = list_applications()
        if apps:
            app_list = ", ".join(apps[:8])
            suffix = f" (+{len(apps) - 8} more)" if len(apps) > 8 else ""
            return (
                f"I could not locate '{target}' on screen. "
                f"Accessible windows: {app_list}{suffix}. "
                f"Is '{target}' visible and the window focused?"
            )
    except Exception:
        pass

    # VLM fallback — describe the screen in plain text
    if screenshot_b64:
        try:
            from bantz.vision.remote_vlm import describe_screen
            result = await describe_screen(screenshot_b64, timeout=10)
            if result.success and result.raw_text:
                return (
                    f"I could not locate '{target}'. "
                    f"Here is what I see on screen:\n{result.raw_text[:600]}"
                )
        except Exception:
            pass

    return (
        f"I could not locate '{target}' via accessibility tree or VLM. "
        f"Please ensure the element is visible and the correct window is focused."
    )


registry.register(VisualClickTool())
