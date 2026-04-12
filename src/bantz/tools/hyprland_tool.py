"""
Bantz — Hyprland Tool (#365)

BaseTool implementation for controlling the Hyprland desktop environment.

Actions:
    status   — Get current desktop status (monitors, workspaces, active window)
    windows  — List all open windows
    focus    — Focus a window by title or class
    move     — Move a window to a workspace
    layout   — Apply / reset Bantz 60/40 tiling layout
    exec     — Launch a command via Hyprland
    widgets  — Get widget data (cpu, ram, gpu, weather, etc.)
"""
from __future__ import annotations

import logging
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tools.hyprland")


class HyprlandTool(BaseTool):
    name = "hyprland"
    description = (
        "Control the Hyprland desktop environment: window management, "
        "workspaces, tiling layout, and widget data. "
        "Actions: status, windows, focus, move, layout, exec, widgets."
    )
    risk_level = "moderate"

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = (kwargs.get("action") or "status").lower().strip()

        dispatch = {
            "status": self._status,
            "windows": self._windows,
            "focus": self._focus,
            "move": self._move,
            "layout": self._layout,
            "exec": self._exec,
            "widgets": self._widgets,
        }
        handler = dispatch.get(action)
        if not handler:
            return ToolResult(
                success=False,
                output=f"Unknown action: {action}",
                error=f"Valid actions: {', '.join(dispatch)}",
            )
        try:
            return await handler(**kwargs)
        except Exception as exc:
            log.exception("hyprland.%s failed", action)
            return ToolResult(success=False, output="", error=str(exc))

    async def _status(self, **kwargs: Any) -> ToolResult:
        from bantz.desktop.hyprctl import HyprctlClient
        client = HyprctlClient()
        summary = await client.status_summary()
        if not summary.get("running"):
            return ToolResult(
                success=True,
                output="Hyprland is not running.",
                data=summary,
            )
        parts = [
            f"🖥️ Hyprland Desktop Status:",
            f"  Monitors: {summary['monitors']}",
            f"  Workspaces: {summary['workspaces']}",
        ]
        if summary.get("active_window"):
            parts.append(f"  Active: {summary['active_window']} ({summary.get('active_class', '')})")
        return ToolResult(
            success=True,
            output="\n".join(parts),
            data=summary,
        )

    async def _windows(self, **kwargs: Any) -> ToolResult:
        from bantz.desktop.hyprctl import HyprctlClient
        client = HyprctlClient()
        clients = await client.get_clients()
        if not clients:
            return ToolResult(
                success=True,
                output="No windows found (Hyprland may not be running).",
            )
        lines = ["Open windows:"]
        for c in clients:
            focus = " ◄" if c.focused else ""
            lines.append(
                f"  [{c.workspace_id}] {c.class_name}: {c.title} "
                f"({c.width}×{c.height}){focus}"
            )
        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"windows": [
                {"title": c.title, "class": c.class_name,
                 "workspace": c.workspace_id, "address": c.address}
                for c in clients
            ]},
        )

    async def _focus(self, **kwargs: Any) -> ToolResult:
        from bantz.desktop.hyprctl import HyprctlClient
        client = HyprctlClient()
        target = kwargs.get("title") or kwargs.get("class_name") or kwargs.get("target", "")
        if not target:
            return ToolResult(
                success=False, output="",
                error="Provide 'title' or 'class_name' to focus.",
            )
        clients = await client.get_clients()
        target_lower = target.lower()
        match = None
        for c in clients:
            if (target_lower in c.title.lower() or
                    target_lower in c.class_name.lower()):
                match = c
                break
        if not match:
            return ToolResult(
                success=False,
                output=f"No window matching '{target}' found.",
            )
        result = await client.focus_window(match.address)
        return ToolResult(
            success=result.success,
            output=f"Focused: {match.title} ({match.class_name})" if result.success
                   else f"Failed to focus: {result.error}",
        )

    async def _move(self, **kwargs: Any) -> ToolResult:
        from bantz.desktop.hyprctl import HyprctlClient
        client = HyprctlClient()
        target = kwargs.get("title") or kwargs.get("class_name") or kwargs.get("target", "")
        workspace = kwargs.get("workspace")
        if not target or workspace is None:
            return ToolResult(
                success=False, output="",
                error="Provide 'target' and 'workspace' parameters.",
            )
        clients = await client.get_clients()
        target_lower = target.lower()
        match = None
        for c in clients:
            if (target_lower in c.title.lower() or
                    target_lower in c.class_name.lower()):
                match = c
                break
        if not match:
            return ToolResult(
                success=False,
                output=f"No window matching '{target}' found.",
            )
        result = await client.move_window_to_workspace(match.address, workspace)
        return ToolResult(
            success=result.success,
            output=f"Moved {match.title} to workspace {workspace}" if result.success
                   else f"Failed: {result.error}",
        )

    async def _layout(self, **kwargs: Any) -> ToolResult:
        from bantz.desktop.hyprctl import HyprctlClient
        client = HyprctlClient()
        ok = await client.apply_bantz_layout()
        return ToolResult(
            success=ok,
            output="Applied Bantz 60/40 tiling layout." if ok
                   else "Could not apply layout (bantz-chat window not found or Hyprland not running).",
        )

    async def _exec(self, **kwargs: Any) -> ToolResult:
        from bantz.desktop.hyprctl import HyprctlClient
        client = HyprctlClient()
        command = kwargs.get("command", "")
        if not command:
            return ToolResult(
                success=False, output="",
                error="Provide 'command' to execute.",
            )
        result = await client.exec_once(command)
        return ToolResult(
            success=result.success,
            output=f"Executed: {command}" if result.success
                   else f"Failed: {result.error}",
        )

    async def _widgets(self, **kwargs: Any) -> ToolResult:
        from bantz.desktop.widgets import WidgetDataProvider
        provider = WidgetDataProvider()
        widget_name = kwargs.get("widget") or kwargs.get("name") or "status"
        data = provider.get(widget_name)
        return ToolResult(
            success=True,
            output=data,
            data={"widget": widget_name, "value": data},
        )


# ── Register ──────────────────────────────────────────────────────────────────
registry.register(HyprlandTool())
