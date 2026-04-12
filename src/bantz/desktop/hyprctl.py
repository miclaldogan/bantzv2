"""
Bantz — Hyprland IPC Client (#365)

Async wrapper around ``hyprctl`` with JSON output parsing.
Provides typed access to monitors, workspaces, windows (clients), and
dispatchers for moving / resizing / focusing windows.

All calls go through ``hyprctl -j`` (JSON mode) and are parsed into
plain dicts. When Hyprland is not running the methods return sensible
defaults (empty lists, False, etc.) instead of raising.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("bantz.desktop.hyprctl")

# ── Hyprctl socket path ──────────────────────────────────────────────────────
# Hyprland exposes a UNIX socket at /tmp/hypr/$HYPRLAND_INSTANCE_SIGNATURE/.socket2.sock
# We use the CLI wrapper ``hyprctl`` instead for simplicity + compatibility.

_HYPRCTL = "hyprctl"


@dataclass
class MonitorInfo:
    """Parsed monitor from ``hyprctl monitors -j``."""
    name: str
    width: int
    height: int
    x: int = 0
    y: int = 0
    scale: float = 1.0
    focused: bool = False
    description: str = ""


@dataclass
class WorkspaceInfo:
    """Parsed workspace from ``hyprctl workspaces -j``."""
    id: int
    name: str
    monitor: str = ""
    windows: int = 0
    last_window_title: str = ""


@dataclass
class WindowInfo:
    """Parsed window/client from ``hyprctl clients -j``."""
    address: str
    title: str
    class_name: str = ""  # app_id for Wayland
    workspace_id: int = 0
    workspace_name: str = ""
    monitor: str = ""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    floating: bool = False
    focused: bool = False
    pid: int = 0


@dataclass
class HyprctlResult:
    """Result of a hyprctl dispatch / command."""
    success: bool
    output: str = ""
    error: str = ""


class HyprctlClient:
    """Async interface to Hyprland via ``hyprctl``."""

    def __init__(self) -> None:
        self._available: bool | None = None

    # ── Availability ──────────────────────────────────────────────────────

    async def is_available(self) -> bool:
        """Check if hyprctl is on PATH and Hyprland is running."""
        if self._available is not None:
            return self._available
        if not shutil.which(_HYPRCTL):
            self._available = False
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                _HYPRCTL, "version", "-j",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
            self._available = proc.returncode == 0
        except (asyncio.TimeoutError, OSError):
            self._available = False
        return self._available

    def reset_cache(self) -> None:
        """Reset the availability cache (useful after Hyprland starts)."""
        self._available = None

    # ── Raw execution ─────────────────────────────────────────────────────

    async def _run_json(self, *args: str) -> list[dict[str, Any]] | dict[str, Any]:
        """Execute ``hyprctl -j <args>`` and return parsed JSON."""
        cmd = [_HYPRCTL, "-j", *args]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            if proc.returncode != 0:
                log.warning("hyprctl %s failed: %s", args, stderr.decode().strip())
                return []
            return json.loads(stdout.decode())
        except asyncio.TimeoutError:
            log.warning("hyprctl %s timed out", args)
            return []
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("hyprctl %s error: %s", args, exc)
            return []

    async def _dispatch(self, *args: str) -> HyprctlResult:
        """Execute ``hyprctl dispatch <args>``."""
        cmd = [_HYPRCTL, "dispatch", *args]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            out = stdout.decode().strip()
            err = stderr.decode().strip()
            return HyprctlResult(
                success=proc.returncode == 0 and "ok" in out.lower(),
                output=out,
                error=err,
            )
        except asyncio.TimeoutError:
            return HyprctlResult(success=False, error="hyprctl dispatch timed out")
        except OSError as exc:
            return HyprctlResult(success=False, error=str(exc))

    async def _keyword(self, key: str, value: str) -> HyprctlResult:
        """Execute ``hyprctl keyword <key> <value>``."""
        cmd = [_HYPRCTL, "keyword", key, value]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            return HyprctlResult(
                success=proc.returncode == 0,
                output=stdout.decode().strip(),
                error=stderr.decode().strip(),
            )
        except (asyncio.TimeoutError, OSError) as exc:
            return HyprctlResult(success=False, error=str(exc))

    # ── Monitors ──────────────────────────────────────────────────────────

    async def get_monitors(self) -> list[MonitorInfo]:
        """List all connected monitors."""
        data = await self._run_json("monitors")
        if not isinstance(data, list):
            return []
        monitors = []
        for m in data:
            monitors.append(MonitorInfo(
                name=m.get("name", ""),
                width=m.get("width", 0),
                height=m.get("height", 0),
                x=m.get("x", 0),
                y=m.get("y", 0),
                scale=m.get("scale", 1.0),
                focused=m.get("focused", False),
                description=m.get("description", ""),
            ))
        return monitors

    # ── Workspaces ────────────────────────────────────────────────────────

    async def get_workspaces(self) -> list[WorkspaceInfo]:
        """List all active workspaces."""
        data = await self._run_json("workspaces")
        if not isinstance(data, list):
            return []
        return [
            WorkspaceInfo(
                id=w.get("id", 0),
                name=w.get("name", ""),
                monitor=w.get("monitor", ""),
                windows=w.get("windows", 0),
                last_window_title=w.get("lastwindowtitle", ""),
            )
            for w in data
        ]

    # ── Windows / Clients ─────────────────────────────────────────────────

    async def get_clients(self) -> list[WindowInfo]:
        """List all open windows."""
        data = await self._run_json("clients")
        if not isinstance(data, list):
            return []
        clients = []
        for c in data:
            ws = c.get("workspace", {})
            at = c.get("at", [0, 0])
            sz = c.get("size", [0, 0])
            clients.append(WindowInfo(
                address=c.get("address", ""),
                title=c.get("title", ""),
                class_name=c.get("class", ""),
                workspace_id=ws.get("id", 0) if isinstance(ws, dict) else 0,
                workspace_name=ws.get("name", "") if isinstance(ws, dict) else "",
                monitor=c.get("monitor", ""),
                x=at[0] if len(at) > 0 else 0,
                y=at[1] if len(at) > 1 else 0,
                width=sz[0] if len(sz) > 0 else 0,
                height=sz[1] if len(sz) > 1 else 0,
                floating=c.get("floating", False),
                focused=c.get("focus", False) or c.get("focusHistoryID", -1) == 0,
                pid=c.get("pid", 0),
            ))
        return clients

    async def get_active_window(self) -> WindowInfo | None:
        """Get the currently focused window."""
        data = await self._run_json("activewindow")
        if not isinstance(data, dict) or not data.get("address"):
            return None
        ws = data.get("workspace", {})
        at = data.get("at", [0, 0])
        sz = data.get("size", [0, 0])
        return WindowInfo(
            address=data.get("address", ""),
            title=data.get("title", ""),
            class_name=data.get("class", ""),
            workspace_id=ws.get("id", 0) if isinstance(ws, dict) else 0,
            workspace_name=ws.get("name", "") if isinstance(ws, dict) else "",
            monitor=data.get("monitor", ""),
            x=at[0] if len(at) > 0 else 0,
            y=at[1] if len(at) > 1 else 0,
            width=sz[0] if len(sz) > 0 else 0,
            height=sz[1] if len(sz) > 1 else 0,
            floating=data.get("floating", False),
            focused=True,
            pid=data.get("pid", 0),
        )

    # ── Dispatchers ───────────────────────────────────────────────────────

    async def move_to_workspace(self, workspace: int | str) -> HyprctlResult:
        """Switch to a workspace."""
        return await self._dispatch("workspace", str(workspace))

    async def move_window_to_workspace(
        self, address: str, workspace: int | str,
    ) -> HyprctlResult:
        """Move a specific window to a workspace by address."""
        return await self._dispatch(
            "movetoworkspacesilent", f"{workspace},address:{address}",
        )

    async def focus_window(self, address: str) -> HyprctlResult:
        """Focus a window by address."""
        return await self._dispatch("focuswindow", f"address:{address}")

    async def resize_window(self, address: str, width: int, height: int) -> HyprctlResult:
        """Resize a window by exact pixel amount."""
        return await self._dispatch(
            "resizewindowpixel", f"exact {width} {height},address:{address}",
        )

    async def move_window(self, address: str, x: int, y: int) -> HyprctlResult:
        """Move a window to exact coordinates."""
        return await self._dispatch(
            "movewindowpixel", f"exact {x} {y},address:{address}",
        )

    async def toggle_floating(self, address: str) -> HyprctlResult:
        """Toggle floating mode for a window."""
        return await self._dispatch("togglefloating", f"address:{address}")

    async def fullscreen(self, mode: int = 0) -> HyprctlResult:
        """Toggle fullscreen. mode: 0=full, 1=maximize, 2=no-gaps."""
        return await self._dispatch("fullscreen", str(mode))

    async def close_window(self, address: str) -> HyprctlResult:
        """Close a window gracefully."""
        return await self._dispatch("closewindow", f"address:{address}")

    async def exec_once(self, command: str) -> HyprctlResult:
        """Execute a command via Hyprland (like exec-once in config)."""
        return await self._dispatch("exec", command)

    # ── Window Rules (runtime) ────────────────────────────────────────────

    async def set_window_rule(self, rule: str) -> HyprctlResult:
        """Set a window rule at runtime via keyword."""
        return await self._keyword("windowrulev2", rule)

    # ── Layout helpers ────────────────────────────────────────────────────

    async def apply_bantz_layout(self) -> bool:
        """Apply the standard Bantz 60/40 tiling layout.

        Finds the bantz-chat window and resizes it to 60% of the primary
        monitor width. Returns True if layout was applied.
        """
        monitors = await self.get_monitors()
        if not monitors:
            return False
        primary = monitors[0]
        clients = await self.get_clients()
        bantz_win = None
        for c in clients:
            if c.class_name == "bantz-chat":
                bantz_win = c
                break
        if not bantz_win:
            return False
        # Resize bantz-chat to 60% width
        target_w = int(primary.width * 0.6)
        target_h = primary.height
        await self.resize_window(bantz_win.address, target_w, target_h)
        await self.move_window(bantz_win.address, primary.x, primary.y)
        return True

    # ── Status summary ────────────────────────────────────────────────────

    async def status_summary(self) -> dict[str, Any]:
        """Return a JSON-friendly status dict for waybar / tools."""
        available = await self.is_available()
        if not available:
            return {"running": False}
        monitors = await self.get_monitors()
        workspaces = await self.get_workspaces()
        active = await self.get_active_window()
        return {
            "running": True,
            "monitors": len(monitors),
            "workspaces": len(workspaces),
            "active_window": active.title if active else None,
            "active_class": active.class_name if active else None,
        }
