"""
Bantz v3 — Desktop Automation Tool (#322)

Unified tool for computer use & desktop automation.  Bridges the AT-SPI
accessibility tree (read) with input_control (act) to provide a single
tool that the LLM can invoke for end-to-end desktop tasks.

Architecture::

    Brain → "open Calculator, type 5*9, press Enter"
        → DesktopTool
            ├── open_app("gnome-calculator")           — subprocess
            ├── get_ui_tree()                          — active window AT-SPI
            ├── interact("name:5", "click")            — find + click
            ├── interact("name:×", "click")            — find + click
            ├── interact("name:9", "click")            — find + click
            ├── press_key("Return")                    — input_control.hotkey
            └── close_app("gnome-calculator")          — WM close

Locator Format
--------------
  ``name:Submit``          — match element name (fuzzy)
  ``role:push button``     — match element role
  ``name:OK,role:push button`` — combine name + role

Requires: AT-SPI2 (``python3-gi``, ``gir1.2-atspi-2.0``), input backend
(``pyautogui`` / ``pynput`` / ``xdotool``).
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tools.desktop")

# ── Well-known app launchers (name → command) ─────────────────────────────────
_APP_COMMANDS: dict[str, str] = {
    "calculator": "gnome-calculator",
    "calc": "gnome-calculator",
    "hesap makinesi": "gnome-calculator",
    "terminal": "gnome-terminal",
    "konsole": "konsole",
    "files": "nautilus",
    "file manager": "nautilus",
    "dosya yöneticisi": "nautilus",
    "text editor": "gnome-text-editor",
    "gedit": "gedit",
    "notepad": "gnome-text-editor",
    "firefox": "firefox",
    "chrome": "google-chrome",
    "chromium": "chromium",
    "brave": "brave-browser",
    "settings": "gnome-control-center",
    "ayarlar": "gnome-control-center",
    "system monitor": "gnome-system-monitor",
    "screenshot": "gnome-screenshot",
}


# ── Locator parsing ──────────────────────────────────────────────────────────

_LOCATOR_RE = re.compile(
    r"(name|role|id)\s*:\s*(.+?)(?:,|$)", re.IGNORECASE,
)


def parse_locator(locator: str) -> dict[str, str]:
    """Parse ``name:OK,role:push button`` → ``{'name': 'OK', 'role': 'push button'}``."""
    parts: dict[str, str] = {}
    for match in _LOCATOR_RE.finditer(locator):
        key = match.group(1).lower().strip()
        val = match.group(2).strip()
        if val:
            parts[key] = val
    # If no key:value found, treat entire string as name search
    if not parts and locator.strip():
        parts["name"] = locator.strip()
    return parts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_app_command(name: str) -> str:
    """Return the actual launch command for an app name."""
    lower = name.lower().strip()
    if lower in _APP_COMMANDS:
        return _APP_COMMANDS[lower]
    # Check if it's already a valid command
    if shutil.which(lower):
        return lower
    # Try with hyphens/underscores
    for variant in (lower.replace(" ", "-"), lower.replace(" ", "_")):
        if shutil.which(variant):
            return variant
    return lower  # best effort


def _get_active_window_name() -> str:
    """Return the name of the currently active window via xdotool."""
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def _get_active_window_pid() -> int:
    """Return PID of the active window."""
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowpid"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return 0


def _get_active_app_name() -> str:
    """Return the process name of the active window's application."""
    pid = _get_active_window_pid()
    if pid:
        try:
            import psutil
            return psutil.Process(pid).name()
        except Exception:
            pass
        # Fallback: /proc
        try:
            comm = open(f"/proc/{pid}/comm").read().strip()
            return comm
        except Exception:
            pass
    return ""


def _format_tree_compact(
    node: dict, lines: list[str], indent: int = 0, max_lines: int = 80,
) -> None:
    """Compact textual representation of the AT-SPI tree for LLM consumption."""
    if len(lines) >= max_lines:
        if len(lines) == max_lines:
            lines.append("  … (truncated)")
        return

    name = node.get("name", "")
    role = node.get("role", "")
    center = node.get("center")
    visible = node.get("visible", True)
    enabled = node.get("enabled", True)

    # Skip invisible / containers without name
    if not visible:
        for child in node.get("children", []):
            _format_tree_compact(child, lines, indent, max_lines)
        return

    prefix = "  " * indent
    parts = [f"{prefix}[{role}]"]
    if name:
        parts.append(f'"{name}"')
    if center:
        parts.append(f"@ ({center[0]}, {center[1]})")
    if not enabled:
        parts.append("(disabled)")

    line = " ".join(parts)
    # Only include lines with meaningful content
    if name or role not in ("filler", "panel", "section", "redundant object"):
        lines.append(line)

    for child in node.get("children", []):
        _format_tree_compact(child, lines, indent + 1, max_lines)


# ── Interactive element extraction ────────────────────────────────────────────

_INTERACTIVE_ROLES: frozenset[str] = frozenset({
    "push button", "toggle button", "check box", "radio button",
    "entry", "text", "link", "menu item", "combo box", "tab",
    "slider", "spin button", "menu", "tool bar item",
    "page tab", "list item", "tree item",
})


def _extract_interactive_elements(
    node: dict, elements: list[dict], max_elements: int = 100,
) -> None:
    """Walk the AT-SPI tree and collect only interactive elements with bounds."""
    if len(elements) >= max_elements:
        return

    role = (node.get("role") or "").lower()
    name = node.get("name", "")
    center = node.get("center")
    visible = node.get("visible", True)
    enabled = node.get("enabled", True)

    if visible and enabled and role in _INTERACTIVE_ROLES and center:
        elements.append({
            "name": name,
            "role": node.get("role", ""),
            "center": center,
            "bounds": node.get("bounds"),
        })

    for child in node.get("children", []):
        _extract_interactive_elements(child, elements, max_elements)


# ══════════════════════════════════════════════════════════════════════════════
#  Desktop Tool
# ══════════════════════════════════════════════════════════════════════════════

class DesktopTool(BaseTool):
    """Unified desktop automation: read UI tree + interact with elements."""

    name = "desktop"
    description = (
        "Desktop automation tool for controlling native applications. "
        "Actions:\n"
        "  get_ui_tree — list interactive UI elements of the active window\n"
        "  interact    — find a UI element by locator and act on it "
        "(click/double_click/right_click/type)\n"
        "  click       — shorthand: find element by locator and click it\n"
        "  type_text   — type text into the currently focused element\n"
        "  press_key   — press a keyboard shortcut (e.g. 'ctrl+s', 'Return')\n"
        "  open_app    — launch a desktop application by name\n"
        "  close_app   — close the currently focused or named application\n"
        "  list_windows — list all accessible application windows\n"
        "  active_window — get info about the currently focused window\n"
        "Params: action (str), app (str), locator (str, e.g. 'name:Send' or "
        "'role:push button'), text (str), keys (str, e.g. 'ctrl+c').\n"
        "Locator format: 'name:Submit', 'role:entry', 'name:OK,role:push button'."
    )
    risk_level = "moderate"

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "get_ui_tree")).strip().lower()

        dispatch = {
            "get_ui_tree": self._get_ui_tree,
            "ui_tree": self._get_ui_tree,
            "tree": self._get_ui_tree,
            "interact": self._interact,
            "click": self._click,
            "type_text": self._type_text,
            "type": self._type_text,
            "press_key": self._press_key,
            "hotkey": self._press_key,
            "open_app": self._open_app,
            "open": self._open_app,
            "close_app": self._close_app,
            "close": self._close_app,
            "list_windows": self._list_windows,
            "active_window": self._active_window,
        }

        handler = dispatch.get(action)
        if handler is None:
            return ToolResult(
                success=False, output="",
                error=(
                    f"Unknown action: '{action}'. Available: "
                    f"{', '.join(sorted(set(dispatch.keys())))}"
                ),
            )

        try:
            return await handler(kwargs)
        except Exception as exc:
            log.exception("DesktopTool.%s failed", action)
            return ToolResult(success=False, output="", error=f"Desktop action '{action}' failed: {exc}")

    # ── get_ui_tree ───────────────────────────────────────────────────────

    async def _get_ui_tree(self, kwargs: dict) -> ToolResult:
        """Return interactive elements for the active (or named) window."""
        app_name = str(kwargs.get("app", "")).strip()

        if not app_name:
            app_name = _get_active_app_name() or _get_active_window_name()
        if not app_name:
            return ToolResult(
                success=False, output="",
                error="Could not determine the active window. Specify 'app' parameter.",
            )

        try:
            from bantz.tools.accessibility import get_element_tree
        except ImportError:
            return ToolResult(
                success=False, output="",
                error="AT-SPI2 is not available. Install: sudo apt install python3-gi gir1.2-atspi-2.0",
            )

        tree = get_element_tree(app_name)
        if tree is None:
            return ToolResult(
                success=False, output="",
                error=f"No accessibility tree found for '{app_name}'. Is the app running and accessible?",
            )

        # Extract interactive elements
        elements: list[dict] = []
        _extract_interactive_elements(tree, elements)

        if not elements:
            # Fall back to showing the full tree
            lines = [f"🌳 UI tree for '{app_name}' (no interactive elements detected):"]
            _format_tree_compact(tree, lines, max_lines=60)
            return ToolResult(
                success=True, output="\n".join(lines),
                data={"app": app_name, "tree": tree, "elements": []},
            )

        lines = [f"🖥️ {len(elements)} interactive elements in '{app_name}':"]
        for i, elem in enumerate(elements, 1):
            name = elem["name"] or "(unnamed)"
            role = elem["role"]
            cx, cy = elem["center"]
            lines.append(f"  {i}. [{role}] \"{name}\" @ ({cx}, {cy})")

        return ToolResult(
            success=True, output="\n".join(lines),
            data={"app": app_name, "elements": elements, "count": len(elements)},
        )

    # ── interact ──────────────────────────────────────────────────────────

    async def _interact(self, kwargs: dict) -> ToolResult:
        """Find element by locator, then perform action (click/type/etc.)."""
        locator_str = str(kwargs.get("locator", "")).strip()
        interact_action = str(kwargs.get("interact_action", "click")).strip().lower()
        text = str(kwargs.get("text", ""))
        app_name = str(kwargs.get("app", "")).strip()

        if not locator_str:
            return ToolResult(success=False, output="", error="Locator is required. Use 'name:Submit' or 'role:entry'.")

        locator = parse_locator(locator_str)
        label = locator.get("name", "")
        role_filter = locator.get("role")

        if not app_name:
            app_name = _get_active_app_name() or _get_active_window_name()

        if not app_name:
            return ToolResult(success=False, output="", error="Could not determine active window. Specify 'app' parameter.")

        # Find the element via AT-SPI
        try:
            from bantz.tools.accessibility import find_element
        except ImportError:
            return ToolResult(success=False, output="", error="AT-SPI2 is not available.")

        if not label and role_filter:
            label = role_filter

        element = find_element(app_name, label, role_filter=role_filter)
        if element is None:
            return ToolResult(
                success=False, output="",
                error=f"Element '{locator_str}' not found in '{app_name}'.",
            )

        cx, cy = element["center"]
        element.get("score", 0.0)

        # Perform the action
        if interact_action in ("click", "left_click"):
            return await self._do_click(cx, cy, element, "click", app_name, locator_str)
        elif interact_action == "double_click":
            return await self._do_click(cx, cy, element, "double_click", app_name, locator_str)
        elif interact_action == "right_click":
            return await self._do_click(cx, cy, element, "right_click", app_name, locator_str)
        elif interact_action in ("type", "type_text"):
            return await self._do_type_at(cx, cy, text, element, app_name, locator_str)
        else:
            return ToolResult(
                success=False, output="",
                error=f"Unknown interact action: '{interact_action}'. Use: click, double_click, right_click, type.",
            )

    # ── click (shorthand) ──────────────────────────────────────────────────

    async def _click(self, kwargs: dict) -> ToolResult:
        """Shorthand: find element by locator and click it."""
        locator_str = str(kwargs.get("locator", kwargs.get("target", ""))).strip()
        if not locator_str:
            return ToolResult(success=False, output="", error="Specify 'locator' (e.g. 'name:OK') or 'target'.")
        kwargs["locator"] = locator_str
        kwargs["interact_action"] = "click"
        return await self._interact(kwargs)

    # ── type_text ─────────────────────────────────────────────────────────

    async def _type_text(self, kwargs: dict) -> ToolResult:
        """Type text. If locator given, click element first."""
        text = str(kwargs.get("text", "")).strip()
        if not text:
            return ToolResult(success=False, output="", error="Specify 'text' to type.")

        locator_str = str(kwargs.get("locator", "")).strip()

        if locator_str:
            # Click element first, then type
            kwargs["interact_action"] = "click"
            click_result = await self._interact(kwargs)
            if not click_result.success:
                return click_result
            await asyncio.sleep(0.15)

        # Type the text
        try:
            import bantz.tools.input_control as ic
            await ic.type_text(text)
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Failed to type text: {exc}")

        display = text[:80] + "…" if len(text) > 80 else text
        return ToolResult(
            success=True,
            output=f"⌨️  Typed: \"{display}\" ({len(text)} chars)",
            data={"text": text, "length": len(text)},
        )

    # ── press_key ─────────────────────────────────────────────────────────

    async def _press_key(self, kwargs: dict) -> ToolResult:
        """Press a keyboard shortcut (e.g. 'ctrl+s', 'Return', 'alt+f4')."""
        keys_raw = str(kwargs.get("keys", kwargs.get("key", ""))).strip()
        if not keys_raw:
            return ToolResult(success=False, output="", error="Specify 'keys' (e.g. 'ctrl+s', 'Return').")

        key_list = [k.strip() for k in keys_raw.split("+") if k.strip()]

        try:
            import bantz.tools.input_control as ic
            await ic.hotkey(*key_list)
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Failed to press key: {exc}")

        combo = "+".join(key_list)
        return ToolResult(
            success=True,
            output=f"⌨️  Pressed: {combo}",
            data={"keys": key_list, "combo": combo},
        )

    # ── open_app ──────────────────────────────────────────────────────────

    async def _open_app(self, kwargs: dict) -> ToolResult:
        """Launch a desktop application by name."""
        app_name = str(kwargs.get("app", kwargs.get("name", ""))).strip()
        if not app_name:
            return ToolResult(success=False, output="", error="Specify 'app' name to open.")

        command = _resolve_app_command(app_name)
        log.info("Opening app: %s → %s", app_name, command)

        try:
            # Try xdg-open for .desktop entries first
            proc = await asyncio.create_subprocess_exec(
                command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            # Give the app a moment to start
            await asyncio.sleep(1.0)

            # Check if process is still alive (didn't crash immediately)
            if proc.returncode is not None and proc.returncode != 0:
                return ToolResult(
                    success=False, output="",
                    error=f"Application '{command}' exited with code {proc.returncode}.",
                )

            return ToolResult(
                success=True,
                output=f"🚀 Launched '{app_name}' (command: {command}, pid: {proc.pid})",
                data={"app": app_name, "command": command, "pid": proc.pid},
            )
        except FileNotFoundError:
            return ToolResult(
                success=False, output="",
                error=f"Application '{command}' not found. Is it installed?",
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Failed to open '{app_name}': {exc}")

    # ── close_app ─────────────────────────────────────────────────────────

    async def _close_app(self, kwargs: dict) -> ToolResult:
        """Close an application by name or the active window."""
        app_name = str(kwargs.get("app", "")).strip()

        if app_name:
            # Try wmctrl first
            try:
                proc = await asyncio.create_subprocess_exec(
                    "wmctrl", "-c", app_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.communicate(), timeout=3)
                if proc.returncode == 0:
                    return ToolResult(
                        success=True,
                        output=f"✓ Closed '{app_name}'.",
                        data={"app": app_name, "method": "wmctrl"},
                    )
            except (FileNotFoundError, asyncio.TimeoutError):
                pass

            # xdotool fallback
            try:
                proc = await asyncio.create_subprocess_exec(
                    "xdotool", "search", "--name", app_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
                wids = stdout.decode().strip().split("\n")
                if wids and wids[0]:
                    await asyncio.create_subprocess_exec(
                        "xdotool", "windowactivate", "--sync", wids[0],
                    )
                    await asyncio.sleep(0.2)
            except (FileNotFoundError, asyncio.TimeoutError):
                pass

        # Send Alt+F4 to close active window
        try:
            import bantz.tools.input_control as ic
            await ic.hotkey("alt", "F4")
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Failed to close: {exc}")

        target = app_name or "active window"
        return ToolResult(
            success=True,
            output=f"✓ Sent close signal to '{target}'.",
            data={"app": target, "method": "alt+f4"},
        )

    # ── list_windows ──────────────────────────────────────────────────────

    async def _list_windows(self, kwargs: dict) -> ToolResult:
        """List all accessible applications/windows."""
        try:
            from bantz.tools.accessibility import list_applications
        except ImportError:
            return ToolResult(success=False, output="", error="AT-SPI2 is not available.")

        apps = list_applications()
        if not apps:
            return ToolResult(
                success=True,
                output="No accessible applications found.",
                data={"apps": []},
            )

        lines = [f"📋 {len(apps)} accessible applications:"]
        for app in sorted(apps):
            lines.append(f"  • {app}")

        return ToolResult(
            success=True, output="\n".join(lines),
            data={"apps": sorted(apps), "count": len(apps)},
        )

    # ── active_window ─────────────────────────────────────────────────────

    async def _active_window(self, kwargs: dict) -> ToolResult:
        """Get information about the currently focused window."""
        title = _get_active_window_name()
        app = _get_active_app_name()
        pid = _get_active_window_pid()

        if not title and not app:
            return ToolResult(success=False, output="", error="Could not determine the active window.")

        output = f"🖥️ Active window: {title or '(unknown title)'}"
        if app:
            output += f"\n   Application: {app}"
        if pid:
            output += f"\n   PID: {pid}"

        return ToolResult(
            success=True, output=output,
            data={"title": title, "app": app, "pid": pid},
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _do_click(
        self, x: int, y: int, element: dict, click_type: str,
        app_name: str, locator_str: str,
    ) -> ToolResult:
        """Execute a click at the resolved coordinates."""
        try:
            import bantz.tools.input_control as ic
            fn = getattr(ic, click_type)
            await fn(x, y)
        except Exception as exc:
            return ToolResult(
                success=False, output="",
                error=f"Found '{locator_str}' at ({x}, {y}) but click failed: {exc}",
            )

        action_past = {
            "click": "Clicked",
            "double_click": "Double-clicked",
            "right_click": "Right-clicked",
        }
        verb = action_past.get(click_type, "Acted on")
        name = element.get("name", locator_str)
        role = element.get("role", "")

        return ToolResult(
            success=True,
            output=f"🎯 {verb} '{name}' ({role}) at ({x}, {y}) in {app_name}",
            data={"x": x, "y": y, "element": element, "action": click_type},
        )

    async def _do_type_at(
        self, x: int, y: int, text: str, element: dict,
        app_name: str, locator_str: str,
    ) -> ToolResult:
        """Click an element, then type text into it."""
        if not text:
            return ToolResult(success=False, output="", error="Specify 'text' to type.")

        # Click first to focus the element
        try:
            import bantz.tools.input_control as ic
            await ic.click(x, y)
            await asyncio.sleep(0.15)
            await ic.type_text(text)
        except Exception as exc:
            return ToolResult(
                success=False, output="",
                error=f"Failed to type into '{locator_str}': {exc}",
            )

        display = text[:80] + "…" if len(text) > 80 else text
        name = element.get("name", locator_str)
        return ToolResult(
            success=True,
            output=f"⌨️  Typed \"{display}\" into '{name}' at ({x}, {y}) in {app_name}",
            data={"x": x, "y": y, "text": text, "element": element},
        )


# ── Auto-register ─────────────────────────────────────────────────────────────
try:
    registry.register(DesktopTool())
except Exception:
    pass
