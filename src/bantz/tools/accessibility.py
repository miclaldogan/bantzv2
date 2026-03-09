"""
Bantz v3 — Accessibility Tool (#119)

Linux AT-SPI2 accessibility reader for instant UI element location.
Uses the OS accessibility tree to find UI elements with bounding boxes —
instant (< 10ms) vs screenshot+VLM (> 2s), zero GPU cost.

Architecture:
    Brain: "click the Send button in Firefox"
        → accessibility.py
            ├── get_element_tree("firefox") → JSON tree with (x,y,w,h)
            └── find_element("firefox", "Send") → (1340, 680)

Requires: python3-gi + gir1.2-atspi-2.0 (system packages)
Works on: X11, Wayland (GNOME, KDE), GTK, Qt, Chromium/Electron apps

Usage:
    from bantz.tools.accessibility import AccessibilityTool
"""
from __future__ import annotations

import logging
import re
import subprocess
from typing import Any, Optional

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.accessibility")

# ── AT-SPI availability check ─────────────────────────────────────────────

_atspi = None
_atspi_available: Optional[bool] = None


def _init_atspi() -> bool:
    """Lazy-init AT-SPI2.  Returns True if available."""
    global _atspi, _atspi_available
    if _atspi_available is not None:
        return _atspi_available
    try:
        import gi
        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi
        _atspi = Atspi
        _atspi_available = True
        log.debug("AT-SPI2 loaded successfully")
    except (ImportError, ValueError) as exc:
        _atspi = None
        _atspi_available = False
        log.debug("AT-SPI2 unavailable: %s", exc)
    return _atspi_available


# ── Display server detection ──────────────────────────────────────────────

def detect_display_server() -> str:
    """Detect whether running X11 or Wayland."""
    import os
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session == "wayland":
        return "wayland"
    if session == "x11":
        return "x11"
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


# ── Fuzzy matching ────────────────────────────────────────────────────────

def _fuzzy_match(needle: str, haystack: str) -> float:
    """Simple fuzzy match score (0.0–1.0) between two strings."""
    if not needle or not haystack:
        return 0.0

    needle_lower = needle.lower().strip()
    haystack_lower = haystack.lower().strip()

    # Exact match
    if needle_lower == haystack_lower:
        return 1.0

    # Substring containment
    if needle_lower in haystack_lower:
        return 0.9

    if haystack_lower in needle_lower:
        return 0.8

    # Word overlap
    needle_words = set(needle_lower.split())
    haystack_words = set(haystack_lower.split())
    if needle_words and haystack_words:
        overlap = needle_words & haystack_words
        if overlap:
            return 0.5 + 0.3 * (len(overlap) / max(len(needle_words), len(haystack_words)))

    # Character-level similarity (Jaccard on bigrams)
    def _bigrams(s: str) -> set[str]:
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) > 1 else {s}

    nb = _bigrams(needle_lower)
    hb = _bigrams(haystack_lower)
    if nb and hb:
        jaccard = len(nb & hb) / len(nb | hb)
        return jaccard * 0.6  # cap at 0.6 for character-only match

    return 0.0


# ── Core AT-SPI functions ─────────────────────────────────────────────────

def _get_desktop():
    """Get the AT-SPI desktop object."""
    if not _init_atspi():
        return None
    return _atspi.get_desktop(0)


def _find_app(app_name: str):
    """Find an application by name in the accessibility tree."""
    desktop = _get_desktop()
    if desktop is None:
        return None

    app_lower = app_name.lower().strip()
    best_app = None
    best_score = 0.0

    count = desktop.get_child_count()
    for i in range(count):
        child = desktop.get_child_at_index(i)
        if child is None:
            continue
        try:
            name = child.get_name() or ""
        except Exception:
            continue

        score = _fuzzy_match(app_lower, name)
        if score > best_score:
            best_score = score
            best_app = child

    if best_score >= 0.5:
        return best_app
    return None


def _accessible_to_dict(node, depth: int = 0, max_depth: int = 6) -> Optional[dict]:
    """Convert an AT-SPI accessible node to a dict with bounds."""
    if node is None or depth > max_depth:
        return None

    try:
        name = node.get_name() or ""
        role = node.get_role_name() or ""

        # Get bounding box
        try:
            component = node.get_component_iface()
            if component:
                ext = component.get_extents(_atspi.CoordType.SCREEN)
                bounds = {
                    "x": ext.x, "y": ext.y,
                    "width": ext.width, "height": ext.height,
                }
            else:
                bounds = None
        except Exception:
            bounds = None

        result: dict[str, Any] = {
            "name": name,
            "role": role,
        }
        if bounds and bounds["width"] > 0 and bounds["height"] > 0:
            result["bounds"] = bounds
            result["center"] = (
                bounds["x"] + bounds["width"] // 2,
                bounds["y"] + bounds["height"] // 2,
            )

        # Get states
        try:
            states = node.get_state_set()
            if states:
                result["visible"] = states.contains(_atspi.StateType.VISIBLE)
                result["enabled"] = states.contains(_atspi.StateType.ENABLED)
                result["focused"] = states.contains(_atspi.StateType.FOCUSED)
            else:
                result["visible"] = True
                result["enabled"] = True
                result["focused"] = False
        except Exception:
            result["visible"] = True
            result["enabled"] = True
            result["focused"] = False

        # Recurse into children
        child_count = node.get_child_count()
        if child_count > 0 and depth < max_depth:
            children = []
            for i in range(min(child_count, 200)):
                child = node.get_child_at_index(i)
                child_dict = _accessible_to_dict(child, depth + 1, max_depth)
                if child_dict:
                    children.append(child_dict)
            if children:
                result["children"] = children

        return result

    except Exception as exc:
        log.debug("Error reading accessible node: %s", exc)
        return None


def get_element_tree(app_name: str) -> Optional[dict]:
    """
    Get the full accessibility tree for an application.
    Returns a JSON-serializable dict with element names, roles, and bounding boxes.
    """
    app = _find_app(app_name)
    if app is None:
        return None
    return _accessible_to_dict(app, max_depth=4)


def find_element(
    app_name: str,
    label: str,
    role_filter: Optional[str] = None,
) -> Optional[dict]:
    """
    Find a specific element by label (fuzzy matched) in an app's accessibility tree.

    Returns dict with keys: name, role, bounds, center (x, y).
    If role_filter is specified, only matches that role (e.g. "push button", "entry").
    """
    app = _find_app(app_name)
    if app is None:
        return None

    best_match: Optional[dict] = None
    best_score = 0.0

    def _search(node, depth: int = 0):
        nonlocal best_match, best_score
        if node is None or depth > 8:
            return

        try:
            name = node.get_name() or ""
            role = node.get_role_name() or ""

            # Check role filter
            if role_filter and role_filter.lower() not in role.lower():
                pass  # still recurse into children
            else:
                score = _fuzzy_match(label, name)
                # Bonus for interactive roles
                _INTERACTIVE_ROLES = {
                    "push button", "toggle button", "check box",
                    "radio button", "entry", "text", "link",
                    "menu item", "combo box", "tab",
                }
                if role.lower() in _INTERACTIVE_ROLES:
                    score += 0.05

                if score > best_score:
                    # Get bounds
                    try:
                        comp = node.get_component_iface()
                        if comp:
                            ext = comp.get_extents(_atspi.CoordType.SCREEN)
                            if ext.width > 0 and ext.height > 0:
                                best_score = score
                                best_match = {
                                    "name": name,
                                    "role": role,
                                    "bounds": {
                                        "x": ext.x, "y": ext.y,
                                        "width": ext.width, "height": ext.height,
                                    },
                                    "center": (
                                        ext.x + ext.width // 2,
                                        ext.y + ext.height // 2,
                                    ),
                                    "score": round(score, 3),
                                }
                    except Exception:
                        pass

            # Recurse
            for i in range(min(node.get_child_count(), 200)):
                _search(node.get_child_at_index(i), depth + 1)

        except Exception:
            pass

    _search(app)
    return best_match if best_score >= 0.4 else None


def list_applications() -> list[str]:
    """List all accessible application names on the desktop."""
    desktop = _get_desktop()
    if desktop is None:
        return []
    apps = []
    for i in range(desktop.get_child_count()):
        child = desktop.get_child_at_index(i)
        if child:
            try:
                name = child.get_name()
                if name:
                    apps.append(name)
            except Exception:
                pass
    return apps


# ── Window focus ──────────────────────────────────────────────────────────

def focus_window(app_name: str) -> bool:
    """
    Focus/raise a window by application name.
    Tries wmctrl first, then xdotool. Works on X11; limited on Wayland.
    """
    display = detect_display_server()

    # Try wmctrl
    try:
        result = subprocess.run(
            ["wmctrl", "-a", app_name],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Try xdotool (X11 only)
    if display == "x11":
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", app_name, "windowactivate"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            pass
        except Exception:
            pass

    # AT-SPI direct activation (works on Wayland too)
    app = _find_app(app_name)
    if app:
        try:
            # Try to find and activate the first window
            for i in range(app.get_child_count()):
                child = app.get_child_at_index(i)
                if child:
                    role = child.get_role_name() or ""
                    if "frame" in role.lower() or "window" in role.lower():
                        comp = child.get_component_iface()
                        if comp:
                            comp.grab_focus()
                            return True
        except Exception:
            pass

    return False


# ── Tool class ─────────────────────────────────────────────────────────────

class AccessibilityTool(BaseTool):
    name = "accessibility"
    description = (
        "Read the OS accessibility tree to locate UI elements in running apps. "
        "Use for: click a button, find a UI element, list windows, focus a window. "
        "Zero GPU cost, instant response (< 50ms)."
    )
    risk_level = "safe"

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "find")
        app = kwargs.get("app", "")
        label = kwargs.get("label", "")
        role = kwargs.get("role")

        if not _init_atspi():
            return ToolResult(
                success=False,
                output="",
                error=(
                    "AT-SPI2 is not available. Install with:\n"
                    "  sudo apt install python3-gi gir1.2-atspi-2.0\n"
                    "and make sure accessibility is enabled in your desktop settings."
                ),
            )

        if action == "list_apps":
            return self._list_apps()
        elif action == "tree":
            return self._get_tree(app)
        elif action == "find":
            return self._find(app, label, role)
        elif action == "focus":
            return self._focus(app)
        elif action == "info":
            return self._info()
        else:
            return ToolResult(
                success=False, output="",
                error=f"Unknown action: {action}. Use: list_apps, tree, find, focus, info",
            )

    def _list_apps(self) -> ToolResult:
        apps = list_applications()
        if not apps:
            return ToolResult(
                success=True,
                output="No accessible applications found.",
                data={"apps": []},
            )
        lines = [f"📋 {len(apps)} accessible applications:"]
        for a in sorted(apps):
            lines.append(f"  • {a}")
        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"apps": apps},
        )

    def _get_tree(self, app_name: str) -> ToolResult:
        if not app_name:
            return ToolResult(
                success=False, output="",
                error="Specify an app name. Use 'list_apps' to see available apps.",
            )

        tree = get_element_tree(app_name)
        if tree is None:
            return ToolResult(
                success=False, output="",
                error=f"Application '{app_name}' not found or no accessibility tree.",
            )

        # Format tree as readable text
        lines = [f"🌳 Element tree for '{app_name}':"]
        self._format_tree_lines(tree, lines, indent=0, max_lines=60)
        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"tree": tree},
        )

    def _format_tree_lines(
        self, node: dict, lines: list[str],
        indent: int = 0, max_lines: int = 60,
    ) -> None:
        if len(lines) >= max_lines:
            if len(lines) == max_lines:
                lines.append("  ... (truncated)")
            return

        prefix = "  " * indent
        name = node.get("name", "")
        role = node.get("role", "")
        bounds = node.get("bounds")
        center = node.get("center")

        parts = [f"{prefix}[{role}]"]
        if name:
            parts.append(f'"{name}"')
        if center:
            parts.append(f"@ ({center[0]}, {center[1]})")
        elif bounds:
            parts.append(f"@ ({bounds['x']},{bounds['y']} {bounds['width']}x{bounds['height']})")
        lines.append(" ".join(parts))

        for child in node.get("children", []):
            self._format_tree_lines(child, lines, indent + 1, max_lines)

    def _find(self, app_name: str, label: str, role: Optional[str]) -> ToolResult:
        if not app_name or not label:
            return ToolResult(
                success=False, output="",
                error="Specify both 'app' and 'label' to find an element.",
            )

        element = find_element(app_name, label, role_filter=role)
        if element is None:
            return ToolResult(
                success=False, output="",
                error=f"Element '{label}' not found in '{app_name}'.",
            )

        center = element["center"]
        return ToolResult(
            success=True,
            output=(
                f"🎯 Found '{element['name']}' ({element['role']}) "
                f"in {app_name}\n"
                f"   Position: ({center[0]}, {center[1]})\n"
                f"   Bounds: {element['bounds']['width']}x{element['bounds']['height']}\n"
                f"   Match score: {element['score']}"
            ),
            data={
                "element": element,
                "x": center[0],
                "y": center[1],
            },
        )

    def _focus(self, app_name: str) -> ToolResult:
        if not app_name:
            return ToolResult(
                success=False, output="",
                error="Specify an app name to focus.",
            )

        if focus_window(app_name):
            return ToolResult(
                success=True,
                output=f"✓ Focused '{app_name}'.",
                data={"focused": app_name},
            )
        return ToolResult(
            success=False, output="",
            error=f"Could not focus '{app_name}'. Window not found or not supported.",
        )

    def _info(self) -> ToolResult:
        """Return accessibility system info."""
        display = detect_display_server()
        apps = list_applications()
        return ToolResult(
            success=True,
            output=(
                f"🖥  Display: {display}\n"
                f"♿ AT-SPI2: available\n"
                f"📋 Accessible apps: {len(apps)}"
            ),
            data={
                "display_server": display,
                "atspi_available": True,
                "app_count": len(apps),
            },
        )


# ── Register ──────────────────────────────────────────────────────────────

try:
    registry.register(AccessibilityTool())
except Exception:
    pass  # AT-SPI deps may not be installed
