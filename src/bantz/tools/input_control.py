"""
Bantz v3 — Input Control Tool (#122)

PyAutoGUI / pynput input simulation — mouse, keyboard, scroll.
Wraps input actions with a safety model matching shell.py's patterns.

Safety Model:
  - safe:        click(x,y), move_to(x,y), scroll
  - moderate:    type_text, double_click, drag
  - destructive: hotkey combos (ctrl+w, alt+f4, ctrl+q, etc.)

Backend Selection:
  - X11:     pyautogui (full mouse/keyboard/screenshot support)
  - Wayland: pynput (mouse via evdev, keyboard via uinput)
  - Fallback: xdotool via subprocess

Architecture:
    Brain: "click Send button in Firefox"
        → accessibility.py → find_element("firefox", "Send") → (1340, 680)
        → input_control.py → click(1340, 680)

Usage:
    from bantz.tools.input_control import InputControlTool
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Any, Optional

from bantz.config import config
from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.input_control")

# ── Backend detection ─────────────────────────────────────────────────────

_backend: Optional[str] = None
_pyautogui = None
_pynput_mouse = None
_pynput_keyboard = None
_pynput_mouse_controller = None
_pynput_keyboard_controller = None


def _detect_backend() -> str:
    """Detect best input backend: pyautogui (X11) or pynput (Wayland)."""
    global _backend
    if _backend is not None:
        return _backend

    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    wayland = session == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))

    if wayland:
        try:
            from pynput import mouse, keyboard  # noqa: F401
            _backend = "pynput"
            log.debug("Input backend: pynput (Wayland)")
            return _backend
        except ImportError:
            pass

    # X11 or fallback
    try:
        import pyautogui  # noqa: F401
        _backend = "pyautogui"
        log.debug("Input backend: pyautogui (X11)")
        return _backend
    except ImportError:
        pass

    # pynput as second fallback
    try:
        from pynput import mouse, keyboard  # noqa: F401
        _backend = "pynput"
        log.debug("Input backend: pynput (fallback)")
        return _backend
    except ImportError:
        pass

    # xdotool as last resort
    try:
        result = subprocess.run(["which", "xdotool"], capture_output=True)
        if result.returncode == 0:
            _backend = "xdotool"
            log.debug("Input backend: xdotool (fallback)")
            return _backend
    except Exception:
        pass

    _backend = "none"
    log.warning("No input backend available")
    return _backend


def _get_pyautogui():
    """Lazy-load pyautogui with FAILSAFE enabled."""
    global _pyautogui
    if _pyautogui is None:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05  # slight pause between actions
        _pyautogui = pyautogui
    return _pyautogui


def _get_pynput():
    """Lazy-load pynput controllers."""
    global _pynput_mouse, _pynput_keyboard
    global _pynput_mouse_controller, _pynput_keyboard_controller
    if _pynput_mouse_controller is None:
        from pynput.mouse import Controller as MouseCtrl
        from pynput.keyboard import Controller as KeyCtrl, Key
        _pynput_mouse = MouseCtrl
        _pynput_keyboard = KeyCtrl
        _pynput_mouse_controller = MouseCtrl()
        _pynput_keyboard_controller = KeyCtrl()
    return _pynput_mouse_controller, _pynput_keyboard_controller


# ── Safety classification ─────────────────────────────────────────────────

# Destructive hotkeys that could close apps, delete data, etc.
DESTRUCTIVE_HOTKEYS: frozenset[tuple[str, ...]] = frozenset({
    ("ctrl", "w"),      # close tab
    ("ctrl", "q"),      # quit app
    ("alt", "f4"),      # close window
    ("ctrl", "shift", "w"),  # close all tabs
    ("ctrl", "shift", "q"),  # quit alt
    ("super", "d"),     # show desktop / minimize all
    ("ctrl", "z"),      # undo (risky in some contexts)
    ("ctrl", "shift", "delete"),  # clear data
    ("ctrl", "alt", "delete"),    # system
    ("alt", "f2"),      # run dialog
    ("ctrl", "delete"), # delete word
})

MODERATE_HOTKEYS: frozenset[tuple[str, ...]] = frozenset({
    ("ctrl", "s"),      # save
    ("ctrl", "a"),      # select all
    ("ctrl", "c"),      # copy
    ("ctrl", "v"),      # paste
    ("ctrl", "x"),      # cut
    ("ctrl", "f"),      # find
    ("ctrl", "n"),      # new
    ("ctrl", "t"),      # new tab
    ("ctrl", "l"),      # address bar
    ("ctrl", "p"),      # print
    ("alt", "tab"),     # switch window
    ("enter",),         # confirm
    ("tab",),           # tab
    ("escape",),        # escape
})


def classify_action(action: str, **kwargs) -> str:
    """
    Classify an input action's risk level.

    Returns: "safe", "moderate", or "destructive"
    """
    action = action.lower()

    if action in ("click", "move_to", "scroll", "get_position"):
        return "safe"

    if action in ("double_click", "right_click", "drag"):
        return "moderate"

    if action == "type_text":
        text = kwargs.get("text", "")
        if len(text) > 100:
            return "moderate"
        return "moderate"

    if action == "hotkey":
        keys = kwargs.get("keys", ())
        if isinstance(keys, str):
            keys = tuple(k.strip().lower() for k in keys.split("+"))
        else:
            keys = tuple(k.lower() for k in keys)

        if keys in DESTRUCTIVE_HOTKEYS:
            return "destructive"
        if keys in MODERATE_HOTKEYS:
            return "moderate"
        # Unknown hotkey combos → default moderate
        return "moderate"

    return "moderate"


# ── Action logging ────────────────────────────────────────────────────────

_action_log: list[dict] = []


def _log_action(action: str, details: dict, risk: str) -> None:
    """Log every input action with timestamp for RL training data."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "action": action,
        "risk": risk,
        **details,
    }
    _action_log.append(entry)
    # Keep last 500 entries in memory
    if len(_action_log) > 500:
        _action_log.pop(0)

    # Also log to memory store if available
    try:
        from bantz.core.memory import memory
        memory.add(
            "system",
            f"[input] {action}: {details}",
        )
    except Exception:
        pass

    log.info("Input action [%s]: %s %s", risk, action, details)


def get_action_log() -> list[dict]:
    """Return the in-memory action log."""
    return list(_action_log)


# ── Core input functions ──────────────────────────────────────────────────

async def click(x: int, y: int) -> dict:
    """Click at (x, y) coordinates."""
    backend = _detect_backend()

    if backend == "pyautogui":
        pag = _get_pyautogui()
        await asyncio.get_event_loop().run_in_executor(None, pag.click, x, y)
    elif backend == "pynput":
        mc, _ = _get_pynput()
        from pynput.mouse import Button
        mc.position = (x, y)
        mc.click(Button.left, 1)
    elif backend == "xdotool":
        await asyncio.create_subprocess_exec(
            "xdotool", "mousemove", str(x), str(y), "click", "1",
        )
    else:
        raise RuntimeError("No input backend available")

    result = {"x": x, "y": y, "button": "left"}
    _log_action("click", result, "safe")
    return result


async def double_click(x: int, y: int) -> dict:
    """Double-click at (x, y)."""
    backend = _detect_backend()

    if backend == "pyautogui":
        pag = _get_pyautogui()
        await asyncio.get_event_loop().run_in_executor(None, pag.doubleClick, x, y)
    elif backend == "pynput":
        mc, _ = _get_pynput()
        from pynput.mouse import Button
        mc.position = (x, y)
        mc.click(Button.left, 2)
    elif backend == "xdotool":
        await asyncio.create_subprocess_exec(
            "xdotool", "mousemove", str(x), str(y),
            "click", "--repeat", "2", "--delay", "50", "1",
        )
    else:
        raise RuntimeError("No input backend available")

    result = {"x": x, "y": y, "button": "left", "clicks": 2}
    _log_action("double_click", result, "moderate")
    return result


async def right_click(x: int, y: int) -> dict:
    """Right-click at (x, y)."""
    backend = _detect_backend()

    if backend == "pyautogui":
        pag = _get_pyautogui()
        await asyncio.get_event_loop().run_in_executor(None, pag.rightClick, x, y)
    elif backend == "pynput":
        mc, _ = _get_pynput()
        from pynput.mouse import Button
        mc.position = (x, y)
        mc.click(Button.right, 1)
    elif backend == "xdotool":
        await asyncio.create_subprocess_exec(
            "xdotool", "mousemove", str(x), str(y), "click", "3",
        )
    else:
        raise RuntimeError("No input backend available")

    result = {"x": x, "y": y, "button": "right"}
    _log_action("right_click", result, "moderate")
    return result


async def move_to(x: int, y: int) -> dict:
    """Move mouse to (x, y) without clicking."""
    backend = _detect_backend()

    if backend == "pyautogui":
        pag = _get_pyautogui()
        await asyncio.get_event_loop().run_in_executor(None, pag.moveTo, x, y)
    elif backend == "pynput":
        mc, _ = _get_pynput()
        mc.position = (x, y)
    elif backend == "xdotool":
        await asyncio.create_subprocess_exec("xdotool", "mousemove", str(x), str(y))
    else:
        raise RuntimeError("No input backend available")

    result = {"x": x, "y": y}
    _log_action("move_to", result, "safe")
    return result


async def get_position() -> dict:
    """Get current mouse position."""
    backend = _detect_backend()

    if backend == "pyautogui":
        pag = _get_pyautogui()
        pos = pag.position()
        return {"x": pos[0], "y": pos[1]}
    elif backend == "pynput":
        mc, _ = _get_pynput()
        pos = mc.position
        return {"x": pos[0], "y": pos[1]}
    elif backend == "xdotool":
        proc = await asyncio.create_subprocess_exec(
            "xdotool", "getmouselocation",
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        text = stdout.decode()
        match = re.search(r"x:(\d+)\s+y:(\d+)", text)
        if match:
            return {"x": int(match.group(1)), "y": int(match.group(2))}
        return {"x": 0, "y": 0}

    return {"x": 0, "y": 0}


async def type_text(text: str, interval: float = 0.05) -> dict:
    """Type text with human-like intervals (default 50ms between chars)."""
    backend = _detect_backend()

    if backend == "pyautogui":
        pag = _get_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pag.typewrite(text, interval=interval)
        )
    elif backend == "pynput":
        _, kc = _get_pynput()
        for char in text:
            kc.type(char)
            await asyncio.sleep(interval)
    elif backend == "xdotool":
        await asyncio.create_subprocess_exec(
            "xdotool", "type", "--delay", str(int(interval * 1000)), text,
        )
    else:
        raise RuntimeError("No input backend available")

    result = {"text": text[:50], "length": len(text), "interval_ms": int(interval * 1000)}
    _log_action("type_text", result, "moderate")
    return result


async def scroll(direction: str = "down", amount: int = 3) -> dict:
    """Scroll up/down/left/right by amount."""
    backend = _detect_backend()
    direction = direction.lower()

    if backend == "pyautogui":
        pag = _get_pyautogui()
        clicks = amount if direction == "up" else -amount
        if direction in ("left", "right"):
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pag.hscroll(amount if direction == "right" else -amount)
            )
        else:
            await asyncio.get_event_loop().run_in_executor(None, pag.scroll, clicks)
    elif backend == "pynput":
        mc, _ = _get_pynput()
        if direction in ("up", "down"):
            dy = amount if direction == "up" else -amount
            mc.scroll(0, dy)
        else:
            dx = amount if direction == "right" else -amount
            mc.scroll(dx, 0)
    elif backend == "xdotool":
        button = "4" if direction == "up" else "5"
        for _ in range(amount):
            await asyncio.create_subprocess_exec(
                "xdotool", "click", button,
            )
    else:
        raise RuntimeError("No input backend available")

    result = {"direction": direction, "amount": amount}
    _log_action("scroll", result, "safe")
    return result


async def drag(from_x: int, from_y: int, to_x: int, to_y: int, duration: float = 0.5) -> dict:
    """Drag from (from_x, from_y) to (to_x, to_y)."""
    backend = _detect_backend()

    if backend == "pyautogui":
        pag = _get_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: (pag.moveTo(from_x, from_y), pag.drag(to_x - from_x, to_y - from_y, duration=duration))
        )
    elif backend == "pynput":
        mc, _ = _get_pynput()
        from pynput.mouse import Button
        mc.position = (from_x, from_y)
        mc.press(Button.left)
        steps = max(10, int(duration * 60))
        dx = (to_x - from_x) / steps
        dy = (to_y - from_y) / steps
        for i in range(steps):
            mc.position = (int(from_x + dx * (i + 1)), int(from_y + dy * (i + 1)))
            await asyncio.sleep(duration / steps)
        mc.release(Button.left)
    elif backend == "xdotool":
        await asyncio.create_subprocess_exec(
            "xdotool", "mousemove", str(from_x), str(from_y),
            "mousedown", "1",
        )
        await asyncio.sleep(0.05)
        await asyncio.create_subprocess_exec(
            "xdotool", "mousemove", "--sync", str(to_x), str(to_y),
            "mouseup", "1",
        )
    else:
        raise RuntimeError("No input backend available")

    result = {"from": (from_x, from_y), "to": (to_x, to_y), "duration_ms": int(duration * 1000)}
    _log_action("drag", result, "moderate")
    return result


async def hotkey(*keys: str) -> dict:
    """
    Press a keyboard shortcut.
    Examples: hotkey("ctrl", "s"), hotkey("alt", "f4")
    """
    backend = _detect_backend()
    key_list = [k.lower().strip() for k in keys]

    if backend == "pyautogui":
        pag = _get_pyautogui()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pag.hotkey(*key_list)
        )
    elif backend == "pynput":
        _, kc = _get_pynput()
        from pynput.keyboard import Key as PKey

        _KEY_MAP = {
            "ctrl": PKey.ctrl, "control": PKey.ctrl,
            "alt": PKey.alt, "shift": PKey.shift,
            "super": PKey.cmd, "win": PKey.cmd, "meta": PKey.cmd,
            "tab": PKey.tab, "enter": PKey.enter, "return": PKey.enter,
            "escape": PKey.esc, "esc": PKey.esc,
            "space": PKey.space, "backspace": PKey.backspace,
            "delete": PKey.delete, "del": PKey.delete,
            "home": PKey.home, "end": PKey.end,
            "pageup": PKey.page_up, "page_up": PKey.page_up,
            "pagedown": PKey.page_down, "page_down": PKey.page_down,
            "up": PKey.up, "down": PKey.down,
            "left": PKey.left, "right": PKey.right,
            "f1": PKey.f1, "f2": PKey.f2, "f3": PKey.f3, "f4": PKey.f4,
            "f5": PKey.f5, "f6": PKey.f6, "f7": PKey.f7, "f8": PKey.f8,
            "f9": PKey.f9, "f10": PKey.f10, "f11": PKey.f11, "f12": PKey.f12,
        }

        mapped: list = []
        for k in key_list:
            if k in _KEY_MAP:
                mapped.append(_KEY_MAP[k])
            elif len(k) == 1:
                mapped.append(k)
            else:
                log.warning("Unknown key: %s", k)
                mapped.append(k)

        # Press all modifiers, tap the last key, release
        for mk in mapped[:-1]:
            kc.press(mk)
        if mapped:
            kc.press(mapped[-1])
            kc.release(mapped[-1])
        for mk in reversed(mapped[:-1]):
            kc.release(mk)
    elif backend == "xdotool":
        xdo_keys = "+".join(key_list)
        await asyncio.create_subprocess_exec("xdotool", "key", xdo_keys)
    else:
        raise RuntimeError("No input backend available")

    risk = classify_action("hotkey", keys=key_list)
    result = {"keys": key_list, "combo": "+".join(key_list)}
    _log_action("hotkey", result, risk)
    return result


# ── Tool class ─────────────────────────────────────────────────────────────

class InputControlTool(BaseTool):
    name = "input_control"
    description = (
        "Simulate mouse and keyboard input: click, type, scroll, drag, hotkey. "
        "Use after accessibility tool locates UI element coordinates. "
        "Safety: safe=click/scroll, moderate=type/drag, destructive=hotkey(ctrl+w)."
    )
    risk_level = "destructive"

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "click")
        backend = _detect_backend()

        if backend == "none":
            return ToolResult(
                success=False, output="",
                error=(
                    "No input backend available. Install:\n"
                    "  pip install pyautogui pynput\n"
                    "or: sudo apt install xdotool"
                ),
            )

        # Check if input control is enabled
        if not config.input_control_enabled:
            return ToolResult(
                success=False, output="",
                error="Input control is disabled. Set BANTZ_INPUT_CONTROL_ENABLED=true.",
            )

        try:
            if action == "click":
                return await self._click(kwargs)
            elif action == "double_click":
                return await self._double_click(kwargs)
            elif action == "right_click":
                return await self._right_click(kwargs)
            elif action == "move_to":
                return await self._move_to(kwargs)
            elif action == "type_text":
                return await self._type_text(kwargs)
            elif action == "scroll":
                return await self._scroll(kwargs)
            elif action == "drag":
                return await self._drag(kwargs)
            elif action == "hotkey":
                return await self._hotkey(kwargs)
            elif action == "get_position":
                return await self._get_position()
            elif action == "action_log":
                return self._action_log_result()
            else:
                return ToolResult(
                    success=False, output="",
                    error=f"Unknown action: {action}. Use: click, double_click, right_click, "
                          f"move_to, type_text, scroll, drag, hotkey, get_position, action_log",
                )
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Input error: {exc}")

    async def _click(self, kwargs: dict) -> ToolResult:
        x = int(kwargs.get("x", 0))
        y = int(kwargs.get("y", 0))
        if x == 0 and y == 0:
            return ToolResult(success=False, output="", error="Specify x and y coordinates.")
        result = await click(x, y)
        return ToolResult(
            success=True,
            output=f"🖱  Clicked at ({x}, {y})",
            data=result,
        )

    async def _double_click(self, kwargs: dict) -> ToolResult:
        x = int(kwargs.get("x", 0))
        y = int(kwargs.get("y", 0))
        if x == 0 and y == 0:
            return ToolResult(success=False, output="", error="Specify x and y coordinates.")
        result = await double_click(x, y)
        return ToolResult(
            success=True,
            output=f"🖱  Double-clicked at ({x}, {y})",
            data=result,
        )

    async def _right_click(self, kwargs: dict) -> ToolResult:
        x = int(kwargs.get("x", 0))
        y = int(kwargs.get("y", 0))
        if x == 0 and y == 0:
            return ToolResult(success=False, output="", error="Specify x and y coordinates.")
        result = await right_click(x, y)
        return ToolResult(
            success=True,
            output=f"🖱  Right-clicked at ({x}, {y})",
            data=result,
        )

    async def _move_to(self, kwargs: dict) -> ToolResult:
        x = int(kwargs.get("x", 0))
        y = int(kwargs.get("y", 0))
        result = await move_to(x, y)
        return ToolResult(
            success=True,
            output=f"🖱  Moved to ({x}, {y})",
            data=result,
        )

    async def _type_text(self, kwargs: dict) -> ToolResult:
        text = kwargs.get("text", "")
        interval = float(kwargs.get("interval", 0.05))
        if not text:
            return ToolResult(success=False, output="", error="Specify text to type.")

        # Safety preview for confirm_input_destructive
        if config.input_confirm_destructive and len(text) > 50:
            preview = text[:50] + "..."
            # Brain handles confirmation; we just note it
            log.info("Typing %d chars (preview: %s)", len(text), preview)

        result = await type_text(text, interval)
        display = text[:80] + "..." if len(text) > 80 else text
        return ToolResult(
            success=True,
            output=f"⌨  Typed: \"{display}\" ({len(text)} chars, {int(interval*1000)}ms interval)",
            data=result,
        )

    async def _scroll(self, kwargs: dict) -> ToolResult:
        direction = kwargs.get("direction", "down")
        amount = int(kwargs.get("amount", 3))
        result = await scroll(direction, amount)
        return ToolResult(
            success=True,
            output=f"🖱  Scrolled {direction} × {amount}",
            data=result,
        )

    async def _drag(self, kwargs: dict) -> ToolResult:
        from_x = int(kwargs.get("from_x", 0))
        from_y = int(kwargs.get("from_y", 0))
        to_x = int(kwargs.get("to_x", 0))
        to_y = int(kwargs.get("to_y", 0))
        duration = float(kwargs.get("duration", 0.5))
        result = await drag(from_x, from_y, to_x, to_y, duration)
        return ToolResult(
            success=True,
            output=f"🖱  Dragged ({from_x},{from_y}) → ({to_x},{to_y})",
            data=result,
        )

    async def _hotkey(self, kwargs: dict) -> ToolResult:
        keys_raw = kwargs.get("keys", "")
        if isinstance(keys_raw, str):
            key_list = [k.strip() for k in keys_raw.split("+") if k.strip()]
        elif isinstance(keys_raw, (list, tuple)):
            key_list = list(keys_raw)
        else:
            return ToolResult(success=False, output="", error="Specify keys (e.g., 'ctrl+s').")

        if not key_list:
            return ToolResult(success=False, output="", error="Specify keys (e.g., 'ctrl+s').")

        risk = classify_action("hotkey", keys=key_list)
        combo = "+".join(key_list)

        # Destructive hotkeys require confirmation
        if risk == "destructive" and config.input_confirm_destructive:
            return ToolResult(
                success=True,
                output=f"⚠️  Destructive hotkey: {combo}\n   This may close windows or delete data.\n   Confirm to proceed.",
                data={"keys": key_list, "combo": combo, "risk": risk, "_needs_confirm": True},
            )

        result = await hotkey(*key_list)
        return ToolResult(
            success=True,
            output=f"⌨  Pressed: {combo}",
            data=result,
        )

    async def _get_position(self) -> ToolResult:
        pos = await get_position()
        return ToolResult(
            success=True,
            output=f"🖱  Mouse at ({pos['x']}, {pos['y']})",
            data=pos,
        )

    def _action_log_result(self) -> ToolResult:
        log_entries = get_action_log()
        if not log_entries:
            return ToolResult(success=True, output="No input actions logged yet.", data={"log": []})

        lines = [f"📋 Last {min(len(log_entries), 20)} input actions:"]
        for entry in log_entries[-20:]:
            lines.append(
                f"  [{entry['risk']}] {entry['action']} "
                f"@ {entry['timestamp']}"
            )
        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"log": log_entries[-20:]},
        )


# ── Register ──────────────────────────────────────────────────────────────

try:
    registry.register(InputControlTool())
except Exception:
    pass  # pyautogui/pynput may not be installed
