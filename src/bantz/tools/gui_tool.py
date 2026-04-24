"""
Bantz v2 — GUITool: pyautogui + xdotool desktop automation bridge (#292)

Low-level input execution layer for desktop GUI automation.
Handles the actual mouse clicks, keystrokes, scrolling, window focus,
and screenshot capture once a higher-level navigator resolves target
coordinates.

Key design decisions (from issue #292 discussion):
 • _dry_run as @property — checked at call time, not import time.
 • pyautogui.FAILSAFE = False — agent mouse moves are intentional,
   corner clicks are legitimate actions, not panic signals.
 • pyautogui.PAUSE = 0 — timing is controlled via DELAY (300 ms).
 • click_image guards against non-existent paths (prevents OpenCV
   C++ assertion crash on hallucinated file paths).
 • focus_window prefers --class over --name with fallback chain.
 • Every action is timestamped in _action_log for replay / debug.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pyautogui
    # ── pyautogui global config ──────────────────────────────────────────────────
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
except (ImportError, KeyError):
    pyautogui = None  # type: ignore

from bantz.tools import BaseTool, ToolResult, registry

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".bantz" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Custom exception ─────────────────────────────────────────────────────────

class GUIToolError(RuntimeError):
    """Raised on recoverable GUI automation failures."""


# ── Core GUITool ─────────────────────────────────────────────────────────────

class GUITool(BaseTool):
    name = "gui"
    description = (
        "Desktop GUI automation: click, type, scroll, focus windows, "
        "screenshot, and template-image matching via pyautogui + xdotool"
    )
    risk_level = "destructive"

    DELAY: float = 0.3  # 300 ms safety delay before every action

    def __init__(self) -> None:
        self._action_log: list[dict[str, Any]] = []

    # ── helpers ───────────────────────────────────────────────────────────

    @property
    def _dry_run(self) -> bool:
        """Checked at call time so toggling the env var takes effect immediately."""
        return os.getenv("BANTZ_DRY_RUN", "0") == "1"

    def _log_action(self, action: str, **params: Any) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **params,
        }
        self._action_log.append(entry)
        logger.debug("gui_action: %s", entry)

    # ── click ─────────────────────────────────────────────────────────────

    def click(self, x: int, y: int) -> None:
        """Click at exact screen coordinates."""
        self._log_action("click", x=x, y=y)
        time.sleep(self.DELAY)
        if self._dry_run:
            logger.info("[DRY_RUN] click(%d, %d)", x, y)
            return
        pyautogui.click(x, y)

    # ── click_image ───────────────────────────────────────────────────────

    def click_image(
        self, template_path: str, confidence: float = 0.9
    ) -> tuple[int, int]:
        """Locate *template_path* on screen and click its centre.

        Guards:
        • FileNotFoundError on disk  → GUIToolError (prevents OpenCV crash)
        • NotImplementedError        → GUIToolError (OpenCV missing)
        • ImageNotFoundException     → GUIToolError (modern pyautogui)
        • None return                → GUIToolError (legacy pyautogui)
        """
        if not Path(template_path).exists():
            raise GUIToolError(
                f"Template image not found on disk: {template_path}"
            )
        self._log_action("click_image", template=template_path, confidence=confidence)
        time.sleep(self.DELAY)
        if self._dry_run:
            logger.info("[DRY_RUN] click_image(%s)", template_path)
            return (0, 0)
        try:
            location = pyautogui.locateOnScreen(
                template_path, confidence=confidence
            )
        except NotImplementedError:
            raise GUIToolError(
                "OpenCV is required for confidence matching. "
                "Run: pip install opencv-python"
            )
        except pyautogui.ImageNotFoundException:
            raise GUIToolError(f"Image not found on screen: {template_path}")
        if location is None:
            raise GUIToolError(f"Image not found on screen: {template_path}")
        center = pyautogui.center(location)
        pyautogui.click(center)
        return (center.x, center.y)

    # ── type ──────────────────────────────────────────────────────────────

    def type_text(self, text: str, interval: float = 0.05) -> None:
        """Type *text* with realistic keystroke delay."""
        self._log_action("type", text=text, interval=interval)
        time.sleep(self.DELAY)
        if self._dry_run:
            logger.info("[DRY_RUN] type(%r)", text)
            return
        pyautogui.typewrite(text, interval=interval)

    # ── focus_window ──────────────────────────────────────────────────────

    def focus_window(self, title: str, wm_class: str | None = None) -> None:
        """Focus a window using xdotool.

        Prefers ``--class`` when *wm_class* is provided, falls back to
        ``--name`` on mismatch.  Handles xdotool absence gracefully.
        """
        self._log_action("focus_window", title=title, wm_class=wm_class)
        if self._dry_run:
            logger.info("[DRY_RUN] focus_window('%s')", title)
            return
        try:
            query = ["--class", wm_class] if wm_class else ["--name", title]
            subprocess.run(
                ["xdotool", "search"] + query + ["windowactivate", "--sync"],
                check=True,
                timeout=5,
            )
        except subprocess.CalledProcessError:
            if wm_class:
                # Fallback: try by name
                try:
                    subprocess.run(
                        [
                            "xdotool", "search", "--name", title,
                            "windowactivate", "--sync",
                        ],
                        check=True,
                        timeout=5,
                    )
                except subprocess.CalledProcessError:
                    raise GUIToolError(
                        f"Window not found by class or name: '{title}'"
                    )
                except subprocess.TimeoutExpired:
                    raise GUIToolError(
                        f"xdotool timed out focusing '{title}' (name fallback)"
                    )
            else:
                raise GUIToolError(f"Window not found: '{title}'")
        except FileNotFoundError:
            logger.warning("xdotool not available — window focus skipped")
        except subprocess.TimeoutExpired:
            raise GUIToolError(f"xdotool timed out focusing '{title}'")

    # ── screenshot ────────────────────────────────────────────────────────

    def screenshot(
        self, region: tuple[int, int, int, int] | None = None
    ) -> str:
        """Capture full screen or *region* → saved to cache, returns path."""
        self._log_action("screenshot", region=region)
        if self._dry_run:
            logger.info("[DRY_RUN] screenshot(region=%s)", region)
            return str(CACHE_DIR / "dry_run_screenshot.png")
        img = pyautogui.screenshot(region=region)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = CACHE_DIR / f"screenshot_{ts}.png"
        img.save(str(path))
        return str(path)

    # ── scroll ────────────────────────────────────────────────────────────

    def scroll(self, x: int, y: int, clicks: int) -> None:
        """Scroll *clicks* at screen coordinates (*x*, *y*)."""
        self._log_action("scroll", x=x, y=y, clicks=clicks)
        time.sleep(self.DELAY)
        if self._dry_run:
            logger.info("[DRY_RUN] scroll(%d, %d, %d)", x, y, clicks)
            return
        pyautogui.moveTo(x, y)
        pyautogui.scroll(clicks)

    # ── action log access ─────────────────────────────────────────────────

    def get_action_log(self) -> list[dict[str, Any]]:
        """Return a copy of the timestamped action log."""
        return list(self._action_log)

    def clear_action_log(self) -> int:
        """Clear the action log and return how many entries were removed."""
        n = len(self._action_log)
        self._action_log.clear()
        return n

    # ── BaseTool interface ────────────────────────────────────────────────

    async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: C901
        action = kwargs.get("action", "click")
        try:
            if action == "click":
                x, y = int(kwargs["x"]), int(kwargs["y"])
                self.click(x, y)
                return ToolResult(success=True, output=f"Clicked ({x}, {y})")

            if action == "click_image":
                template = kwargs["template"]
                confidence = float(kwargs.get("confidence", 0.9))
                cx, cy = self.click_image(template, confidence)
                return ToolResult(
                    success=True,
                    output=f"Clicked image match at ({cx}, {cy})",
                    data={"x": cx, "y": cy},
                )

            if action == "type":
                text = kwargs["text"]
                interval = float(kwargs.get("interval", 0.05))
                self.type_text(text, interval)
                return ToolResult(
                    success=True, output=f"Typed {len(text)} characters"
                )

            if action == "focus_window":
                title = kwargs["title"]
                wm_class = kwargs.get("wm_class")
                self.focus_window(title, wm_class)
                return ToolResult(
                    success=True, output=f"Focused window '{title}'"
                )

            if action == "screenshot":
                region = kwargs.get("region")
                if region is not None:
                    region = tuple(int(v) for v in region)
                path = self.screenshot(region)
                return ToolResult(
                    success=True,
                    output=f"Screenshot saved to {path}",
                    data={"path": path},
                )

            if action == "scroll":
                x, y = int(kwargs["x"]), int(kwargs["y"])
                clicks = int(kwargs["clicks"])
                self.scroll(x, y, clicks)
                return ToolResult(
                    success=True, output=f"Scrolled {clicks} at ({x}, {y})"
                )

            if action == "action_log":
                log = self.get_action_log()
                return ToolResult(
                    success=True,
                    output=f"{len(log)} recorded actions",
                    data={"log": log},
                )

            if action == "clear_log":
                n = self.clear_action_log()
                return ToolResult(
                    success=True, output=f"Cleared {n} action log entries"
                )

            return ToolResult(
                success=False, output="", error=f"Unknown action: {action}"
            )

        except GUIToolError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except KeyError as exc:
            return ToolResult(
                success=False, output="",
                error=f"Missing required parameter: {exc}",
            )


# ── Auto-register ────────────────────────────────────────────────────────────
gui_tool = GUITool()
registry.register(gui_tool)
