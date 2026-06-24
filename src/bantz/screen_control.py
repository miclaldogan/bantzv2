"""
Bantz v2 — Reliable Screen Control

Replaces ad-hoc pyautogui/xdotool calls with one verified interaction layer:

  - Backend auto-detection: ydotool on Wayland (needs ydotoold), xdotool on
    X11/XWayland, pyautogui as last resort.
  - Screenshots reuse the existing capture ladder in
    ``bantz.vision.screenshot`` (grim → gnome-screenshot → scrot → import
    → pillow) instead of duplicating it.
  - Every mutating action can be verified: ``click`` compares before/after
    screenshots and retries once with a small offset if the screen did not
    change; ``wait_for_stable`` polls until the pixel diff settles.
  - All methods return ``bool`` success and log failures — never silent.

Usage:
    from bantz.screen_control import ScreenControl
    screen = ScreenControl()
    await screen.click(640, 400)
    await screen.type_text("hello")
    await screen.key("ctrl+t")
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import subprocess
from typing import Awaitable, Callable, Optional

log = logging.getLogger("bantz.screen_control")

# Vision locator: async callable (description, png_bytes) → (x, y) | None.
# Injected by the caller (VisionExecutor) to avoid a circular import.
VisionLocator = Callable[[str, bytes], Awaitable[Optional[tuple[int, int]]]]

_SETTLE_DELAY_DEFAULT = 0.5   # post-action delay before the next screenshot
_SETTLE_DELAY_LAUNCH = 2.0    # after app launches / window changes


def _detect_backend() -> str:
    """Pick the input backend for this session.

    Wayland → ydotool (verify the daemon responds), else xdotool (covers
    X11 and XWayland windows), else pyautogui if importable, else "none".
    """
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("ydotool"):
        try:
            r = subprocess.run(
                ["ydotool", "mousemove", "--", "0", "0"],
                capture_output=True, timeout=2,
            )
            if r.returncode == 0:
                return "ydotool"
            log.warning("ydotool present but ydotoold not reachable: %s",
                        r.stderr.decode(errors="replace").strip())
        except Exception as exc:  # pragma: no cover - environment dependent
            log.warning("ydotool probe failed: %s", exc)
    if shutil.which("xdotool"):
        return "xdotool"
    try:  # pragma: no cover - optional dependency
        import pyautogui  # noqa: F401
        return "pyautogui"
    except Exception:
        return "none"


def _image_diff(png_a: bytes, png_b: bytes) -> float:
    """Mean absolute pixel difference in [0, 1] between two PNG buffers.

    Downscales to 128px wide grayscale first — we only need "did the screen
    change", not where.
    """
    from PIL import Image, ImageChops, ImageStat

    def _load(b: bytes) -> "Image.Image":
        img = Image.open(io.BytesIO(b)).convert("L")
        w, h = img.size
        return img.resize((128, max(1, int(h * 128 / w))))

    a, b = _load(png_a), _load(png_b)
    if a.size != b.size:
        return 1.0
    stat = ImageStat.Stat(ImageChops.difference(a, b))
    return stat.mean[0] / 255.0


class ScreenControl:
    """Verified mouse/keyboard/screenshot interaction for the desktop.

    Parameters
    ----------
    vision_locator:
        Optional async ``(description, screenshot_png) → (x, y) | None``
        used by :meth:`find_element` when an action specifies a target by
        description instead of coordinates.
    """

    def __init__(self, vision_locator: VisionLocator | None = None) -> None:
        self.backend = _detect_backend()
        self._vision_locator = vision_locator
        if self.backend == "none":
            log.error("No input backend available (ydotool/xdotool/pyautogui)")
        else:
            log.info("ScreenControl backend: %s", self.backend)

    # ── screenshot ──────────────────────────────────────────────────────

    async def screenshot(self) -> bytes | None:
        """Full-screen PNG bytes via the bantz.vision capture ladder."""
        from bantz.vision.screenshot import capture_raw
        try:
            return await capture_raw()
        except Exception as exc:
            log.error("screenshot failed: %s", exc)
            return None

    # ── low-level runner ────────────────────────────────────────────────

    async def _run(self, *argv: str, timeout: float = 5.0) -> bool:
        """Run one backend command; True on rc=0, logged failure otherwise."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, err = await asyncio.wait_for(proc.communicate(), timeout)
            if proc.returncode != 0:
                log.warning("%s failed (rc=%s): %s", argv[0], proc.returncode,
                            (err or b"").decode(errors="replace").strip())
                return False
            return True
        except Exception as exc:
            log.warning("%s failed: %s", argv[0], exc)
            return False

    # ── actions ─────────────────────────────────────────────────────────

    async def click(
        self, x: int, y: int, button: str = "left", *, verify: bool = True,
    ) -> bool:
        """Move to (x, y) and click. With ``verify``, confirm the screen
        changed; if it did not, retry once 4px lower-right (hit-slop for
        slightly-off vision coordinates)."""
        before = await self.screenshot() if verify else None
        if not await self._click_raw(x, y, button):
            return False
        if before is None:
            return True
        await asyncio.sleep(_SETTLE_DELAY_DEFAULT)
        after = await self.screenshot()
        if after is None or _image_diff(before, after) > 0.005:
            return True
        log.info("click(%d,%d) produced no screen change — retrying offset", x, y)
        if not await self._click_raw(x + 4, y + 4, button):
            return False
        await asyncio.sleep(_SETTLE_DELAY_DEFAULT)
        after2 = await self.screenshot()
        changed = after2 is not None and _image_diff(before, after2) > 0.005
        if not changed:
            log.warning("click(%d,%d) verified unchanged screen after retry", x, y)
        return changed

    async def _click_raw(self, x: int, y: int, button: str) -> bool:
        btn = {"left": "1", "middle": "2", "right": "3"}.get(button, "1")
        if self.backend == "ydotool":
            # ydotool button codes: 0x40=down|0x80=up combined click on btn0
            code = {"left": "0xC0", "middle": "0xC1", "right": "0xC2"}.get(button, "0xC0")
            return (await self._run("ydotool", "mousemove", "--absolute",
                                    "--", str(x), str(y))
                    and await self._run("ydotool", "click", code))
        if self.backend == "xdotool":
            return await self._run("xdotool", "mousemove", str(x), str(y),
                                   "click", btn)
        if self.backend == "pyautogui":  # pragma: no cover
            try:
                import pyautogui
                pyautogui.click(x, y, button=button)
                return True
            except Exception as exc:
                log.warning("pyautogui click failed: %s", exc)
                return False
        return False

    async def type_text(self, text: str, interval_ms: int = 50) -> bool:
        """Type *text* into the focused window. Handles unicode via the
        backend's own encoder (xdotool type / ydotool type)."""
        if not text:
            return True
        if self.backend == "ydotool":
            return await self._run("ydotool", "type", "--key-delay",
                                   str(interval_ms), "--", text,
                                   timeout=10 + len(text) * interval_ms / 1000)
        if self.backend == "xdotool":
            return await self._run("xdotool", "type", "--delay",
                                   str(interval_ms), "--", text,
                                   timeout=10 + len(text) * interval_ms / 1000)
        if self.backend == "pyautogui":  # pragma: no cover
            try:
                import pyautogui
                pyautogui.write(text, interval=interval_ms / 1000)
                return True
            except Exception as exc:
                log.warning("pyautogui type failed: %s", exc)
                return False
        return False

    async def key(self, keycombo: str) -> bool:
        """Press a key or combo. Examples: 'ctrl+t', 'Return', 'Escape'."""
        if self.backend == "ydotool":
            return await self._run("ydotool", "key", keycombo)
        if self.backend == "xdotool":
            return await self._run("xdotool", "key", keycombo)
        if self.backend == "pyautogui":  # pragma: no cover
            try:
                import pyautogui
                pyautogui.hotkey(*keycombo.lower().split("+"))
                return True
            except Exception as exc:
                log.warning("pyautogui key failed: %s", exc)
                return False
        return False

    async def scroll(self, x: int, y: int, direction: str = "down",
                     clicks: int = 3) -> bool:
        """Scroll at (x, y). xdotool buttons: 4=up, 5=down."""
        if self.backend == "xdotool":
            btn = "4" if direction == "up" else "5"
            ok = await self._run("xdotool", "mousemove", str(x), str(y))
            for _ in range(clicks):
                ok = ok and await self._run("xdotool", "click", btn)
            return ok
        if self.backend == "ydotool":
            # ydotool has no scroll subcommand — arrow-key taps work in
            # most scrollable views (Linux event codes: 103=Up, 108=Down).
            code = "103" if direction == "up" else "108"
            ok = await self._run("ydotool", "mousemove", "--absolute",
                                 "--", str(x), str(y))
            for _ in range(clicks):
                ok = ok and await self._run("ydotool", "key",
                                            f"{code}:1", f"{code}:0")
            return ok
        if self.backend == "pyautogui":  # pragma: no cover
            try:
                import pyautogui
                pyautogui.scroll(clicks if direction == "up" else -clicks, x, y)
                return True
            except Exception as exc:
                log.warning("pyautogui scroll failed: %s", exc)
                return False
        return False

    # ── vision-assisted lookup ──────────────────────────────────────────

    async def find_element(
        self, description: str, screenshot: bytes | None = None,
    ) -> tuple[int, int] | None:
        """Locate a UI element by natural-language description.

        Delegates to the injected vision locator; returns center (x, y) or
        None. Used when a step gives ``target_description`` but no coords.
        """
        if self._vision_locator is None:
            log.warning("find_element(%r): no vision locator injected", description)
            return None
        shot = screenshot or await self.screenshot()
        if shot is None:
            return None
        try:
            return await self._vision_locator(description, shot)
        except Exception as exc:
            log.warning("find_element(%r) failed: %s", description, exc)
            return None

    # ── focused window (ground truth for the vision loop) ───────────────

    async def active_window(self) -> str:
        """Class/title of the focused window, or "" if unknown.

        Hyprland: ``hyprctl activewindow -j``; X11 fallback: xdotool.
        The vision model guesses focus from pixels and gets it wrong —
        this is the compositor's authoritative answer.
        """
        import json as _json
        if shutil.which("hyprctl") or os.path.exists("/usr/bin/hyprctl"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    shutil.which("hyprctl") or "/usr/bin/hyprctl",
                    "activewindow", "-j",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                out, _ = await asyncio.wait_for(proc.communicate(), 3)
                data = _json.loads(out or b"{}")
                cls = data.get("class", "")
                title = (data.get("title", "") or "")[:60]
                if cls or title:
                    return f"{cls}: {title}"
            except Exception:
                pass
        xdo = shutil.which("xdotool") or "/usr/sbin/xdotool"
        if os.path.exists(xdo):
            try:
                proc = await asyncio.create_subprocess_exec(
                    xdo, "getactivewindow", "getwindowname",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                out, _ = await asyncio.wait_for(proc.communicate(), 3)
                return (out or b"").decode(errors="replace").strip()[:80]
            except Exception:
                pass
        return ""

    async def focus_window(self, app: str) -> bool:
        """Focus a window by application class (launches nothing).

        Window management belongs to the compositor, not the vision model
        — a VLM pressing "super" on Hyprland is a bare-modifier no-op and
        deadlocks the loop. Hyprland: ``hyprctl dispatch focuswindow``;
        X11 fallback: xdotool search+activate.
        """
        hyprctl = shutil.which("hyprctl") or "/usr/bin/hyprctl"
        if os.path.exists(hyprctl):
            ok = await self._run(hyprctl, "dispatch", "focuswindow",
                                 f"class:^({app}).*")
            if ok:
                await asyncio.sleep(0.3)
                active = await self.active_window()
                if app.lower() in active.lower():
                    return True
        xdo = shutil.which("xdotool") or "/usr/sbin/xdotool"
        if os.path.exists(xdo):
            return await self._run(xdo, "search", "--onlyvisible",
                                   "--class", app, "windowactivate")
        return False

    # ── settle detection ────────────────────────────────────────────────

    async def wait_for_stable(
        self, timeout: float = 3.0, threshold: float = 0.02,
        poll: float = 0.3,
    ) -> bool:
        """Poll screenshots until consecutive frames differ by less than
        *threshold* (screen has settled) or *timeout* elapses."""
        prev = await self.screenshot()
        if prev is None:
            return False
        waited = 0.0
        while waited < timeout:
            await asyncio.sleep(poll)
            waited += poll
            cur = await self.screenshot()
            if cur is None:
                return False
            if _image_diff(prev, cur) < threshold:
                return True
            prev = cur
        log.debug("wait_for_stable: screen still changing after %.1fs", timeout)
        return False
