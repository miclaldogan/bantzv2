"""
Bantz v3 — Browser Control Tool

Visual + keyboard browser automation.

The observe-think-act loop:
  1. open(app)         — launch the app via subprocess
  2. navigate(url)     — Ctrl+L → xdotool type → Enter
  3. wait_for_load     — screenshot loop until page stabilizes
  4. screenshot        — capture what's visible, optionally describe with VLM
  5. find_and_click    — AT-SPI → VLM screenshot analysis → hotkey fallback
  6. type_in_element   — find element + click + xdotool type
  7. new_tab / hotkey / type / scroll

VLM pipeline (find_and_click / type_in_element):
  screenshot → moondream/llava "where is [element]? reply x,y" → click

This is registered as a tool so the planner can chain these steps.

Supported actions:
  open, screenshot, navigate, new_tab, wait_for_load,
  find_and_click, type_in_element, hotkey, type, scroll
"""
from __future__ import annotations

import asyncio
import base64
import logging
import subprocess
import time
from typing import Any

from bantz.config import config
from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tools.browser_control")

# Common app-to-binary mapping
_APP_BINARIES: dict[str, list[str]] = {
    "firefox": ["firefox", "firefox-esr"],
    "chrome": ["google-chrome", "chromium-browser", "chromium"],
    "chromium": ["chromium-browser", "chromium", "google-chrome"],
    "terminal": ["gnome-terminal", "xterm", "konsole", "alacritty"],
    "files": ["nautilus", "thunar", "dolphin"],
    "vscode": ["code"],
    "gedit": ["gedit"],
}


def _focus_window(app: str) -> bool:
    """Focus an app window using xdotool, return True on success."""
    try:
        result = subprocess.run(
            ["xdotool", "search", "--name", app],
            capture_output=True, text=True, timeout=3,
        )
        wids = result.stdout.strip().split()
        if not wids:
            # Try by class name instead
            result2 = subprocess.run(
                ["xdotool", "search", "--class", app],
                capture_output=True, text=True, timeout=3,
            )
            wids = result2.stdout.strip().split()
        if wids:
            # Activate last (most recently opened) matching window
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", wids[-1]],
                capture_output=True, timeout=3,
            )
            log.debug("Focused %s (wid=%s)", app, wids[-1])
            return True
    except Exception as exc:
        log.debug("xdotool focus failed for %s: %s", app, exc)
    return False


def _find_binary(app: str) -> str | None:
    """Find the actual executable for an app name."""
    candidates = _APP_BINARIES.get(app.lower(), [app])
    for binary in candidates:
        try:
            result = subprocess.run(
                ["which", binary], capture_output=True, timeout=2,
            )
            if result.returncode == 0:
                return binary
        except Exception:
            continue
    return None


class BrowserControlTool(BaseTool):
    name = "browser_control"
    description = (
        "Open apps, visually control the browser using screenshots + VLM, and automate GUI tasks. "
        "Uses a see → think → act loop: take screenshot, VLM finds element coordinates, click/type. "
        "Actions: open, screenshot, navigate, new_tab, wait_for_load, "
        "find_and_click, type_in_element, hotkey, type, scroll. "
        "Examples: open firefox → action=open app=firefox; "
        "go to example.com → action=navigate url=https://example.com; "
        "wait for page → action=wait_for_load app=firefox; "
        "click a button → action=find_and_click target='play button' app=firefox; "
        "search on YouTube → action=type_in_element target='search box' text='cats' site='youtube'."
    )
    risk_level = "moderate"

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        if not action:
            # Smart action inference from available kwargs
            url    = str(kwargs.get("url",    "") or "")
            target = str(kwargs.get("target", "") or "")
            text   = str(kwargs.get("text",   "") or "")
            app    = str(kwargs.get("app",    "") or "")
            if url:
                action = "navigate"
            elif target:
                action = "find_and_click"
            elif text:
                action = "type"
            elif app:
                action = "open"
            else:
                return ToolResult(success=False, output="", error="Specify action: open, screenshot, navigate, new_tab, wait_for_load, find_and_click, type_in_element, hotkey, type, scroll")
            log.info("browser_control: inferred action=%s from kwargs", action)

        action = action.lower().strip()

        if not config.input_control_enabled and action not in ("screenshot", "open"):
            return ToolResult(
                success=False, output="",
                error="Input control disabled. Set BANTZ_INPUT_CONTROL_ENABLED=true.",
            )

        try:
            if action == "open":
                return await self._open(kwargs)
            elif action == "screenshot":
                return await self._screenshot(kwargs)
            elif action == "navigate":
                return await self._navigate(kwargs)
            elif action == "new_tab":
                return await self._new_tab(kwargs)
            elif action == "wait_for_load":
                return await self._wait_for_load(kwargs)
            elif action == "find_and_click":
                return await self._find_and_click(kwargs)
            elif action == "type_in_element":
                return await self._type_in_element(kwargs)
            elif action == "hotkey":
                return await self._hotkey(kwargs)
            elif action == "type":
                return await self._type(kwargs)
            elif action == "scroll":
                return await self._scroll(kwargs)
            else:
                return ToolResult(
                    success=False, output="",
                    error=f"Unknown action '{action}'. Use: open, screenshot, navigate, new_tab, wait_for_load, find_and_click, type_in_element, hotkey, type, scroll",
                )
        except Exception as exc:
            log.error("browser_control %s failed: %s", action, exc, exc_info=True)
            return ToolResult(success=False, output="", error=f"Action '{action}' failed: {exc}")

    # ── Actions ───────────────────────────────────────────────────────────

    async def _open(self, kwargs: dict) -> ToolResult:
        """Launch an application."""
        app = kwargs.get("app", "firefox")

        # If already running, just focus it then navigate if url given
        url = kwargs.get("url", "")
        if _focus_window(app):
            if url:
                await self._navigate({"url": url, "app": app, "wait": kwargs.get("nav_wait", 2.5)})
                return ToolResult(success=True, output=f"{app} already open — navigated to {url}.")
            return ToolResult(success=True, output=f"{app} is already open — focused it.")

        binary = _find_binary(app)
        if not binary:
            return ToolResult(
                success=False, output="",
                error=f"Could not find binary for '{app}'. Is it installed?",
            )

        try:
            subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Failed to open {app}: {exc}")

        # Wait for window to appear, then focus
        wait = float(kwargs.get("wait", 3.0))
        await asyncio.sleep(wait)
        _focus_window(app)

        # If a URL was provided, navigate immediately after opening
        url = kwargs.get("url", "")
        if url:
            nav = await self._navigate({
                "url": url, "app": app,
                "wait": kwargs.get("nav_wait", 2.5),
            })
            return ToolResult(
                success=True,
                output=f"Opened {app} and navigated to {url}.",
            )

        return ToolResult(
            success=True,
            output=f"Opened {app} (binary: {binary}) and focused window.",
        )

    async def _screenshot(self, kwargs: dict) -> ToolResult:
        """Take a screenshot and optionally describe what's visible."""
        try:
            from bantz.vision.screenshot import capture_base64, capture
        except ImportError:
            return ToolResult(success=False, output="", error="Screenshot module not available.")

        app = kwargs.get("app", "")
        describe = kwargs.get("describe", False)

        try:
            if app:
                from bantz.vision.screenshot import capture_window_base64
                img_b64 = await capture_window_base64(app)
            else:
                img_b64 = await capture_base64()
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Screenshot failed: {exc}")

        if not img_b64:
            return ToolResult(success=False, output="", error="Screenshot captured no data (display server issue?)")

        raw_bytes = base64.b64decode(img_b64)
        img_bytes = len(raw_bytes)
        result_data = {
            "screenshot_b64": img_b64,
            "screenshot": raw_bytes,   # raw JPEG bytes for Telegram attachment promotion (#189)
            "size_bytes": img_bytes,
            "mime_type": "image/jpeg",
        }

        description = ""
        if describe and config.vlm_enabled:
            try:
                from bantz.vision.remote_vlm import describe_screen
                vlm = await describe_screen(img_b64, timeout=8)
                if vlm.success and vlm.raw_text:
                    description = vlm.raw_text[:800]
            except Exception as exc:
                log.debug("VLM describe failed: %s", exc)

        output_parts = [f"Screenshot captured ({img_bytes:,} bytes)."]
        if description:
            output_parts.append(f"Screen shows: {description}")
        else:
            output_parts.append("Screenshot data saved. Use find_and_click to interact with elements.")

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            data=result_data,
        )

    async def _navigate(self, kwargs: dict) -> ToolResult:
        """Navigate to a URL in the current browser window using Ctrl+L."""
        url = kwargs.get("url", "")
        # Normalise bare site names to full URLs
        _SITE_MAP = {
            "wikipedia": "https://en.wikipedia.org",
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "github": "https://github.com",
            "reddit": "https://www.reddit.com",
            "twitter": "https://twitter.com",
            "instagram": "https://www.instagram.com",
            "stackoverflow": "https://stackoverflow.com",
            "netflix": "https://www.netflix.com",
            "amazon": "https://www.amazon.com",
        }
        if url and not url.startswith("http") and "." not in url:
            url = _SITE_MAP.get(url.lower(), f"https://{url}.com")
            kwargs = {**kwargs, "url": url}
        if not url:
            return ToolResult(success=False, output="", error="Provide url=https://...")

        # Focus the browser window before sending keyboard input
        app = kwargs.get("app", "firefox")
        _focus_window(app)
        await asyncio.sleep(0.3)

        from bantz.tools.input_control import hotkey

        # Ctrl+L — focuses address bar AND selects all existing text automatically
        await hotkey("ctrl", "l")
        await asyncio.sleep(0.5)

        # Use xdotool type --clearmodifiers for reliable URL typing.
        # pyautogui.typewrite() mishandles ':' and '/' on non-US keyboard
        # layouts (e.g. Turkish), turning '//' into '77'.
        proc = await asyncio.create_subprocess_exec(
            "xdotool", "type", "--clearmodifiers", "--delay", "40", url,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        await asyncio.sleep(0.2)

        # Enter to navigate
        await hotkey("enter")
        await asyncio.sleep(float(kwargs.get("wait", 2.0)))

        return ToolResult(
            success=True,
            output=f"Navigated to: {url}",
        )

    async def _wait_for_load(self, kwargs: dict) -> ToolResult:
        """Wait until the browser page finishes loading (screenshot stabilizes)."""
        app = kwargs.get("app", "firefox")
        max_wait = float(kwargs.get("max_wait", 15.0))
        _focus_window(app)
        await asyncio.sleep(0.5)

        from bantz.vision.browser_vision import wait_for_load
        loaded = await wait_for_load(app=app, max_wait=max_wait)
        if loaded:
            return ToolResult(success=True, output="Page finished loading.")
        return ToolResult(success=True, output=f"Page load wait timed out after {max_wait:.0f}s — proceeding.")

    async def _new_tab(self, kwargs: dict) -> ToolResult:
        """Open a new browser tab with Ctrl+T."""
        app = kwargs.get("app", "firefox")
        _focus_window(app)
        await asyncio.sleep(0.2)
        from bantz.tools.input_control import hotkey
        await hotkey("ctrl", "t")
        await asyncio.sleep(0.5)
        return ToolResult(success=True, output="Opened new tab (Ctrl+T).")

    async def _find_and_click(self, kwargs: dict) -> ToolResult:
        """
        Find a UI element and click it.

        Pipeline (in order):
          1. Keyboard shortcuts  — instant for well-known browser elements
          2. AT-SPI              — native accessibility tree (GTK/Firefox chrome)
          3. VLM screenshot      — screenshot → moondream/llava → (x,y) coordinates
                                   (works for ANY visible element, including web content)
        """
        target = kwargs.get("target", "")
        app = kwargs.get("app", "firefox")
        click_action = kwargs.get("click_action", "click")
        wait_load = str(kwargs.get("wait_load", "false")).lower() == "true"

        if not target:
            return ToolResult(success=False, output="", error="Provide target=<UI element description>")

        # Focus the app window
        _focus_window(app)
        await asyncio.sleep(0.3)

        # ── Strategy 1: keyboard shortcuts for well-known browser elements ──
        _HOTKEYS = {
            "new tab": ("ctrl", "t"),
            "new tab button": ("ctrl", "t"),
            "address bar": ("ctrl", "l"),
            "url bar": ("ctrl", "l"),
            "reload": ("f5",), "refresh": ("f5",),
            "back": ("alt", "left"), "forward": ("alt", "right"),
            "zoom in": ("ctrl", "+"), "zoom out": ("ctrl", "-"),
            "fullscreen": ("f11",),
            "downloads": ("ctrl", "j"),
            "bookmarks": ("ctrl", "b"),
            "close tab": ("ctrl", "w"),
            "reopen tab": ("ctrl", "shift", "t"),
        }
        target_lower = target.lower().strip()
        for hint, keys in _HOTKEYS.items():
            if hint in target_lower or target_lower in hint:
                from bantz.tools.input_control import hotkey as do_hotkey
                await do_hotkey(*keys)
                await asyncio.sleep(0.3)
                return ToolResult(
                    success=True,
                    output=f"Used shortcut {'+'.join(keys)} for '{target}'.",
                )

        # ── Strategy 2: AT-SPI accessibility tree ──
        try:
            from bantz.vision.navigator import navigator
            nav = await navigator.navigate_to(app_name=app, element_label=target)
            if nav.found:
                from bantz.tools.input_control import click, double_click, right_click
                cx, cy = nav.center
                if click_action == "double_click":
                    await double_click(cx, cy)
                elif click_action == "right_click":
                    await right_click(cx, cy)
                else:
                    await click(cx, cy)
                await asyncio.sleep(0.3)
                return ToolResult(
                    success=True,
                    output=f"Clicked '{target}' at ({cx}, {cy}) via AT-SPI.",
                    data=nav.to_dict(),
                )
        except Exception as exc:
            log.debug("AT-SPI failed for '%s': %s", target, exc)

        # ── Strategy 3: VLM screenshot analysis ──
        try:
            from bantz.vision.browser_vision import find_and_click_element
            ok, msg = await find_and_click_element(
                element_description=target,
                app=app,
                wait_for_page=wait_load,
                max_load_wait=float(kwargs.get("max_load_wait", 10.0)),
                site_hint=kwargs.get("site", ""),
            )
            if ok:
                return ToolResult(success=True, output=msg)
            log.debug("VLM find failed: %s", msg)
        except Exception as exc:
            log.debug("VLM strategy failed for '%s': %s", target, exc)

        return ToolResult(
            success=False,
            output=f"Could not find '{target}' via AT-SPI or VLM. Is the element visible on screen?",
            error=f"Element '{target}' not found",
        )

    async def _type_in_element(self, kwargs: dict) -> ToolResult:
        """
        Find a UI element, click on it, then type text into it.

        For search boxes on well-known sites the most reliable strategy is
        to navigate directly to the site's search URL (e.g. YouTube results page)
        instead of interacting with the DOM — AT-SPI does not expose web content
        in Firefox by default.

        Pipeline (for search targets):
          1. Detect known-site search URL pattern → navigate directly (most reliable)
          2. AT-SPI element find → click → type
          3. Tab-cycle to first focusable input → type
          4. Ctrl+L → type site-prefixed query (last resort)

        Params:
          target      — description of the element (e.g. "search box", "search input")
          text        — text to type
          app         — browser app name (default: firefox)
          site        — hint about which site we are on (e.g. "youtube", "google")
          press_enter — whether to press Enter after typing (default: True)
          url         — current page URL (used to infer site for search URL construction)
        """
        target = kwargs.get("target", "")
        text = kwargs.get("text", "")
        app = kwargs.get("app", "firefox")
        press_enter = str(kwargs.get("press_enter", "true")).lower() != "false"
        site_hint = kwargs.get("site", "").lower()
        current_url = kwargs.get("url", "").lower()

        if not target:
            return ToolResult(success=False, output="", error="Provide target=<element description>")
        if not text:
            return ToolResult(success=False, output="", error="Provide text=<text to type>")

        # ── Strategy 1: Direct search URL (most reliable for known sites) ──
        # Determine site from hint or current URL
        target_lower = target.lower()
        is_search_target = any(w in target_lower for w in ("search", "query", "find", "arama"))

        if is_search_target:
            import urllib.parse
            encoded = urllib.parse.quote_plus(text)
            search_url = None

            # Parse hostname from current_url to prevent substring-in-URL spoofing (CWE-184).
            # Simple `in` checks on a raw URL string can be bypassed by crafted paths such as
            # "evil.com/page/youtube.com".  Comparing against the parsed hostname is safe.
            _url_host = urllib.parse.urlparse(current_url).hostname or ""

            def _on_site(*domains: str) -> bool:
                return any(d in site_hint for d in domains) or any(
                    _url_host == d or _url_host.endswith("." + d) for d in domains
                )

            if _on_site("youtube", "youtube.com"):
                search_url = f"https://www.youtube.com/results?search_query={encoded}"
            elif _on_site("google", "google.com"):
                search_url = f"https://www.google.com/search?q={encoded}"
            elif _on_site("bing", "bing.com"):
                search_url = f"https://www.bing.com/search?q={encoded}"
            elif _on_site("github", "github.com"):
                search_url = f"https://github.com/search?q={encoded}"
            elif _on_site("duckduckgo", "duckduckgo.com") or "ddg" in site_hint:
                search_url = f"https://duckduckgo.com/?q={encoded}"
            elif _on_site("twitter", "twitter.com", "x.com"):
                search_url = f"https://twitter.com/search?q={encoded}"

            if search_url:
                log.info("type_in_element: using direct search URL %s", search_url)
                nav = await self._navigate({"url": search_url, "app": app, "wait": 2.5})
                display = text[:60] + "..." if len(text) > 60 else text
                return ToolResult(
                    success=True,
                    output=f"Searched for \"{display}\" via direct URL navigation.",
                )

        # ── Strategy 2: AT-SPI element find ──
        _focus_window(app)
        await asyncio.sleep(0.3)

        click_result = await self._find_and_click({"target": target, "app": app})
        clicked = click_result.success

        # ── Strategy 3: Tab-cycle to first focusable input ──
        if not clicked:
            log.debug("AT-SPI find_and_click failed for '%s', trying Tab cycle", target)
            from bantz.tools.input_control import hotkey as do_hotkey
            # Press Escape first to close any overlay, then Tab to reach first input
            await do_hotkey("escape")
            await asyncio.sleep(0.2)
            # Tab through up to 5 elements to find an input
            for _ in range(5):
                await do_hotkey("tab")
                await asyncio.sleep(0.15)

        await asyncio.sleep(0.3)

        # Type text using xdotool (handles all keyboard layouts)
        proc = await asyncio.create_subprocess_exec(
            "xdotool", "type", "--clearmodifiers", "--delay", "40", text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        await asyncio.sleep(0.2)

        if press_enter:
            from bantz.tools.input_control import hotkey as do_hotkey
            await do_hotkey("enter")
            await asyncio.sleep(0.3)

        display = text[:60] + "..." if len(text) > 60 else text
        return ToolResult(
            success=True,
            output=f"Typed \"{display}\" into '{target}'" + (" and pressed Enter." if press_enter else "."),
        )

    async def _hotkey(self, kwargs: dict) -> ToolResult:
        """Send a keyboard shortcut."""
        keys_raw = kwargs.get("keys", "")
        if not keys_raw:
            return ToolResult(success=False, output="", error="Provide keys=ctrl+t or keys=['ctrl','t']")

        from bantz.tools.input_control import hotkey as do_hotkey

        if isinstance(keys_raw, str):
            keys = [k.strip() for k in keys_raw.split("+") if k.strip()]
        elif isinstance(keys_raw, (list, tuple)):
            keys = list(keys_raw)
        else:
            return ToolResult(success=False, output="", error="keys must be string 'ctrl+t' or list ['ctrl','t']")

        await do_hotkey(*keys)
        return ToolResult(success=True, output=f"Sent hotkey: {'+'.join(keys)}")

    async def _type(self, kwargs: dict) -> ToolResult:
        """Type text using xdotool for correct keyboard layout handling."""
        text = kwargs.get("text", "")
        if not text:
            return ToolResult(success=False, output="", error="Provide text=...")

        interval_ms = int(float(kwargs.get("interval", 0.04)) * 1000)
        proc = await asyncio.create_subprocess_exec(
            "xdotool", "type", "--clearmodifiers", "--delay", str(interval_ms), text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        display = text[:60] + "..." if len(text) > 60 else text
        return ToolResult(success=True, output=f"Typed: \"{display}\"")

    async def _scroll(self, kwargs: dict) -> ToolResult:
        """Scroll the page."""
        direction = kwargs.get("direction", "down")
        amount = int(kwargs.get("amount", 3))
        from bantz.tools.input_control import scroll
        await scroll(direction, amount)
        return ToolResult(success=True, output=f"Scrolled {direction} × {amount}")


registry.register(BrowserControlTool())
