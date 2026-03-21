"""
Bantz v3 — Browser Vision Module

VLM-based browser automation:
  1. wait_for_load    — take screenshots until the page stops changing (loaded)
  2. find_element     — screenshot + VLM prompt → pixel (x, y) coordinates
  3. describe_page    — screenshot + VLM → human-readable description of what's on screen

Pipeline used by browser_control for "find and click" operations:
  screenshot → VLM: "where is [element]? reply x,y" → parse → pyautogui click

The VLM model is moondream (preferred, ~1.7GB) or llava.
If no vision model is available, falls back to None and the caller uses
AT-SPI or hotkey heuristics instead.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import time
from typing import Optional

log = logging.getLogger("bantz.vision.browser_vision")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _screenshot_hash(b64: str) -> str:
    """Quick perceptual hash of a base64 image (MD5 of raw bytes)."""
    raw = base64.b64decode(b64)
    return hashlib.md5(raw).hexdigest()


async def _take_screenshot(app: str = "") -> Optional[str]:
    """Capture a screenshot and return base64 string, or None on failure."""
    try:
        if app:
            from bantz.vision.screenshot import capture_window_base64
            return await capture_window_base64(app)
        else:
            from bantz.vision.screenshot import capture_base64
            return await capture_base64()
    except Exception as exc:
        log.debug("screenshot failed: %s", exc)
        return None


# ── Page Load Waiting ─────────────────────────────────────────────────────────

async def wait_for_load(
    app: str = "firefox",
    max_wait: float = 15.0,
    stable_for: float = 1.5,
    check_interval: float = 0.8,
    min_wait: float = 1.5,
) -> bool:
    """
    Wait until the browser page stops changing (loaded).

    Takes screenshots every `check_interval` seconds.  Two acceptance criteria:

    1. **Fast path** (dynamic SPAs like YouTube): after ``min_wait`` seconds,
       if we have seen ``_FAST_STABLE_CHECKS`` consecutive identical hashes,
       the page is considered loaded.  This fires within ~1.6 s for any brief
       quiet moment — far faster than the full ``stable_for`` window.

    2. **Classic path**: hash stays the same for ``stable_for`` seconds
       (default 1.5 s).  Catches slower static pages that only stabilise once.

    Returns True if the page stabilised within ``max_wait``, False on timeout
    (caller treats timeout as "proceed anyway").
    """
    _FAST_STABLE_CHECKS = 2   # 2 consecutive identical hashes → fast-exit

    start = time.monotonic()
    deadline = start + max_wait
    last_hash = ""
    stable_since = 0.0
    consecutive_stable = 0

    while time.monotonic() < deadline:
        b64 = await _take_screenshot(app)
        if not b64:
            await asyncio.sleep(check_interval)
            continue

        h = _screenshot_hash(b64)
        now = time.monotonic()
        elapsed = now - start

        if h == last_hash:
            consecutive_stable += 1
            if stable_since == 0.0:
                stable_since = now

            # Fast path: 2+ stable checks after min_wait
            if consecutive_stable >= _FAST_STABLE_CHECKS and elapsed >= min_wait:
                log.info(
                    "Page load fast-stable (%d checks) after %.1fs",
                    consecutive_stable, elapsed,
                )
                return True

            # Classic path: continuously stable for stable_for seconds
            if now - stable_since >= stable_for:
                log.info("Page load stable after %.1fs", elapsed)
                return True
        else:
            last_hash = h
            stable_since = 0.0
            consecutive_stable = 0

        await asyncio.sleep(check_interval)

    log.warning("Page load wait timed out after %.1fs", max_wait)
    return False


# ── Window Geometry ──────────────────────────────────────────────────────────

def get_window_bounds(app: str = "firefox") -> dict | None:
    """
    Get position and size of the browser window using wmctrl.
    Returns dict with x, y, w, h keys or None.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["wmctrl", "-lG"], capture_output=True, text=True, timeout=3,
        )
        for line in result.stdout.splitlines():
            if app.lower() in line.lower():
                parts = line.split()
                if len(parts) >= 7:
                    # wmctrl -lG format: wid desktop x y w h hostname title
                    return {
                        "x": int(parts[2]), "y": int(parts[3]),
                        "w": int(parts[4]), "h": int(parts[5]),
                    }
    except Exception as exc:
        log.debug("wmctrl failed: %s", exc)
    return None


def estimate_element_coords(
    element_description: str,
    bounds: dict,
    site_hint: str = "",
) -> tuple[int, int] | None:
    """
    Estimate pixel coordinates of a well-known browser/page element
    based on the browser window bounds and common layout heuristics.

    Works for elements with predictable positions:
    - Browser chrome: address bar, tabs, reload button
    - YouTube: search bar (top center), first video result
    - Google: search box (center), first result
    """
    if not bounds:
        return None

    x0, y0, w, h = bounds["x"], bounds["y"], bounds["w"], bounds["h"]
    el = element_description.lower()

    # Normalize site_hint: if it looks like a URL extract the hostname so that a
    # crafted URL such as "evil.com/path/google.com" cannot spoof site detection
    # via a plain substring check (CWE-184).
    import urllib.parse as _up
    _sh = _up.urlparse(site_hint).hostname or site_hint if "://" in site_hint else site_hint

    # Browser chrome estimates (relative to window top)
    BROWSER_CHROME_H = 75  # approx height of tabs + address bar

    if any(k in el for k in ("address bar", "url bar", "location bar")):
        return (x0 + w // 2, y0 + 50)

    if any(k in el for k in ("tab", "new tab")):
        return (x0 + 100, y0 + 20)

    if any(k in el for k in ("reload", "refresh")):
        return (x0 + 60, y0 + 50)

    # Content area top-left
    content_x = x0
    content_y = y0 + BROWSER_CHROME_H
    content_w = w
    content_h = h - BROWSER_CHROME_H

    # YouTube-specific heuristics
    if "youtube" in _sh or "youtube" in el:
        if any(k in el for k in ("search", "search bar", "search box", "arama")):
            # YouTube search bar: centered, near top of content
            return (content_x + content_w // 2, content_y + 45)
        if any(k in el for k in ("first video", "first result", "top result", "video result")):
            # First YouTube video result: left-ish, ~300px from content top
            return (content_x + content_w // 4, content_y + 300)
        if "play button" in el or "play" in el:
            return (content_x + content_w // 2, content_y + content_h // 2)

    # Google-specific heuristics
    if "google" in _sh or "google" in el:
        if any(k in el for k in ("search", "search bar", "search box")):
            return (content_x + content_w // 2, content_y + content_h // 3)
        if any(k in el for k in ("first result", "top result")):
            return (content_x + content_w // 3, content_y + int(content_h * 0.45))

    # Generic: center of page
    if any(k in el for k in ("search", "input", "text box")):
        return (content_x + content_w // 2, content_y + 100)

    return None


# ── VLM Element Detection ─────────────────────────────────────────────────────

_COORD_PROMPT = (
    "Look at this screenshot carefully. Find the '{element}'. "
    "Reply with ONLY the pixel coordinates of its center as two integers: x,y "
    "(e.g. '320,450'). No other text, no explanation."
)

_REGION_PROMPT = (
    "Look at this screenshot. Is there a '{element}' visible? "
    "If yes, answer with one of: top-left, top-center, top-right, "
    "middle-left, middle-center, middle-right, bottom-left, bottom-center, bottom-right. "
    "If not visible, say 'not found'."
)

_DESCRIBE_PROMPT = (
    "Describe what is visible on this screenshot in 2-3 sentences. "
    "Focus on the main content and any interactive elements (buttons, search bars, forms). "
    "Be specific about positions: top, center, bottom, left, right."
)


async def find_element(
    element_description: str,
    screenshot_b64: Optional[str] = None,
    app: str = "firefox",
    site_hint: str = "",
    timeout: int = 15,
) -> Optional[tuple[int, int]]:
    """
    Find a UI element's pixel coordinates using a layered strategy:

      1. VLM direct coordinate output (llava/bakllava — precise pixel coords)
      2. VLM region detection (moondream — coarse region → approximate coords)
      3. Layout heuristics (no VLM — hardcoded estimates for known sites)

    Args:
        element_description: Natural language description, e.g. "search box"
        screenshot_b64:      Pre-captured screenshot (taken fresh if None)
        app:                 App name for window capture
        site_hint:           Site name hint to improve heuristics (e.g. "youtube")
        timeout:             VLM inference timeout in seconds

    Returns:
        (x, y) pixel coordinates or None if all strategies fail.
    """
    if screenshot_b64 is None:
        screenshot_b64 = await _take_screenshot(app)

    # ── Strategy 1: VLM pixel-coordinate output (llava/bakllava) ──
    if screenshot_b64:
        try:

            # Try models that can return pixel coordinates
            coord_models = ["llava", "bakllava", "llava-llama3"]
            for model in coord_models:
                import httpx
                from bantz.config import config as _cfg
                url = f"{_cfg.ollama_base_url.rstrip('/')}/api/generate"
                prompt = _COORD_PROMPT.format(element=element_description)

                try:
                    async with httpx.AsyncClient(timeout=float(timeout)) as client:
                        resp = await client.post(url, json={
                            "model": model, "prompt": prompt,
                            "images": [screenshot_b64], "stream": False,
                        })
                        if resp.status_code == 404:
                            continue  # model not installed
                        data = resp.json()
                        raw = (data.get("response") or "").strip()
                        if raw:
                            m = re.search(r'(\d{2,4})\s*[,x ]\s*(\d{2,4})', raw)
                            if m:
                                x, y = int(m.group(1)), int(m.group(2))
                                if 10 < x < 4096 and 10 < y < 4096:
                                    log.info("VLM[%s] found '%s' at (%d,%d)", model, element_description, x, y)
                                    return (x, y)
                except Exception:
                    continue

        except Exception as exc:
            log.debug("VLM coord strategy failed: %s", exc)

    # ── Strategy 2: VLM region detection (moondream) → approximate coords ──
    if screenshot_b64:
        try:
            import httpx
            from bantz.config import config as _cfg
            url = f"{_cfg.ollama_base_url.rstrip('/')}/api/generate"
            prompt = _REGION_PROMPT.format(element=element_description)

            async with httpx.AsyncClient(timeout=float(timeout)) as client:
                resp = await client.post(url, json={
                    "model": "moondream", "prompt": prompt,
                    "images": [screenshot_b64], "stream": False,
                })
                if resp.status_code != 404:
                    raw = (resp.json().get("response") or "").strip().lower()
                    if raw and "not found" not in raw:
                        bounds = get_window_bounds(app)
                        if bounds:
                            # Map region name to fraction of content area
                            _REGIONS = {
                                "top-left": (0.2, 0.12), "top-center": (0.5, 0.12),
                                "top-right": (0.8, 0.12),
                                "middle-left": (0.2, 0.5), "middle-center": (0.5, 0.5),
                                "middle-right": (0.8, 0.5),
                                "bottom-left": (0.2, 0.88), "bottom-center": (0.5, 0.88),
                                "bottom-right": (0.8, 0.88),
                            }
                            for region, (fx, fy) in _REGIONS.items():
                                if region in raw:
                                    CHROME_H = 75
                                    x = int(bounds["x"] + bounds["w"] * fx)
                                    y = int(bounds["y"] + CHROME_H + (bounds["h"] - CHROME_H) * fy)
                                    log.info("moondream region '%s' → '%s' → (%d,%d)",
                                             element_description, region, x, y)
                                    return (x, y)
        except Exception as exc:
            log.debug("moondream region strategy failed: %s", exc)

    # ── Strategy 3: Layout heuristics ──
    bounds = get_window_bounds(app)
    if bounds:
        coords = estimate_element_coords(element_description, bounds, site_hint)
        if coords:
            log.info("Heuristic coords for '%s' on '%s': %s", element_description, site_hint, coords)
            return coords

    log.debug("All find_element strategies failed for '%s'", element_description)
    return None


async def describe_page(
    screenshot_b64: Optional[str] = None,
    app: str = "firefox",
    timeout: int = 12,
) -> str:
    """
    Use VLM to describe what's currently on screen.
    Returns a brief description string, or "" if VLM not available.
    """
    if screenshot_b64 is None:
        screenshot_b64 = await _take_screenshot(app)
    if not screenshot_b64:
        return ""

    try:
        from bantz.vision.remote_vlm import _call_ollama_vlm
        result = await _call_ollama_vlm(screenshot_b64, _DESCRIBE_PROMPT, timeout=timeout)
        if result.success and result.raw_text:
            return result.raw_text.strip()[:500]
    except Exception as exc:
        log.debug("VLM describe_page failed: %s", exc)

    return ""


async def find_and_click_element(
    element_description: str,
    app: str = "firefox",
    wait_for_page: bool = True,
    max_load_wait: float = 12.0,
    vlm_timeout: int = 15,
    site_hint: str = "",
) -> tuple[bool, str]:
    """
    Full observe → find → click loop for a browser element.

    1. (Optional) Wait for page to finish loading
    2. Take screenshot
    3. VLM: find element coordinates
    4. Click at those coordinates

    Returns (success, message).
    """
    if wait_for_page:
        loaded = await wait_for_load(app=app, max_wait=max_load_wait)
        if not loaded:
            log.warning("Page didn't fully load within %.1fs — proceeding anyway", max_load_wait)

    # Take screenshot
    screenshot_b64 = await _take_screenshot(app)
    if not screenshot_b64:
        return False, "Could not capture screenshot."

    # VLM: find coordinates
    coords = await find_element(
        element_description,
        screenshot_b64=screenshot_b64,
        app=app,
        site_hint=site_hint,
        timeout=vlm_timeout,
    )

    if not coords:
        return False, f"VLM could not locate '{element_description}' on screen."

    x, y = coords

    # Click — use xdotool to bypass pyautogui FAILSAFE issues
    try:
        proc = await asyncio.create_subprocess_exec(
            "xdotool", "mousemove", str(x), str(y),
            "click", "--clearmodifiers", "1",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        await asyncio.sleep(0.3)
        return True, f"Clicked '{element_description}' at ({x}, {y}) via VLM."
    except Exception as exc:
        return False, f"Click at ({x}, {y}) failed: {exc}"
