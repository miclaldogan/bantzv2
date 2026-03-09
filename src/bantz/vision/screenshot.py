"""
Bantz v3 — Screenshot Capture (#120)

Captures the full screen or the active window, with optional ROI
(Region Of Interest) cropping.  Outputs JPEG bytes at configurable
quality to minimise bandwidth when sending to a remote VLM.

Backends (tried in order):
  1. gnome-screenshot / grim (Wayland)
  2. scrot / import (X11)
  3. Pillow ImageGrab (fallback)

Usage:
    from bantz.vision.screenshot import capture, capture_window

    img_bytes = await capture()                     # full screen
    img_bytes = await capture_window("firefox")     # active window
    img_bytes = await capture(roi=(100, 100, 800, 600))  # cropped
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from bantz.config import config

log = logging.getLogger("bantz.vision.screenshot")

# ── Types ──────────────────────────────────────────────────────────────────

ROI = Tuple[int, int, int, int]  # (x, y, width, height)


@dataclass
class Screenshot:
    """Captured screenshot data."""
    data: bytes           # JPEG bytes
    width: int
    height: int
    source: str           # "full", "window", "roi"
    format: str = "jpeg"


# ── Display server detection ──────────────────────────────────────────────

def _detect_display() -> str:
    """Detect X11 or Wayland."""
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


def _which(cmd: str) -> bool:
    """Check if a command exists on PATH."""
    try:
        result = subprocess.run(
            ["which", cmd], capture_output=True, timeout=3,
        )
        return result.returncode == 0
    except Exception:
        return False


# ── JPEG compression ──────────────────────────────────────────────────────

def _raw_to_jpeg(png_bytes: bytes, quality: int = 70) -> tuple[bytes, int, int]:
    """Convert raw PNG bytes to JPEG with given quality.  Returns (jpeg_bytes, w, h)."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes))
        w, h = img.size
        if img.mode == "RGBA":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), w, h
    except ImportError:
        # Without Pillow, return the PNG as-is
        log.warning("Pillow not installed — returning raw PNG instead of JPEG")
        return png_bytes, 0, 0


def _crop_image(image_bytes: bytes, roi: ROI, quality: int = 70) -> tuple[bytes, int, int]:
    """Crop an image to the given region of interest.  Returns (jpeg_bytes, w, h)."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        x, y, w, h = roi
        # PIL crop takes (left, upper, right, lower)
        cropped = img.crop((x, y, x + w, y + h))
        if cropped.mode == "RGBA":
            cropped = cropped.convert("RGB")
        cw, ch = cropped.size
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), cw, ch
    except ImportError:
        log.warning("Pillow not installed — cannot crop, returning full image")
        return image_bytes, 0, 0


# ── Screenshot backends ───────────────────────────────────────────────────

async def _capture_grim() -> Optional[bytes]:
    """Wayland: grim → stdout PNG."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "grim", "-t", "png", "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode == 0 and stdout:
            return stdout
    except (FileNotFoundError, asyncio.TimeoutError):
        pass
    return None


async def _capture_gnome_screenshot() -> Optional[bytes]:
    """GNOME screenshot tool → temp file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        proc = await asyncio.create_subprocess_exec(
            "gnome-screenshot", "-f", tmp,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=5)
        p = Path(tmp)
        if p.exists() and p.stat().st_size > 0:
            data = p.read_bytes()
            return data
    except (FileNotFoundError, asyncio.TimeoutError):
        pass
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return None


async def _capture_scrot() -> Optional[bytes]:
    """X11: scrot → temp file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        proc = await asyncio.create_subprocess_exec(
            "scrot", tmp,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=5)
        p = Path(tmp)
        if p.exists() and p.stat().st_size > 0:
            data = p.read_bytes()
            return data
    except (FileNotFoundError, asyncio.TimeoutError):
        pass
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return None


async def _capture_import() -> Optional[bytes]:
    """X11: ImageMagick import → stdout PNG."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "import", "-window", "root", "png:-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode == 0 and stdout:
            return stdout
    except (FileNotFoundError, asyncio.TimeoutError):
        pass
    return None


async def _capture_pillow() -> Optional[bytes]:
    """Pillow ImageGrab fallback (X11 only)."""
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


async def _capture_window_xdotool(app_name: str) -> Optional[bytes]:
    """X11: capture a specific window by name using xdotool + import."""
    try:
        # Find window ID
        proc = await asyncio.create_subprocess_exec(
            "xdotool", "search", "--name", app_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
        if proc.returncode != 0 or not stdout:
            return None

        # Take the first window ID
        wid = stdout.decode().strip().split("\n")[0]
        if not wid:
            return None

        proc2 = await asyncio.create_subprocess_exec(
            "import", "-window", wid, "png:-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout2, _ = await asyncio.wait_for(proc2.communicate(), timeout=5)
        if proc2.returncode == 0 and stdout2:
            return stdout2
    except (FileNotFoundError, asyncio.TimeoutError):
        pass
    return None


# ── Public API ────────────────────────────────────────────────────────────

async def capture_raw() -> Optional[bytes]:
    """
    Capture a full-screen screenshot as raw PNG bytes.
    Tries multiple backends in order of preference.
    """
    display = _detect_display()

    if display == "wayland":
        backends = [_capture_grim, _capture_gnome_screenshot, _capture_pillow]
    elif display == "x11":
        backends = [_capture_scrot, _capture_import, _capture_gnome_screenshot, _capture_pillow]
    else:
        backends = [_capture_pillow]

    for backend in backends:
        raw = await backend()
        if raw:
            log.debug("Screenshot captured via %s (%d bytes)", backend.__name__, len(raw))
            return raw

    log.warning("All screenshot backends failed")
    return None


async def capture(roi: Optional[ROI] = None) -> Optional[Screenshot]:
    """
    Capture a full-screen screenshot.

    Args:
        roi: Optional (x, y, width, height) to crop the screenshot to.

    Returns:
        Screenshot object with JPEG data, or None if capture failed.
    """
    quality = config.screenshot_quality

    raw = await capture_raw()
    if raw is None:
        return None

    if roi:
        data, w, h = _crop_image(raw, roi, quality=quality)
        return Screenshot(data=data, width=w, height=h, source="roi")

    data, w, h = _raw_to_jpeg(raw, quality=quality)
    return Screenshot(data=data, width=w, height=h, source="full")


async def capture_window(app_name: str) -> Optional[Screenshot]:
    """
    Capture a specific application window.

    Falls back to full-screen if window-specific capture fails.
    """
    quality = config.screenshot_quality
    display = _detect_display()

    # Try window-specific capture on X11
    if display == "x11":
        raw = await _capture_window_xdotool(app_name)
        if raw:
            data, w, h = _raw_to_jpeg(raw, quality=quality)
            return Screenshot(data=data, width=w, height=h, source="window")

    # Fallback to full-screen
    log.debug("Window capture failed for '%s', falling back to full screen", app_name)
    return await capture()


async def capture_base64(roi: Optional[ROI] = None) -> Optional[str]:
    """
    Capture screenshot and return as base64-encoded JPEG string.

    This is the format expected by the VLM endpoint.
    """
    import base64
    shot = await capture(roi=roi)
    if shot is None:
        return None
    return base64.b64encode(shot.data).decode("ascii")


async def capture_window_base64(app_name: str) -> Optional[str]:
    """Capture a window and return as base64 JPEG."""
    import base64
    shot = await capture_window(app_name)
    if shot is None:
        return None
    return base64.b64encode(shot.data).decode("ascii")
