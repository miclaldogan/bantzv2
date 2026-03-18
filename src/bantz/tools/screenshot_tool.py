"""
Bantz — ScreenshotTool: capture the luminous glass pane (#189)

Captures a JPEG daguerreotype of the desktop (full screen, specific window,
or a rectangular region) and returns the raw bytes in ToolResult.data so
the Telegram handler can dispatch them to ma'am without touching disk.

Butler says:
  "One moment, ma'am. I am preparing the daguerreotype apparatus…"
"""
from __future__ import annotations

import io
import logging
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tool.screenshot")

# Trigger words that indicate the user wants an image delivered
SCREENSHOT_TRIGGERS: frozenset[str] = frozenset({
    "screenshot", "daguerreotype", "photo", "picture", "snap",
    "show me", "send me", "let me see", "what does it look like",
    "what's on the screen", "capture", "ekran görüntüsü", "ekran",
})


class ScreenshotTool(BaseTool):
    """Capture a daguerreotype (screenshot) of the desktop or a window.

    Returns raw JPEG bytes in ``ToolResult.data["screenshot"]`` so that
    Telegram (and other image-capable interfaces) can send the photo.
    The bytes are *never* written to disk.
    """

    name = "screenshot"
    description = (
        "Capture a daguerreotype of the current desktop screen. "
        "Returns the image bytes for transmission to ma'am. "
        "Optionally capture a specific application window."
    )
    risk_level = "safe"

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Capture screenshot.

        Keyword args:
            app (str):    Optional application name for window capture.
            region (str): Optional ``x,y,w,h`` for region capture.
            quality (int): JPEG quality 1-95 (default from config).

        Returns:
            ToolResult with data={"screenshot": bytes, "width": int, "height": int}
            The butler's output text uses daguerreotype vocabulary.
        """
        from bantz.config import config

        if not config.telegram_screenshot_enabled:
            return ToolResult(
                success=False, output="",
                error="Daguerreotype apparatus is disabled in configuration, ma'am.",
            )

        app = str(kwargs.get("app", "")).strip()
        region_str = str(kwargs.get("region", "")).strip()
        quality = int(kwargs.get("quality", config.telegram_screenshot_quality))

        try:
            from bantz.vision import screenshot as _ss

            if app:
                shot = await _ss.capture_window(app)
            else:
                shot = await _ss.capture()

            if shot is None:
                return ToolResult(
                    success=False, output="",
                    error=(
                        "I regret to inform you, ma'am, that the daguerreotype apparatus "
                        "has malfunctioned. The exposure did not take."
                    ),
                )

            # Optionally crop to region
            image_data = shot.data
            width = getattr(shot, "width", 0)
            height = getattr(shot, "height", 0)

            if region_str:
                try:
                    x, y, w, h = map(int, region_str.split(","))
                    image_data, width, height = _crop_jpeg(image_data, x, y, w, h)
                except Exception as exc:
                    log.debug("Region crop failed: %s", exc)

            # Re-encode at requested quality (vision captures at default quality)
            image_data = _reencode_jpeg(image_data, quality)

            return ToolResult(
                success=True,
                output=(
                    f"I have prepared a daguerreotype of your luminous glass pane, ma'am. "
                    f"The likeness is faithful ({width}×{height}). "
                    f"I must confess the contraption's garish colours do strain "
                    f"one's sensibilities, but it is ready for dispatch."
                ),
                data={
                    "screenshot": image_data,
                    "width": width,
                    "height": height,
                    "mime_type": "image/jpeg",
                },
            )

        except Exception as exc:
            log.warning("ScreenshotTool.execute error: %s", exc)
            return ToolResult(
                success=False, output="",
                error=(
                    f"The daguerreotype apparatus has encountered an impediment, ma'am: {exc}"
                ),
            )


def _reencode_jpeg(data: bytes, quality: int) -> bytes:
    """Re-encode image bytes at the requested JPEG quality."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as img:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=quality)
            return buf.getvalue()
    except Exception:
        return data  # return original if PIL unavailable


def _crop_jpeg(data: bytes, x: int, y: int, w: int, h: int) -> tuple[bytes, int, int]:
    """Crop image bytes to the specified rectangle."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as img:
            cropped = img.crop((x, y, x + w, y + h))
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG")
            return buf.getvalue(), cropped.width, cropped.height
    except Exception:
        return data, w, h


# ── Auto-register ─────────────────────────────────────────────────────────────
registry.register(ScreenshotTool())
