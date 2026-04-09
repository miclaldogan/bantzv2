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

    When action="analyze", captures the screen AND runs VLM analysis,
    returning a text description of what's visible on screen.
    """

    name = "screenshot"
    description = (
        "Capture and DELIVER a screenshot image, or ANALYZE what's on screen. "
        "Use for: 'take a screenshot' (action=capture), "
        "'what do you see on my screen' / 'what's on the screen' / 'describe my screen' (action=analyze). "
        "action=analyze captures a screenshot AND describes the screen content using vision AI. "
        "action=capture just takes and delivers the photo. Default: capture."
    )
    risk_level = "safe"

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Capture screenshot, optionally with VLM analysis.

        Keyword args:
            action (str): "capture" (default) or "analyze" — analyze runs VLM.
            app (str):    Optional application name for window capture.
            region (str): Optional ``x,y,w,h`` for region capture.
            quality (int): JPEG quality 1-95 (default from config).

        Returns:
            ToolResult with data={"screenshot": bytes, "width": int, "height": int}
            When action=analyze, output contains the VLM screen description.
        """
        from bantz.config import config

        action = str(kwargs.get("action", "capture")).strip().lower()

        if not config.telegram_screenshot_enabled and action != "analyze":
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

            # ── Analyze mode: run VLM on the screenshot ──────────────
            if action == "analyze":
                description = await self._vlm_analyze(image_data)
                if description:
                    return ToolResult(
                        success=True,
                        output=description,
                        data={
                            "screenshot": image_data,
                            "width": width,
                            "height": height,
                            "mime_type": "image/jpeg",
                            "screen_description": description,
                        },
                    )
                else:
                    return ToolResult(
                        success=True,
                        output=(
                            "I have captured the screen, ma'am, but my visual apparatus "
                            "could not discern the details. The image is available for your inspection."
                        ),
                        data={
                            "screenshot": image_data,
                            "width": width,
                            "height": height,
                            "mime_type": "image/jpeg",
                        },
                    )

            # ── Standard capture mode ────────────────────────────────
            return ToolResult(
                success=True,
                output=(
                    f"Screenshot captured ({width}×{height}), ready for dispatch."
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

    async def _vlm_analyze(self, image_data: bytes) -> str:
        """Run VLM analysis on screenshot bytes and return description."""
        try:
            import base64
            from bantz.config import config as _cfg
            from bantz.vision.remote_vlm import describe_screen

            b64 = base64.b64encode(image_data).decode()

            result = await describe_screen(b64, timeout=_cfg.vlm_timeout)
            if result.success and result.raw_text:
                return result.raw_text

            # VLM disabled or failed — try Ollama directly
            if not _cfg.vlm_enabled:
                log.debug("VLM disabled, trying Ollama VLM fallback for screen analysis")
                from bantz.vision.remote_vlm import _call_ollama_vlm, DESCRIBE_PROMPT
                result = await _call_ollama_vlm(b64, DESCRIBE_PROMPT, timeout=15)
                if result.success and result.raw_text:
                    return result.raw_text

            return ""
        except Exception as exc:
            log.debug("VLM analysis failed: %s", exc)
            return ""


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
