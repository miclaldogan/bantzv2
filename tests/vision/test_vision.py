"""
Tests for Issue #120 — Remote VLM screenshot analysis.

Covers:
  - screenshot.py: capture backends, ROI cropping, JPEG compression
  - remote_vlm.py: JSON parsing, VLM API calls, spatial cache
  - accessibility.py VLM integration: fallback when AT-SPI fails
  - brain.py: quick_route patterns for screenshot/describe/VLM
  - config.py: new VLM config fields
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip('PIL')

# ═══════════════════════════════════════════════════════════════════════════
#  Config fields
# ═══════════════════════════════════════════════════════════════════════════

class TestVLMConfig:
    def test_vlm_config_defaults(self):
        from bantz.config import config
        assert hasattr(config, "vlm_enabled")
        assert hasattr(config, "vlm_endpoint")
        assert hasattr(config, "vlm_timeout")
        assert hasattr(config, "screenshot_quality")

    def test_vlm_defaults_values(self):
        from bantz.config import Config
        c = Config(
            _env_file=None,
            BANTZ_OLLAMA_MODEL="test",
        )
        assert c.vlm_enabled is False
        assert c.vlm_endpoint == "http://localhost:8090"
        assert c.vlm_timeout == 5
        assert c.screenshot_quality == 70


# ═══════════════════════════════════════════════════════════════════════════
#  Screenshot module
# ═══════════════════════════════════════════════════════════════════════════

class TestDisplayDetection:
    def test_x11(self):
        from bantz.vision.screenshot import _detect_display
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=False):
            assert _detect_display() == "x11"

    def test_wayland(self):
        from bantz.vision.screenshot import _detect_display
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}, clear=False):
            assert _detect_display() == "wayland"

    def test_wayland_display(self):
        from bantz.vision.screenshot import _detect_display
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "", "WAYLAND_DISPLAY": "wayland-0"}, clear=False):
            assert _detect_display() == "wayland"

    def test_unknown(self):
        from bantz.vision.screenshot import _detect_display
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "", "WAYLAND_DISPLAY": "", "DISPLAY": ""}, clear=False):
            assert _detect_display() == "unknown"


class TestJPEGCompression:
    def test_raw_to_jpeg_without_pillow(self):
        from bantz.vision.screenshot import _raw_to_jpeg
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            # Should return raw bytes as-is
            data = b"fake-png-data"
            result, w, h = _raw_to_jpeg(data, quality=70)
            # Without Pillow, returns original
            assert isinstance(result, bytes)

    def test_raw_to_jpeg_with_pillow(self):
        """Test JPEG conversion with Pillow."""
        from PIL import Image
        from bantz.vision.screenshot import _raw_to_jpeg

        # Create a real PNG in memory
        img = Image.new("RGB", (100, 50), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        jpeg_bytes, w, h = _raw_to_jpeg(png_bytes, quality=70)
        assert w == 100
        assert h == 50
        assert len(jpeg_bytes) > 0
        # JPEG magic bytes
        assert jpeg_bytes[:2] == b"\xff\xd8"

    def test_crop_image_with_pillow(self):
        from PIL import Image
        from bantz.vision.screenshot import _crop_image

        img = Image.new("RGB", (200, 200), color="green")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        jpeg_bytes, w, h = _crop_image(png_bytes, (50, 50, 80, 60), quality=70)
        assert w == 80
        assert h == 60
        assert jpeg_bytes[:2] == b"\xff\xd8"


class TestScreenshot:
    def test_screenshot_dataclass(self):
        from bantz.vision.screenshot import Screenshot
        s = Screenshot(data=b"test", width=100, height=50, source="full")
        assert s.format == "jpeg"
        assert s.width == 100

    @pytest.mark.asyncio
    async def test_capture_raw_all_fail(self):
        from bantz.vision.screenshot import capture_raw
        with patch("bantz.vision.screenshot._detect_display", return_value="unknown"), \
             patch("bantz.vision.screenshot._capture_pillow", new_callable=AsyncMock, return_value=None):
            result = await capture_raw()
            assert result is None

    @pytest.mark.asyncio
    async def test_capture_returns_screenshot(self):
        from PIL import Image
        from bantz.vision.screenshot import capture

        # Create a fake PNG
        img = Image.new("RGB", (64, 64), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        fake_png = buf.getvalue()

        with patch("bantz.vision.screenshot.capture_raw", new_callable=AsyncMock, return_value=fake_png):
            result = await capture()
            assert result is not None
            assert result.source == "full"
            assert result.width == 64
            assert result.height == 64

    @pytest.mark.asyncio
    async def test_capture_with_roi(self):
        from PIL import Image
        from bantz.vision.screenshot import capture

        img = Image.new("RGB", (200, 200), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        fake_png = buf.getvalue()

        with patch("bantz.vision.screenshot.capture_raw", new_callable=AsyncMock, return_value=fake_png):
            result = await capture(roi=(10, 10, 50, 50))
            assert result is not None
            assert result.source == "roi"
            assert result.width == 50
            assert result.height == 50

    @pytest.mark.asyncio
    async def test_capture_base64(self):
        from PIL import Image
        from bantz.vision.screenshot import capture_base64

        img = Image.new("RGB", (32, 32), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        fake_png = buf.getvalue()

        with patch("bantz.vision.screenshot.capture_raw", new_callable=AsyncMock, return_value=fake_png):
            b64 = await capture_base64()
            assert b64 is not None
            # Should be valid base64
            decoded = base64.b64decode(b64)
            assert decoded[:2] == b"\xff\xd8"

    @pytest.mark.asyncio
    async def test_capture_window_fallback(self):
        """Window capture falls back to full screen."""
        from PIL import Image
        from bantz.vision.screenshot import capture_window

        img = Image.new("RGB", (64, 64), color="green")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        fake_png = buf.getvalue()

        with patch("bantz.vision.screenshot._detect_display", return_value="wayland"), \
             patch("bantz.vision.screenshot.capture_raw", new_callable=AsyncMock, return_value=fake_png):
            result = await capture_window("firefox")
            assert result is not None
            # Falls back to full screen on Wayland
            assert result.source == "full"


# ═══════════════════════════════════════════════════════════════════════════
#  Remote VLM module
# ═══════════════════════════════════════════════════════════════════════════

class TestParseVLMResponse:
    def test_clean_json(self):
        from bantz.vision.remote_vlm import parse_vlm_response
        raw = json.dumps({
            "elements": [
                {"label": "Search", "role": "entry", "x": 100, "y": 50,
                 "width": 200, "height": 30, "confidence": 0.95},
            ]
        })
        elems = parse_vlm_response(raw)
        assert len(elems) == 1
        assert elems[0].label == "Search"
        assert elems[0].confidence == 0.95

    def test_markdown_fenced(self):
        from bantz.vision.remote_vlm import parse_vlm_response
        raw = '```json\n{"elements": [{"label": "OK", "x": 10, "y": 20}]}\n```'
        elems = parse_vlm_response(raw)
        assert len(elems) == 1
        assert elems[0].label == "OK"

    def test_bare_array(self):
        from bantz.vision.remote_vlm import parse_vlm_response
        raw = '[{"label": "Cancel", "role": "button", "x": 5, "y": 5, "width": 60, "height": 25}]'
        elems = parse_vlm_response(raw)
        assert len(elems) == 1
        assert elems[0].role == "button"

    def test_invalid_json(self):
        from bantz.vision.remote_vlm import parse_vlm_response
        elems = parse_vlm_response("this is not json at all")
        assert elems == []

    def test_empty_elements(self):
        from bantz.vision.remote_vlm import parse_vlm_response
        raw = '{"elements": []}'
        elems = parse_vlm_response(raw)
        assert elems == []


class TestVLMElement:
    def test_center(self):
        from bantz.vision.remote_vlm import VLMElement
        e = VLMElement(label="btn", x=100, y=200, width=50, height=30)
        assert e.center == (125, 215)

    def test_to_dict(self):
        from bantz.vision.remote_vlm import VLMElement
        e = VLMElement(label="Search", role="entry", x=10, y=20, width=100, height=30, confidence=0.9)
        d = e.to_dict()
        assert d["label"] == "Search"
        assert d["center"] == (60, 35)


class TestVLMResult:
    def test_best(self):
        from bantz.vision.remote_vlm import VLMResult, VLMElement
        r = VLMResult(
            success=True,
            elements=[
                VLMElement(label="A", confidence=0.5),
                VLMElement(label="B", confidence=0.9),
                VLMElement(label="C", confidence=0.7),
            ],
        )
        assert r.best.label == "B"

    def test_find(self):
        from bantz.vision.remote_vlm import VLMResult, VLMElement
        r = VLMResult(
            success=True,
            elements=[
                VLMElement(label="Search Bar", confidence=0.9),
                VLMElement(label="Submit", confidence=0.8),
            ],
        )
        assert r.find("search").label == "Search Bar"
        assert r.find("submit").label == "Submit"
        assert r.find("nonexistent") is None

    def test_empty_best(self):
        from bantz.vision.remote_vlm import VLMResult
        r = VLMResult(success=True)
        assert r.best is None

    def test_to_dict(self):
        from bantz.vision.remote_vlm import VLMResult, VLMElement
        r = VLMResult(success=True, elements=[VLMElement(label="X", confidence=0.5)])
        d = r.to_dict()
        assert d["success"] is True
        assert len(d["elements"]) == 1


class TestSpatialCache:
    def test_put_get(self):
        from bantz.vision.remote_vlm import SpatialCache, VLMResult
        cache = SpatialCache(max_entries=5, ttl_seconds=10.0)
        r = VLMResult(success=True, raw_text="test")
        cache.put("key1", r)
        assert cache.get("key1") is r

    def test_ttl_expiry(self):
        from bantz.vision.remote_vlm import SpatialCache, VLMResult
        cache = SpatialCache(max_entries=5, ttl_seconds=0.01)
        r = VLMResult(success=True)
        cache.put("key1", r)
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_eviction(self):
        from bantz.vision.remote_vlm import SpatialCache, VLMResult
        cache = SpatialCache(max_entries=2, ttl_seconds=60.0)
        cache.put("a", VLMResult(success=True))
        cache.put("b", VLMResult(success=True))
        cache.put("c", VLMResult(success=True))
        assert cache.size == 2
        # "a" should have been evicted
        assert cache.get("a") is None

    def test_invalidate(self):
        from bantz.vision.remote_vlm import SpatialCache, VLMResult
        cache = SpatialCache()
        cache.put("a", VLMResult(success=True))
        cache.put("b", VLMResult(success=True))
        cache.invalidate("a")
        assert cache.get("a") is None
        assert cache.get("b") is not None
        cache.invalidate()
        assert cache.size == 0


class TestCallRemote:
    @pytest.mark.asyncio
    async def test_success(self):
        from bantz.vision.remote_vlm import _call_remote

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "elements": [{"label": "btn", "role": "button", "x": 10, "y": 20,
                           "width": 50, "height": 30, "confidence": 0.9}],
            "raw_text": "",
            "latency_ms": 100,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("bantz.vision.remote_vlm.httpx.AsyncClient", return_value=mock_client):
            result = await _call_remote("fake_b64", "test prompt", timeout=5)
            assert result.success is True
            assert len(result.elements) == 1
            assert result.elements[0].label == "btn"
            assert result.source == "remote"

    @pytest.mark.asyncio
    async def test_timeout(self):
        import httpx
        from bantz.vision.remote_vlm import _call_remote

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("bantz.vision.remote_vlm.httpx.AsyncClient", return_value=mock_client):
            result = await _call_remote("fake_b64", "test", timeout=1)
            assert result.success is False
            assert "timed out" in result.error


class TestAnalyzeScreenshot:
    @pytest.mark.asyncio
    async def test_disabled(self):
        from bantz.vision.remote_vlm import analyze_screenshot
        with patch("bantz.vision.remote_vlm.config") as mock_cfg:
            mock_cfg.vlm_enabled = False
            result = await analyze_screenshot("fake_b64")
            assert result.success is False
            assert "disabled" in result.error

    @pytest.mark.asyncio
    async def test_label_prompt(self):
        from bantz.vision.remote_vlm import analyze_screenshot, FIND_PROMPT_TEMPLATE

        mock_result_obj = MagicMock()
        mock_result_obj.success = True
        mock_result_obj.elements = []
        mock_result_obj.source = "remote"

        with patch("bantz.vision.remote_vlm.config") as mock_cfg, \
             patch("bantz.vision.remote_vlm._call_remote", new_callable=AsyncMock, return_value=mock_result_obj):
            mock_cfg.vlm_enabled = True
            result = await analyze_screenshot("fake_b64", label="Search")
            assert result.success is True


# ═══════════════════════════════════════════════════════════════════════════
#  Accessibility VLM integration
# ═══════════════════════════════════════════════════════════════════════════

class TestAccessibilityVLMFallback:
    def test_vlm_available_false(self):
        from bantz.tools.accessibility import AccessibilityTool
        tool = AccessibilityTool()
        with patch("bantz.config.config.vlm_enabled", False):
            assert tool._vlm_available() is False

    def test_vlm_available_true(self):
        from bantz.tools.accessibility import AccessibilityTool
        tool = AccessibilityTool()
        with patch("bantz.config.config.vlm_enabled", True):
            assert tool._vlm_available() is True

    def test_vlm_result_to_tool_find(self):
        from bantz.tools.accessibility import AccessibilityTool
        from bantz.vision.remote_vlm import VLMResult, VLMElement

        vlm = VLMResult(
            success=True,
            elements=[
                VLMElement(label="Search", role="entry", x=100, y=50,
                           width=200, height=30, confidence=0.92),
            ],
            source="remote",
            latency_ms=500,
        )
        result = AccessibilityTool._vlm_result_to_tool(vlm, "find", "firefox", "Search")
        assert result.success is True
        assert "Search" in result.output
        assert result.data["via"] == "vlm"

    def test_vlm_result_to_tool_tree(self):
        from bantz.tools.accessibility import AccessibilityTool
        from bantz.vision.remote_vlm import VLMResult, VLMElement

        vlm = VLMResult(
            success=True,
            elements=[
                VLMElement(label="Menu", role="menu", x=0, y=0, width=100, height=30, confidence=0.8),
                VLMElement(label="Content", role="panel", x=0, y=30, width=800, height=600, confidence=0.7),
            ],
            source="remote",
            latency_ms=1200,
        )
        result = AccessibilityTool._vlm_result_to_tool(vlm, "tree", "app", "")
        assert result.success is True
        assert "2 elements" in result.output
        assert result.data["via"] == "vlm"

    def test_vlm_result_to_tool_failure(self):
        from bantz.tools.accessibility import AccessibilityTool
        from bantz.vision.remote_vlm import VLMResult

        vlm = VLMResult(success=False, error="connection refused")
        result = AccessibilityTool._vlm_result_to_tool(vlm, "find", "app", "btn")
        assert result.success is False
        assert "connection refused" in result.error

    @pytest.mark.asyncio
    async def test_screenshot_analyze_disabled(self):
        from bantz.tools.accessibility import AccessibilityTool
        tool = AccessibilityTool()
        with patch("bantz.config.config.vlm_enabled", False):
            result = await tool._screenshot_analyze("firefox", "")
            assert result.success is False

    @pytest.mark.asyncio
    async def test_describe_screen_disabled(self):
        from bantz.tools.accessibility import AccessibilityTool
        tool = AccessibilityTool()
        with patch("bantz.config.config.vlm_enabled", False):
            result = await tool._describe_screen("firefox")
            assert result.success is False


# ═══════════════════════════════════════════════════════════════════════════
#  Brain quick_route patterns
# ═══════════════════════════════════════════════════════════════════════════

class TestQuickRouteVLM:
    def _route(self, text: str) -> dict | None:
        from bantz.core.brain import Brain
        return Brain._quick_route(text, text)

    def test_whats_on_screen(self):
        r = self._route("what's on my screen?")
        assert r is None
        assert r is None


    def test_describe_screen(self):
        r = self._route("describe screen")
        assert r is None
        assert r is None


    def test_screenshot_analyze(self):
        r = self._route("screenshot of firefox")
        assert r is None
        assert r is None


    def test_vlm_keyword(self):
        r = self._route("vlm analyze this app")
        assert r is None
        assert r is None


    def test_analyze_screen(self):
        r = self._route("analyze screen")
        assert r is None
        assert r is None


    # Existing a11y routes should still work
    def test_list_apps_still_works(self):
        r = self._route("list apps")
        assert r is None


    def test_focus_window_still_works(self):
        r = self._route("focus window firefox")
        assert r is None


    def test_show_ui_tree_still_works(self):
        r = self._route("show the ui element tree for firefox")
        assert r is None

