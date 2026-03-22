"""
Tests for Bantz v3 — Remote VLM Client (#120)

Coverage:
  - VLMElement / VLMResult dataclasses
  - parse_vlm_response (JSON, markdown, array fallbacks)
  - _call_remote (success, timeout, HTTP error, generic error)
  - _call_ollama_vlm (success, fallback through models, failure)
  - analyze_screenshot (routing, enabled check)
  - describe_screen / find_element_vlm
  - SpatialCache (in-memory)
"""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock, AsyncMock

import httpx

from bantz.vision.remote_vlm import (
    VLMElement,
    VLMResult,
    parse_vlm_response,
    _call_remote,
    _call_ollama_vlm,
    analyze_screenshot,
    describe_screen,
    find_element_vlm,
    SpatialCache,
    FIND_PROMPT_TEMPLATE,
    DESCRIBE_PROMPT,
)


# ── Result Types ──────────────────────────────────────────────────────────

class TestVLMElement:
    def test_center(self):
        elem = VLMElement(label="btn", x=100, y=200, width=50, height=20)
        assert elem.center == (125, 210)

    def test_to_dict(self):
        elem = VLMElement(label="btn", role="button", x=10, y=20, width=30, height=40, confidence=0.9)
        d = elem.to_dict()
        assert d == {
            "label": "btn",
            "role": "button",
            "x": 10,
            "y": 20,
            "width": 30,
            "height": 40,
            "confidence": 0.9,
            "center": (25, 40),
        }

class TestVLMResult:
    def test_best_element(self):
        res = VLMResult(success=True, elements=[
            VLMElement(label="a", confidence=0.5),
            VLMElement(label="b", confidence=0.9),
            VLMElement(label="c", confidence=0.1),
        ])
        assert res.best.label == "b"

    def test_best_empty(self):
        res = VLMResult(success=True, elements=[])
        assert res.best is None

    def test_find_element(self):
        res = VLMResult(success=True, elements=[
            VLMElement(label="Submit Button", confidence=0.8),
            VLMElement(label="Cancel", confidence=0.9),
        ])
        # Case insensitive match
        elem = res.find("submit")
        assert elem is not None
        assert elem.label == "Submit Button"

        # Substring match
        elem2 = res.find("Button")
        assert elem2 is not None

        # Not found
        assert res.find("X") is None

    def test_to_dict(self):
        res = VLMResult(success=True, raw_text="raw", latency_ms=100, error="err", source="test")
        d = res.to_dict()
        assert d["success"] is True
        assert d["raw_text"] == "raw"
        assert d["latency_ms"] == 100
        assert d["error"] == "err"
        assert d["source"] == "test"
        assert "elements" in d


# ── Parsing ───────────────────────────────────────────────────────────────

class TestParseVLMResponse:
    def test_clean_json(self):
        raw = '{"elements": [{"label": "x", "x": 10, "y": 20, "width": 30, "height": 40, "confidence": 0.5}]}'
        elements = parse_vlm_response(raw)
        assert len(elements) == 1
        assert elements[0].label == "x"
        assert elements[0].x == 10
        assert elements[0].confidence == 0.5

    def test_markdown_code_block(self):
        raw = "```json\n" + '{"elements": [{"label": "y"}]}' + "\n```"
        elements = parse_vlm_response(raw)
        assert len(elements) == 1
        assert elements[0].label == "y"

    def test_bare_array(self):
        raw = '[{"label": "z"}]'
        elements = parse_vlm_response(raw)
        assert len(elements) == 1
        assert elements[0].label == "z"

    def test_embedded_object(self):
        raw = 'Here is the data: {"elements": [{"label": "a"}]} End.'
        elements = parse_vlm_response(raw)
        assert len(elements) == 1
        assert elements[0].label == "a"

    def test_embedded_array(self):
        raw = 'I found these: [{"label": "b"}] OK?'
        elements = parse_vlm_response(raw)
        assert len(elements) == 1
        assert elements[0].label == "b"

    def test_invalid_json(self):
        raw = "This is just text, no JSON."
        elements = parse_vlm_response(raw)
        assert len(elements) == 0

    def test_missing_fields_defaults(self):
        raw = '{"elements": [{"label": "x"}]}'
        elements = parse_vlm_response(raw)
        assert elements[0].role == "other"
        assert elements[0].x == 0
        assert elements[0].width == 0
        assert elements[0].confidence == 0.0


# ── Spatial Cache ─────────────────────────────────────────────────────────

class TestSpatialCache:
    def test_put_and_get(self):
        cache = SpatialCache(max_entries=2, ttl_seconds=1.0)
        res = VLMResult(success=True)
        cache.put("key1", res)
        assert cache.get("key1") is res
        assert cache.size == 1

    def test_ttl_expiry(self):
        cache = SpatialCache(ttl_seconds=0.1)
        res = VLMResult(success=True)
        cache.put("key1", res)
        time.sleep(0.15)
        assert cache.get("key1") is None
        assert cache.size == 0

    def test_lru_eviction(self):
        cache = SpatialCache(max_entries=2, ttl_seconds=10.0)
        cache.put("k1", VLMResult(success=True))
        time.sleep(0.01)
        cache.put("k2", VLMResult(success=True))
        time.sleep(0.01)
        cache.put("k3", VLMResult(success=True))

        assert cache.size == 2
        assert cache.get("k1") is None  # evicted
        assert cache.get("k2") is not None
        assert cache.get("k3") is not None

    def test_invalidate(self):
        cache = SpatialCache()
        cache.put("k1", VLMResult(success=True))
        cache.put("k2", VLMResult(success=True))

        cache.invalidate("k1")
        assert cache.get("k1") is None
        assert cache.get("k2") is not None

        cache.invalidate()
        assert cache.size == 0


# ── API Calls ─────────────────────────────────────────────────────────────

class TestCallRemote:
    @patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    async def test_call_remote_success_parsed(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "elements": [{"label": "btn", "confidence": 0.9}],
            "latency_ms": 150
        }
        mock_post.return_value = mock_resp

        res = await _call_remote("b64", "prompt")
        assert res.success is True
        assert len(res.elements) == 1
        assert res.elements[0].label == "btn"
        assert res.latency_ms == 150
        assert res.source == "remote"

    @patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    async def test_call_remote_success_raw_text(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "raw_text": '{"elements": [{"label": "raw_btn"}]}',
            "latency_ms": 200
        }
        mock_post.return_value = mock_resp

        res = await _call_remote("b64", "prompt")
        assert res.success is True
        assert len(res.elements) == 1
        assert res.elements[0].label == "raw_btn"

    @patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    async def test_call_remote_timeout(self, mock_post):
        mock_post.side_effect = httpx.TimeoutException("timeout")
        res = await _call_remote("b64", "prompt")
        assert res.success is False
        assert "timed out" in res.error

    @patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    async def test_call_remote_http_error(self, mock_post):
        exc = httpx.HTTPStatusError("error", request=MagicMock(), response=MagicMock(status_code=500))
        mock_post.side_effect = exc
        res = await _call_remote("b64", "prompt")
        assert res.success is False
        assert "500" in res.error

    @patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    async def test_call_remote_generic_error(self, mock_post):
        mock_post.side_effect = ValueError("bad data")
        res = await _call_remote("b64", "prompt")
        assert res.success is False
        assert "bad data" in res.error


class TestCallOllamaVLM:
    @patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    async def test_ollama_success_first_model(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "response": '{"elements": [{"label": "ollama_btn"}]}'
        }
        mock_post.return_value = mock_resp

        res = await _call_ollama_vlm("b64", "prompt")
        assert res.success is True
        assert len(res.elements) == 1
        assert res.elements[0].label == "ollama_btn"
        assert res.source == "ollama:llava"

    @patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    async def test_ollama_fallback_through_models(self, mock_post):
        # Fail first model with 404, succeed second
        resp_404 = MagicMock(status_code=404)
        resp_200 = MagicMock(status_code=200)
        resp_200.raise_for_status.return_value = None
        resp_200.json.return_value = {"response": '{"elements": [{"label": "bakllava_btn"}]}'}

        mock_post.side_effect = [resp_404, resp_200]

        res = await _call_ollama_vlm("b64", "prompt")
        assert res.success is True
        assert res.elements[0].label == "bakllava_btn"
        assert res.source == "ollama:bakllava"

    @patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    async def test_ollama_all_fail(self, mock_post):
        mock_post.side_effect = httpx.TimeoutException("timeout")
        res = await _call_ollama_vlm("b64", "prompt")
        assert res.success is False
        assert "No VLM model available" in res.error

# ── Public API ────────────────────────────────────────────────────────────

class TestPublicAPI:
    @patch("bantz.vision.remote_vlm.config")
    async def test_analyze_screenshot_disabled(self, mock_config):
        mock_config.vlm_enabled = False
        res = await analyze_screenshot("b64")
        assert res.success is False
        assert "disabled" in res.error

    @patch("bantz.vision.remote_vlm.config")
    @patch("bantz.vision.remote_vlm._call_remote", new_callable=AsyncMock)
    async def test_analyze_screenshot_remote_success(self, mock_remote, mock_config):
        mock_config.vlm_enabled = True
        mock_remote.return_value = VLMResult(success=True, elements=[VLMElement(label="remote_btn")])

        res = await analyze_screenshot("b64", prompt="custom")
        assert res.success is True
        assert res.elements[0].label == "remote_btn"
        mock_remote.assert_called_once_with("b64", "custom", timeout=None)

    @patch("bantz.vision.remote_vlm.config")
    @patch("bantz.vision.remote_vlm._call_remote", new_callable=AsyncMock)
    @patch("bantz.vision.remote_vlm._call_ollama_vlm", new_callable=AsyncMock)
    async def test_analyze_screenshot_remote_fails_ollama_success(self, mock_ollama, mock_remote, mock_config):
        mock_config.vlm_enabled = True
        mock_remote.return_value = VLMResult(success=False, error="remote down")
        mock_ollama.return_value = VLMResult(success=True, elements=[VLMElement(label="ollama_btn")])

        res = await analyze_screenshot("b64", label="search")
        assert res.success is True
        assert res.elements[0].label == "ollama_btn"
        # Check prompt formatting for label
        expected_prompt = FIND_PROMPT_TEMPLATE.format(label="search")
        mock_remote.assert_called_once_with("b64", expected_prompt, timeout=None)
        mock_ollama.assert_called_once_with("b64", expected_prompt, timeout=None)

    @patch("bantz.vision.remote_vlm.analyze_screenshot", new_callable=AsyncMock)
    async def test_describe_screen(self, mock_analyze):
        mock_analyze.return_value = VLMResult(success=True, raw_text="description")
        res = await describe_screen("b64")
        assert res.success is True
        assert res.raw_text == "description"
        mock_analyze.assert_called_once_with("b64", prompt=DESCRIBE_PROMPT, timeout=None)

    @patch("bantz.vision.remote_vlm.analyze_screenshot", new_callable=AsyncMock)
    async def test_find_element_vlm_success(self, mock_analyze):
        mock_analyze.return_value = VLMResult(success=True, elements=[
            VLMElement(label="Search", confidence=0.9),
            VLMElement(label="Cancel", confidence=0.5)
        ])
        res = await find_element_vlm("b64", "Search")
        assert res is not None
        assert res.label == "Search"
        mock_analyze.assert_called_once_with("b64", label="Search", timeout=None)

    @patch("bantz.vision.remote_vlm.analyze_screenshot", new_callable=AsyncMock)
    async def test_find_element_vlm_not_found(self, mock_analyze):
        mock_analyze.return_value = VLMResult(success=False)
        res = await find_element_vlm("b64", "Search")
        assert res is None
