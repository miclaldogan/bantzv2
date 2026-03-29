"""
Bantz v3 — Remote VLM Client (#120)

REST client for analysing screenshots via a remote Vision Language Model.
The VLM can run on:
  - Jetson Nano (LAN, always-on, ~1–2 s)
  - Google Colab (free GPU, on-demand, ~2–4 s)
  - Local Ollama (if a VLM model is loaded, ~3–5 s)

API contract
────────────
POST /analyze
  Body (JSON):
    {
      "image": "<base64 JPEG>",
      "prompt": "Identify all UI elements with bounding boxes",
      "format": "json"
    }

  Response (JSON):
    {
      "elements": [
        {"label": "search_bar", "role": "entry", "x": 450, "y": 120,
         "width": 300, "height": 32, "confidence": 0.92},
        ...
      ],
      "raw_text": "...",
      "latency_ms": 1234
    }

Usage:
    from bantz.vision.remote_vlm import analyze_screenshot, VLMResult

    result = await analyze_screenshot(jpeg_b64, prompt="find the search bar")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from bantz.config import config

log = logging.getLogger("bantz.vision.vlm")

_shared_client: httpx.AsyncClient | None = None

def _get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient()
    return _shared_client

# ── Prompt templates ──────────────────────────────────────────────────────

DEFAULT_PROMPT = (
    "Identify all visible UI elements in this screenshot. "
    "For each element return: label, role (button/entry/link/text/image/icon/menu/tab/other), "
    "bounding box (x, y, width, height in pixels), and confidence (0.0–1.0). "
    "Return ONLY valid JSON: {\"elements\": [{\"label\": ..., \"role\": ..., "
    "\"x\": ..., \"y\": ..., \"width\": ..., \"height\": ..., \"confidence\": ...}]}"
)

FIND_PROMPT_TEMPLATE = (
    "Find the UI element labelled '{label}' in this screenshot. "
    "Return its bounding box as JSON: "
    "{{\"elements\": [{{\"label\": \"{label}\", \"role\": \"...\", "
    "\"x\": ..., \"y\": ..., \"width\": ..., \"height\": ..., \"confidence\": ...}}]}}"
)

DESCRIBE_PROMPT = (
    "Describe what is visible on this screen. "
    "List the main UI elements, their apparent purpose, and layout."
)


# ── Result types ──────────────────────────────────────────────────────────

@dataclass
class VLMElement:
    """A UI element detected by the VLM."""
    label: str
    role: str = "other"
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    confidence: float = 0.0

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "role": self.role,
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "confidence": self.confidence,
            "center": self.center,
        }


@dataclass
class VLMResult:
    """Result from a VLM analysis."""
    success: bool
    elements: list[VLMElement] = field(default_factory=list)
    raw_text: str = ""
    latency_ms: int = 0
    error: str = ""
    source: str = ""  # "remote", "ollama"

    @property
    def best(self) -> Optional[VLMElement]:
        """Highest-confidence element."""
        if not self.elements:
            return None
        return max(self.elements, key=lambda e: e.confidence)

    def find(self, label: str) -> Optional[VLMElement]:
        """Find element by label (case-insensitive substring match)."""
        label_lower = label.lower()
        for elem in sorted(self.elements, key=lambda e: -e.confidence):
            if label_lower in elem.label.lower() or elem.label.lower() in label_lower:
                return elem
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "elements": [e.to_dict() for e in self.elements],
            "raw_text": self.raw_text,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "source": self.source,
        }


# ── JSON parsing ──────────────────────────────────────────────────────────

def parse_vlm_response(raw: str) -> list[VLMElement]:
    """
    Parse VLM response text into VLMElement objects.

    Handles:
      - Clean JSON: {"elements": [...]}
      - JSON embedded in markdown code blocks
      - JSON array directly: [{"label": ...}, ...]
    """
    import json
    import re

    text = raw.strip()

    # Strip markdown code fences
    code_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_match:
        text = code_match.group(1).strip()

    elements: list[VLMElement] = []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object or array within the text
        obj_match = re.search(r'(\{[^{}]*"elements"\s*:\s*\[.*?\]\s*\})', text, re.DOTALL)
        if obj_match:
            try:
                data = json.loads(obj_match.group(1))
            except json.JSONDecodeError:
                log.warning("VLM response is not valid JSON: %s", text[:200])
                return elements
        else:
            arr_match = re.search(r'(\[.*?\])', text, re.DOTALL)
            if arr_match:
                try:
                    data = json.loads(arr_match.group(1))
                except json.JSONDecodeError:
                    return elements
            else:
                return elements

    # Normalise: {"elements": [...]} or bare [...]
    if isinstance(data, dict):
        raw_list = data.get("elements", [])
    elif isinstance(data, list):
        raw_list = data
    else:
        return elements

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        elements.append(VLMElement(
            label=str(item.get("label", "")),
            role=str(item.get("role", "other")),
            x=int(item.get("x", 0)),
            y=int(item.get("y", 0)),
            width=int(item.get("width", 0)),
            height=int(item.get("height", 0)),
            confidence=float(item.get("confidence", 0.0)),
        ))

    return elements


# ── Remote endpoint ───────────────────────────────────────────────────────

async def _call_remote(
    image_b64: str,
    prompt: str,
    timeout: int | None = None,
) -> VLMResult:
    """
    Call remote VLM endpoint (Jetson / Colab).

    POST /analyze  {image, prompt, format}
    """
    endpoint = config.vlm_endpoint.rstrip("/")
    url = f"{endpoint}/analyze"
    timeout_s = timeout or config.vlm_timeout

    t0 = time.monotonic()
    try:
        client = _get_client()
        resp = await client.post(url, json={
            "image": image_b64,
            "prompt": prompt,
            "format": "json",
        }, timeout=float(timeout_s))
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        elapsed = int((time.monotonic() - t0) * 1000)
        return VLMResult(
            success=False, latency_ms=elapsed,
            error=f"VLM endpoint timed out after {timeout_s}s",
            source="remote",
        )
    except httpx.HTTPStatusError as exc:
        elapsed = int((time.monotonic() - t0) * 1000)
        return VLMResult(
            success=False, latency_ms=elapsed,
            error=f"VLM endpoint returned {exc.response.status_code}",
            source="remote",
        )
    except Exception as exc:
        elapsed = int((time.monotonic() - t0) * 1000)
        return VLMResult(
            success=False, latency_ms=elapsed,
            error=f"VLM endpoint error: {exc}",
            source="remote",
        )

    elapsed = int((time.monotonic() - t0) * 1000)

    # Parse response
    raw_text = data.get("raw_text", "")
    elements_raw = data.get("elements")

    if elements_raw and isinstance(elements_raw, list):
        # Response already has parsed elements
        elements = []
        for item in elements_raw:
            if isinstance(item, dict):
                elements.append(VLMElement(
                    label=str(item.get("label", "")),
                    role=str(item.get("role", "other")),
                    x=int(item.get("x", 0)),
                    y=int(item.get("y", 0)),
                    width=int(item.get("width", 0)),
                    height=int(item.get("height", 0)),
                    confidence=float(item.get("confidence", 0.0)),
                ))
    elif raw_text:
        elements = parse_vlm_response(raw_text)
    else:
        elements = []

    return VLMResult(
        success=True,
        elements=elements,
        raw_text=raw_text,
        latency_ms=data.get("latency_ms", elapsed),
        source="remote",
    )


# ── Ollama VLM fallback ──────────────────────────────────────────────────

async def _call_ollama_vlm(
    image_b64: str,
    prompt: str,
    timeout: int | None = None,
) -> VLMResult:
    """
    Use local Ollama with a VLM model (llava, bakllava, etc.) as fallback.
    """
    from bantz.config import config as _cfg
    base_url = _cfg.ollama_base_url.rstrip("/")
    url = f"{base_url}/api/generate"
    timeout_s = timeout or _cfg.vlm_timeout

    # Try common VLM models
    vlm_models = ["llava", "bakllava", "llava-llama3", "moondream"]

    t0 = time.monotonic()

    for model in vlm_models:
        try:
            client = _get_client()
            resp = await client.post(url, json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
            }, timeout=float(timeout_s))
            if resp.status_code == 404:
                continue  # model not available
            resp.raise_for_status()
            data = resp.json()
            raw_text = data.get("response", "")
            elapsed = int((time.monotonic() - t0) * 1000)

            elements = parse_vlm_response(raw_text)
            return VLMResult(
                success=True,
                elements=elements,
                raw_text=raw_text,
                latency_ms=elapsed,
                source=f"ollama:{model}",
            )
        except httpx.TimeoutException:
            continue
        except Exception:
            continue

    elapsed = int((time.monotonic() - t0) * 1000)
    return VLMResult(
        success=False, latency_ms=elapsed,
        error="No VLM model available in Ollama (tried: llava, bakllava, llava-llama3, moondream)",
        source="ollama",
    )


# ── Public API ────────────────────────────────────────────────────────────

async def analyze_screenshot(
    image_b64: str,
    prompt: str | None = None,
    label: str | None = None,
    timeout: int | None = None,
) -> VLMResult:
    """
    Analyse a screenshot via remote VLM, with Ollama fallback.

    Args:
        image_b64: Base64-encoded JPEG image.
        prompt: Custom prompt (overrides default).
        label: If set, uses a targeted find-element prompt.
        timeout: Max seconds to wait (default from config).

    Returns:
        VLMResult with detected elements and metadata.
    """
    if not config.vlm_enabled:
        return VLMResult(
            success=False,
            error="VLM is disabled. Set BANTZ_VLM_ENABLED=true to enable.",
        )

    # Build prompt
    if prompt:
        final_prompt = prompt
    elif label:
        final_prompt = FIND_PROMPT_TEMPLATE.format(label=label)
    else:
        final_prompt = DEFAULT_PROMPT

    # Try remote endpoint first
    result = await _call_remote(image_b64, final_prompt, timeout=timeout)
    if result.success:
        log.info(
            "VLM analysis: %d elements, %dms (remote)",
            len(result.elements), result.latency_ms,
        )
        return result

    log.debug("Remote VLM failed (%s), trying Ollama fallback", result.error)

    # Fallback to Ollama VLM
    result = await _call_ollama_vlm(image_b64, final_prompt, timeout=timeout)
    if result.success:
        log.info(
            "VLM analysis: %d elements, %dms (%s)",
            len(result.elements), result.latency_ms, result.source,
        )
    else:
        log.warning("All VLM backends failed: %s", result.error)

    return result


async def describe_screen(image_b64: str, timeout: int | None = None) -> VLMResult:
    """Describe what is on screen (free-text description, not structured elements)."""
    return await analyze_screenshot(
        image_b64, prompt=DESCRIBE_PROMPT, timeout=timeout,
    )


async def find_element_vlm(
    image_b64: str,
    label: str,
    timeout: int | None = None,
) -> Optional[VLMElement]:
    """
    Find a specific UI element by label using VLM analysis.

    Returns the best-matching VLMElement, or None if not found.
    """
    result = await analyze_screenshot(image_b64, label=label, timeout=timeout)
    if not result.success or not result.elements:
        return None
    return result.find(label) or result.best


# ── Spatial cache ─────────────────────────────────────────────────────────

class SpatialCache:
    """
    In-memory cache for VLM element detections.

    Keeps the last N analysis results so repeated queries for the same
    app don't trigger a new screenshot + VLM round-trip.
    TTL-based expiry prevents stale data.
    """

    def __init__(self, max_entries: int = 10, ttl_seconds: float = 30.0) -> None:
        self._cache: dict[str, tuple[float, VLMResult]] = {}
        self._max = max_entries
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[VLMResult]:
        """Get cached result if still fresh."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, result = entry
        if (time.monotonic() - ts) > self._ttl:
            del self._cache[key]
            return None
        return result

    def put(self, key: str, result: VLMResult) -> None:
        """Store a result in the cache."""
        # Evict oldest if full
        if len(self._cache) >= self._max and key not in self._cache:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[key] = (time.monotonic(), result)

    def invalidate(self, key: str | None = None) -> None:
        """Clear one key or the entire cache."""
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# Module-level singleton
spatial_cache = SpatialCache()
