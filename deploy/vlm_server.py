#!/usr/bin/env python3
"""
Bantz VLM Server — Jetson / Colab Deployment

FastAPI server that accepts screenshots and returns UI element
bounding boxes via a Vision Language Model (Qwen-VL, LLaVA, etc.).

API Contract:
    POST /analyze
      Body: {"image": "<base64 JPEG>", "prompt": "...", "format": "json"}
      Returns: {"elements": [...], "raw_text": "...", "latency_ms": ...}

Deployment:
    # Jetson Nano (with ollama)
    pip install fastapi uvicorn httpx
    python vlm_server.py --model llava --port 8090

    # Colab (see deploy/vlm_colab.ipynb)
    !pip install fastapi uvicorn httpx pyngrok
    # Then run this script with --colab flag

Usage:
    curl -X POST http://localhost:8090/analyze \
      -H "Content-Type: application/json" \
      -d '{"image": "<b64>", "prompt": "Find all buttons"}'
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vlm-server")

app = FastAPI(title="Bantz VLM Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration ─────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "llava"
VLM_MODELS = ["llava", "bakllava", "llava-llama3", "moondream"]

_active_model: str = DEFAULT_MODEL


# ── Request / Response models ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image: str                     # base64 JPEG
    prompt: str = (
        "Identify all visible UI elements in this screenshot. "
        "For each element return: label, role, bounding box (x, y, width, height), "
        "and confidence. Return ONLY valid JSON."
    )
    format: str = "json"


class UIElement(BaseModel):
    label: str
    role: str = "other"
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    confidence: float = 0.0


class AnalyzeResponse(BaseModel):
    elements: list[UIElement] = []
    raw_text: str = ""
    latency_ms: int = 0
    model: str = ""
    error: str = ""


# ── Health endpoint ───────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": _active_model}


# ── Main endpoint ─────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    t0 = time.monotonic()

    # Validate image
    try:
        img_bytes = base64.b64decode(req.image)
        if len(img_bytes) < 100:
            raise ValueError("Image too small")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    # Call Ollama VLM
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": _active_model,
                    "prompt": req.prompt,
                    "images": [req.image],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        elapsed = int((time.monotonic() - t0) * 1000)
        return AnalyzeResponse(
            error="Ollama timeout", latency_ms=elapsed, model=_active_model,
        )
    except Exception as e:
        elapsed = int((time.monotonic() - t0) * 1000)
        return AnalyzeResponse(
            error=f"Ollama error: {e}", latency_ms=elapsed, model=_active_model,
        )

    elapsed = int((time.monotonic() - t0) * 1000)
    raw_text = data.get("response", "")

    # Parse elements from VLM response
    elements = _parse_elements(raw_text)

    log.info(
        "Analyzed: %d elements in %dms (model=%s, img=%dKB)",
        len(elements), elapsed, _active_model, len(img_bytes) // 1024,
    )

    return AnalyzeResponse(
        elements=elements,
        raw_text=raw_text,
        latency_ms=elapsed,
        model=_active_model,
    )


def _parse_elements(raw: str) -> list[UIElement]:
    """Parse VLM response into UIElement list."""
    import re

    text = raw.strip()

    # Strip markdown code fences
    code_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_match:
        text = code_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        obj_match = re.search(r'(\{[^{}]*"elements"\s*:\s*\[.*?\]\s*\})', text, re.DOTALL)
        if obj_match:
            try:
                data = json.loads(obj_match.group(1))
            except json.JSONDecodeError:
                return []
        else:
            arr_match = re.search(r'(\[.*?\])', text, re.DOTALL)
            if arr_match:
                try:
                    data = json.loads(arr_match.group(1))
                except json.JSONDecodeError:
                    return []
            else:
                return []

    raw_list = data.get("elements", data) if isinstance(data, dict) else data
    if not isinstance(raw_list, list):
        return []

    elements = []
    for item in raw_list:
        if isinstance(item, dict):
            elements.append(UIElement(
                label=str(item.get("label", "")),
                role=str(item.get("role", "other")),
                x=int(item.get("x", 0)),
                y=int(item.get("y", 0)),
                width=int(item.get("width", 0)),
                height=int(item.get("height", 0)),
                confidence=float(item.get("confidence", 0.0)),
            ))
    return elements


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    global OLLAMA_BASE, _active_model

    parser = argparse.ArgumentParser(description="Bantz VLM Server")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama VLM model name")
    parser.add_argument("--port", type=int, default=8090, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--ollama", default=OLLAMA_BASE, help="Ollama base URL")
    parser.add_argument("--colab", action="store_true", help="Enable ngrok tunnel (for Colab)")
    args = parser.parse_args()

    OLLAMA_BASE = args.ollama
    _active_model = args.model

    log.info("Starting VLM server: model=%s, port=%d, ollama=%s",
             _active_model, args.port, OLLAMA_BASE)

    if args.colab:
        try:
            from pyngrok import ngrok
            tunnel = ngrok.connect(args.port)
            log.info("🌐 Public URL: %s", tunnel.public_url)
            print(f"\n  VLM_ENDPOINT={tunnel.public_url}\n")
        except ImportError:
            log.warning("pyngrok not installed — no tunnel. pip install pyngrok")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
