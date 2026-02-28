"""
Bantz v2 — Gemini Client
Optional high-quality finalizer using Google Gemini Flash.
Falls back to Ollama if Gemini is unavailable or disabled.

Usage:
    from bantz.llm.gemini import gemini

    if gemini.is_enabled():
        response = await gemini.chat(messages)
"""
from __future__ import annotations

import json
import httpx
from typing import AsyncIterator

from bantz.config import config


class GeminiClient:
    """Lightweight Gemini API client using REST (no SDK dependency)."""

    def __init__(self) -> None:
        self._api_key = config.gemini_api_key
        self._model = config.gemini_model
        self._enabled = config.gemini_enabled and bool(self._api_key)
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"

    def is_enabled(self) -> bool:
        return self._enabled

    async def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        """
        Send messages to Gemini and return the response text.

        Accepts OpenAI-style messages: [{"role": "system"|"user", "content": "..."}]
        Converts to Gemini format internally.

        Raises on failure — caller should handle gracefully.
        """
        if not self._enabled:
            raise RuntimeError("Gemini is not enabled")

        # Convert OpenAI-style messages to Gemini format
        system_instruction = ""
        contents = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction += content + "\n"
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1024,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction.strip()}]
            }

        url = (
            f"{self._base_url}/models/{self._model}:generateContent"
            f"?key={self._api_key}"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract text from response
        try:
            candidates = data["candidates"]
            parts = candidates[0]["content"]["parts"]
            return parts[0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response format: {e}") from e

    async def chat_stream(self, messages: list[dict], temperature: float = 0.3) -> AsyncIterator[str]:
        """
        Stream tokens from Gemini via streamGenerateContent SSE endpoint.
        Yields text chunks as they arrive.
        """
        if not self._enabled:
            raise RuntimeError("Gemini is not enabled")

        system_instruction = ""
        contents = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction += content + "\n"
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1024,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction.strip()}]
            }

        url = (
            f"{self._base_url}/models/{self._model}:streamGenerateContent"
            f"?key={self._api_key}&alt=sse"
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", url, json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    json_str = line[6:]  # strip "data: " prefix
                    if json_str == "[DONE]":
                        return
                    try:
                        data = json.loads(json_str)
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                text = part.get("text", "")
                                if text:
                                    yield text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def is_available(self) -> bool:
        """Quick health check — list models."""
        if not self._enabled:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models?key={self._api_key}"
                )
                return resp.status_code == 200
        except Exception:
            return False


gemini = GeminiClient()
