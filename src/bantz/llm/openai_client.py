"""
Bantz — OpenAI-compatible client
Works with OpenAI, any OpenAI-compatible endpoint (local vLLM, LM Studio, etc.)
via BANTZ_OPENAI_BASE_URL.
"""
from __future__ import annotations

import json
import httpx
from typing import AsyncIterator

from bantz.config import config


class OpenAIClient:

    def __init__(self) -> None:
        self._api_key  = config.openai_api_key
        self._model    = config.openai_model
        self._base_url = config.openai_base_url.rstrip("/")
        self._enabled  = bool(self._api_key)
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    def is_enabled(self) -> bool:
        return self._enabled

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        resp = await self.client.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json={"model": self._model, "messages": messages, "stream": False},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def chat_stream(self, messages: list[dict], temperature: float = 0.7) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json={"model": self._model, "messages": messages, "stream": True},
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                json_str = line[6:]
                if json_str == "[DONE]":
                    return
                try:
                    data = json.loads(json_str)
                    content = data["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def is_available(self) -> bool:
        if not self._enabled:
            return False
        try:
            resp = await self.client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False


openai_client = OpenAIClient()
