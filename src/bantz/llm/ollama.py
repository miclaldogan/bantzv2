"""
Bantz v2 — Ollama Client
Communicates with Ollama. Streaming or single-response output.
"""
from __future__ import annotations

import json
import httpx
from typing import AsyncIterator

from bantz.config import config


def _notify_health(ok: bool) -> None:
    """Fire event-driven health status to OperationsHeader (#136)."""
    try:
        from bantz.interface.tui.panels.header import ServiceStatus
        from textual.app import App
        app = App.current
        if app and hasattr(app, "notify_service_health"):
            status = ServiceStatus.UP if ok else ServiceStatus.DOWN
            app.call_from_thread(app.notify_service_health, "ollama", status)
    except Exception:
        pass


class OllamaClient:
    def __init__(self) -> None:
        # Strip trailing /api/chat (common misconfiguration) and slashes
        base = config.ollama_base_url.rstrip("/")
        for suffix in ("/api/chat", "/api"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        self.base_url = base
        self.model = config.ollama_model

    async def chat(self, messages: list[dict], stream: bool = False) -> str:
        """Simple chat — returns a single string."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={"model": self.model, "messages": messages, "stream": False},
                )
                resp.raise_for_status()
                data = resp.json()
                _notify_health(True)
                return data["message"]["content"]
        except Exception:
            _notify_health(False)
            raise

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """
        Stream tokens from Ollama via NDJSON.
        Ollama /api/chat with stream:true returns lines like:
          {"message": {"content": "token"}, "done": false}
          {"message": {"content": ""}, "done": true}
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        return

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False


ollama = OllamaClient()