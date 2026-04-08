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
    """Fire event-driven health status to OperationsHeader (#136).

    Textual v8: call_from_thread raises RuntimeError when called from the
    same thread as the app.  Detect context and call directly on main thread.
    """
    import threading as _threading
    try:
        from bantz.interface.tui.panels.header import ServiceStatus
        from textual.app import App
        app = App.current
        if app and hasattr(app, "notify_service_health"):
            status = ServiceStatus.UP if ok else ServiceStatus.DOWN
            if _threading.current_thread() is _threading.main_thread():
                app.notify_service_health("ollama", status)
            else:
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
        # Optional fast model for routing — falls back to main model
        self.routing_model = config.ollama_routing_model or self.model

        # Layer 1: instant URL format check — no I/O, safe in __init__
        from urllib.parse import urlparse
        parsed = urlparse(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(
                f"Invalid BANTZ_OLLAMA_BASE_URL: '{self.base_url}'. "
                f"Expected format: http://<host>:<port>"
            )
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def verify_connection(self) -> None:
        """Layers 2+3: connectivity + model availability check.

        Call at startup or via `bantz --doctor`. Never call from __init__
        (network I/O on the main thread would freeze the TUI).
        """
        try:
            r = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            r.raise_for_status()
            data = r.json()
        except httpx.ConnectError:
            is_remote = "localhost" not in self.base_url and "127.0.0.1" not in self.base_url
            tunnel_hint = (
                "\n  Is the SSH tunnel active?\n"
                "  ssh -N -L 11434:localhost:11434 root@<VPS_IP>"
                if is_remote
                else ""
            )
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}.{tunnel_hint}"
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama at {self.base_url} timed out after 5s. "
                f"Server may be overloaded or tunnel may be down."
            )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}. "
                f"Check if '{self.base_url}' is actually an Ollama instance."
            )
        except (ValueError, KeyError):
            raise RuntimeError(
                f"Invalid response from {self.base_url}. "
                f"Wrong host URL or captive portal intercept."
            )

        # Layer 3: model availability
        available = [m["name"] for m in data.get("models", [])]
        if self.model not in available:
            raise RuntimeError(
                f"Model '{self.model}' not found on {self.base_url}.\n"
                f"  Available: {available or ['(none)']}\n"
                f"  Run: ollama pull {self.model}"
            )

        # Layer 3b: routing model availability
        if self.routing_model != self.model and self.routing_model not in available:
            import logging as _logging
            _log = _logging.getLogger("bantz.llm.ollama")
            _log.warning(
                "Routing model '%s' not found — falling back to main model '%s'. "
                "Install it with: ollama pull %s",
                self.routing_model, self.model, self.routing_model,
            )
            self.routing_model = self.model

    async def chat(self, messages: list[dict], stream: bool = False, *, options: dict | None = None, model_override: str = "") -> str:
        """Simple chat — returns a single string."""
        try:
            payload: dict = {"model": model_override or self.model, "messages": messages, "stream": False}
            if options:
                payload["options"] = options
            resp = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            _notify_health(True)
            return data["message"]["content"]
        except Exception:
            _notify_health(False)
            raise

    async def chat_stream(self, messages: list[dict], *, options: dict | None = None, model_override: str = "") -> AsyncIterator[str]:
        """
        Stream tokens from Ollama via NDJSON.
        Ollama /api/chat with stream:true returns lines like:
          {"message": {"content": "token"}, "done": false}
          {"message": {"content": ""}, "done": true}
        """
        payload: dict = {"model": model_override or self.model, "messages": messages, "stream": True}
        if options:
            payload["options"] = options
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120.0,
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
            resp = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


ollama = OllamaClient()