"""
Bantz — WebSocket broadcast server for the Tauri Operations Center UI.

Listens on ws://localhost:8765 (configurable via WS_PORT env var).

Client → Server messages
────────────────────────
  {"type": "chat",  "text": "user message"}
      Routed through brain.process().  Tokens are streamed back as
      {"type": "token", "text": "…"} followed by {"type": "done"}.

  {"type": "ping"}
      Returns {"type": "pong"}.

Server → Client pushes
───────────────────────
  {"type": "vitals", "cpu": %, "ram_used": GB, "ram_total": GB,
   "disk_used": GB, "disk_total": GB, "vram_used": MB, "vram_total": MB}
      Broadcast every 2 s to all connected clients.

  {"type": "log", "msg": "…", "level": "info|warning|error|debug"}
      Every bantz.* log record forwarded to the UI.

  {"type": "alert", "title": "…", "reason": "…", "source": "health|observer"}
      Fired when health.py emits "health_alert" or observer.py emits
      "observer_error" on the EventBus.

  {"type": "broadcast", "text": "…"}
      Brain responses that are not streamed (tool results, confirmations)
      are forwarded as a single broadcast message.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import Any

import psutil
import websockets
from websockets.asyncio.server import ServerConnection, serve

from bantz.core.event_bus import bus, Event

log = logging.getLogger("bantz.ws_server")

_VITALS_INTERVAL = 2.0
_WS_PORT = 8765


# ═══════════════════════════════════════════════════════════════════════════
# Singleton server
# ═══════════════════════════════════════════════════════════════════════════

class WsBroadcastServer:
    """Manages the WebSocket server lifecycle and connected-client set."""

    def __init__(self, port: int = _WS_PORT) -> None:
        self._port = port
        self._clients: set[ServerConnection] = set()
        self._server: Any | None = None
        self._tasks: list[asyncio.Task] = []

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the WS server + background tasks.  Idempotent."""
        if self._server is not None:
            return

        self._server = await serve(
            self._handle_client,
            "localhost",
            self._port,
            ping_interval=20,
            ping_timeout=20,
        )
        bus.bind_loop()
        bus.on("health_alert", self._on_health_alert)
        bus.on("observer_error", self._on_observer_error)

        self._tasks = [
            asyncio.create_task(self._vitals_loop(), name="ws-vitals"),
            asyncio.create_task(self._log_queue_loop(), name="ws-log-queue"),
        ]
        log.info("WebSocket server listening on ws://localhost:%d", self._port)

    async def stop(self) -> None:
        """Gracefully shut down."""
        for t in self._tasks:
            t.cancel()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        log.info("WebSocket server stopped")

    # ── client handler ─────────────────────────────────────────────────────

    async def _handle_client(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        addr = ws.remote_address
        log.info("WS client connected: %s", addr)
        try:
            async for raw in ws:
                await self._dispatch(ws, raw)
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosedError as exc:
            log.debug("WS client disconnected with error: %s", exc)
        finally:
            self._clients.discard(ws)
            log.info("WS client disconnected: %s", addr)

    async def _dispatch(self, ws: ServerConnection, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            await _send(ws, {"type": "error", "msg": "invalid JSON"})
            return

        mtype = msg.get("type")

        if mtype == "ping":
            await _send(ws, {"type": "pong"})

        elif mtype == "chat":
            text = str(msg.get("text", "")).strip()
            if text:
                asyncio.create_task(self._handle_chat(ws, text))
            else:
                await _send(ws, {"type": "error", "msg": "empty text"})

        else:
            await _send(ws, {"type": "error", "msg": f"unknown type: {mtype}"})

    # ── chat processing ────────────────────────────────────────────────────

    async def _handle_chat(self, ws: ServerConnection, text: str) -> None:
        from bantz.core.brain import brain

        try:
            result = await brain.process(text)
        except Exception as exc:
            await _send(ws, {"type": "error", "msg": str(exc)})
            return

        if result.stream is not None:
            try:
                async for token in result.stream:
                    await _send(ws, {"type": "token", "text": token})
            except Exception as exc:
                await _send(ws, {"type": "error", "msg": f"stream error: {exc}"})
                return
            await _send(ws, {"type": "done"})
        else:
            response = result.response or ""
            if response.strip():
                await _send(ws, {"type": "broadcast", "text": response})
            await _send(ws, {"type": "done"})

    # ── vitals broadcast ───────────────────────────────────────────────────

    async def _vitals_loop(self) -> None:
        while True:
            try:
                payload = _collect_vitals()
                await self._broadcast(payload)
            except Exception:
                pass
            await asyncio.sleep(_VITALS_INTERVAL)

    # ── log forwarding ─────────────────────────────────────────────────────

    # The server owns an asyncio.Queue that _WSLogHandler fills from any thread.
    _log_q: asyncio.Queue[dict] = asyncio.Queue()

    async def _log_queue_loop(self) -> None:
        handler = _WSLogHandler(self._log_q)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        bantz_log = logging.getLogger("bantz")
        bantz_log.addHandler(handler)
        try:
            while True:
                payload = await self._log_q.get()
                await self._broadcast(payload)
        finally:
            bantz_log.removeHandler(handler)

    # ── event bus callbacks ────────────────────────────────────────────────

    def _on_health_alert(self, event: Event) -> None:
        payload = {
            "type": "alert",
            "source": "health",
            "title": event.data.get("title", "Health Alert"),
            "reason": event.data.get("reason", ""),
        }
        asyncio.create_task(self._broadcast(payload))

    def _on_observer_error(self, event: Event) -> None:
        payload = {
            "type": "alert",
            "source": "observer",
            "title": "Observer error",
            "reason": str(event.data.get("message", event.data)),
        }
        asyncio.create_task(self._broadcast(payload))

    # ── helpers ────────────────────────────────────────────────────────────

    async def _broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        data = json.dumps(payload)
        dead: set[ServerConnection] = set()
        for ws in list(self._clients):
            try:
                await ws.send(data)
            except Exception:
                dead.add(ws)
        self._clients -= dead


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

async def _send(ws: ServerConnection, payload: dict) -> None:
    try:
        await ws.send(json.dumps(payload))
    except Exception:
        pass


def _collect_vitals() -> dict:
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    vram_used, vram_total = _collect_vram()
    return {
        "type": "vitals",
        "cpu": round(cpu, 1),
        "ram_used": round(mem.used / (1024 ** 3), 2),
        "ram_total": round(mem.total / (1024 ** 3), 2),
        "disk_used": round(disk.used / (1024 ** 3), 1),
        "disk_total": round(disk.total / (1024 ** 3), 1),
        "vram_used": round(vram_used, 0),
        "vram_total": round(vram_total, 0),
    }


def _collect_vram() -> tuple[float, float]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            if len(parts) == 2:
                return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        pass
    return 0.0, 0.0


class _WSLogHandler(logging.Handler):
    """Puts log records into the server's asyncio queue (thread-safe)."""

    def __init__(self, q: asyncio.Queue) -> None:
        super().__init__()
        self._q = q
        self._loop: asyncio.AbstractEventLoop | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = record.levelname.lower()
            payload = {"type": "log", "msg": self.format(record), "level": level}
            loop = self._loop
            if loop is None:
                try:
                    loop = asyncio.get_running_loop()
                    self._loop = loop
                except RuntimeError:
                    return
            if loop and not loop.is_closed():
                loop.call_soon_threadsafe(self._q.put_nowait, payload)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

ws_server = WsBroadcastServer()
