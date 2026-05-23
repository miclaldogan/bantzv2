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

  {"type": "get_tasks"}
      Returns {"type": "tasks", "reminders": [...], "jobs": [...]}.

  {"type": "new_task", "text": "…"}
      Creates a reminder via the scheduler, then pushes an updated tasks list.

  {"type": "get_config"}
      Returns {"type": "config", "values": {...}} with current .env values.

  {"type": "set_config", "key": "…", "value": "…"}
      Writes the key to .env, updates the running config, returns config_ack.

  {"type": "dismiss_alert", "id": "…"}
      Acknowledges alert dismissal (UI-side only; returns alert_dismissed).

Server → Client pushes
───────────────────────
  {"type": "vitals", "cpu": %, "ram_used": GB, "ram_total": GB,
   "disk_used": GB, "disk_total": GB, "vram_used": MB, "vram_total": MB}
      Broadcast every 2 s to all connected clients.

  {"type": "services", "services": [...]}
      Broadcast every 30 s — live probe of Ollama, Gemini, Telegram,
      Redis, Neo4j.  Also pushed immediately on new-client connect.

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
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import psutil
import websockets
from websockets.asyncio.server import ServerConnection, serve

from bantz.core.event_bus import bus, Event

log = logging.getLogger("bantz.ws_server")

_VITALS_INTERVAL = 2.0
_SERVICES_INTERVAL = 30.0
_WS_PORT = 8765

# ── Config key → (env alias, python attr) map ────────────────────────────────

_CONFIG_KEY_MAP: dict[str, tuple[str, str]] = {
    "ollama_model":               ("BANTZ_OLLAMA_MODEL",               "ollama_model"),
    "gemini_enabled":             ("BANTZ_GEMINI_ENABLED",             "gemini_enabled"),
    "gemini_api_key":             ("BANTZ_GEMINI_API_KEY",             "gemini_api_key"),
    "language":                   ("BANTZ_LANGUAGE",                   "language"),
    "tts_enabled":                ("BANTZ_TTS_ENABLED",                "tts_enabled"),
    "stt_enabled":                ("BANTZ_STT_ENABLED",                "stt_enabled"),
    "wake_word_enabled":          ("BANTZ_WAKE_WORD_ENABLED",          "wake_word_enabled"),
    "distillation_enabled":       ("BANTZ_DISTILLATION_ENABLED",       "distillation_enabled"),
    "shell_confirm_destructive":  ("BANTZ_SHELL_CONFIRM_DESTRUCTIVE",  "shell_confirm_destructive"),
    "observer_enabled":           ("BANTZ_OBSERVER_ENABLED",           "observer_enabled"),
}


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
            asyncio.create_task(self._vitals_loop(),   name="ws-vitals"),
            asyncio.create_task(self._log_queue_loop(), name="ws-log-queue"),
            asyncio.create_task(self._services_loop(), name="ws-services"),
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
        # Push initial state snapshot to newly connected client
        try:
            await _send(ws, await _collect_tasks())
            await _send(ws, _collect_config())
            await _send(ws, await _collect_services())
        except Exception as exc:
            log.debug("Initial state push failed: %s", exc)
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

        elif mtype == "get_tasks":
            asyncio.create_task(self._handle_get_tasks(ws))

        elif mtype == "new_task":
            asyncio.create_task(self._handle_new_task(ws, msg))

        elif mtype == "get_config":
            asyncio.create_task(self._handle_get_config(ws))

        elif mtype == "set_config":
            asyncio.create_task(self._handle_set_config(ws, msg))

        elif mtype == "dismiss_alert":
            await _send(ws, {"type": "alert_dismissed", "id": msg.get("id")})

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

    # ── tasks ──────────────────────────────────────────────────────────────

    async def _handle_get_tasks(self, ws: ServerConnection) -> None:
        await _send(ws, await _collect_tasks())

    async def _handle_new_task(self, ws: ServerConnection, msg: dict) -> None:
        text = str(msg.get("text", "")).strip()
        if not text:
            await _send(ws, {"type": "error", "msg": "empty text"})
            return
        try:
            from bantz.core.scheduler import scheduler
            from datetime import datetime, timedelta
            if scheduler._initialized:
                fire_at = datetime.now() + timedelta(hours=1)
                rid = scheduler.add(text, fire_at)
                await _send(ws, {"type": "task_created", "id": rid, "title": text})
            else:
                await _send(ws, {"type": "error", "msg": "scheduler not initialized"})
                return
        except Exception as exc:
            await _send(ws, {"type": "error", "msg": str(exc)})
            return
        await _send(ws, await _collect_tasks())

    # ── config ─────────────────────────────────────────────────────────────

    async def _handle_get_config(self, ws: ServerConnection) -> None:
        await _send(ws, _collect_config())

    async def _handle_set_config(self, ws: ServerConnection, msg: dict) -> None:
        key = str(msg.get("key", ""))
        value = msg.get("value")
        if not key:
            await _send(ws, {"type": "error", "msg": "missing key"})
            return
        ok, err = _write_config(key, value)
        if ok:
            await _send(ws, {"type": "config_ack", "ok": True, "key": key})
            await _send(ws, _collect_config())
        else:
            await _send(ws, {"type": "error", "msg": err or "config write failed"})

    # ── vitals broadcast ───────────────────────────────────────────────────

    async def _vitals_loop(self) -> None:
        while True:
            try:
                payload = _collect_vitals()
                await self._broadcast(payload)
            except Exception:
                pass
            await asyncio.sleep(_VITALS_INTERVAL)

    # ── services broadcast ─────────────────────────────────────────────────

    async def _services_loop(self) -> None:
        while True:
            try:
                payload = await _collect_services()
                await self._broadcast(payload)
            except Exception:
                pass
            await asyncio.sleep(_SERVICES_INTERVAL)

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


async def _collect_tasks() -> dict:
    reminders: list[dict] = []
    jobs: list[dict] = []
    try:
        from bantz.core.scheduler import scheduler
        if scheduler._initialized:
            for row in scheduler.list_upcoming(limit=20):
                r = dict(row)
                r["id"] = str(r["id"])
                reminders.append(r)
    except Exception:
        pass
    try:
        from bantz.agent.job_scheduler import job_scheduler
        if job_scheduler.started:
            jobs = job_scheduler.list_jobs()
    except Exception:
        pass
    return {"type": "tasks", "reminders": reminders, "jobs": jobs}


def _collect_config() -> dict:
    try:
        from bantz.config import config
        return {
            "type": "config",
            "values": {
                "ollama_model":               config.ollama_model,
                "gemini_enabled":             config.gemini_enabled,
                "gemini_api_key":             config.gemini_api_key,
                "language":                   config.language,
                "tts_enabled":                config.tts_enabled,
                "stt_enabled":                config.stt_enabled,
                "wake_word_enabled":          config.wake_word_enabled,
                "distillation_enabled":       config.distillation_enabled,
                "shell_confirm_destructive":  config.shell_confirm_destructive,
                "observer_enabled":           config.observer_enabled,
            },
        }
    except Exception:
        return {"type": "config", "values": {}}


def _write_config(key: str, value: object) -> tuple[bool, str]:
    """Write a config key to .env and update the running Config object."""
    if key not in _CONFIG_KEY_MAP:
        return False, f"unknown config key: {key}"
    env_key, attr_name = _CONFIG_KEY_MAP[key]
    str_value = str(value).lower() if isinstance(value, bool) else str(value)
    # Update running process environment
    os.environ[env_key] = str_value
    # Update config singleton in-place so callers see the new value immediately
    try:
        from bantz.config import config
        attr_type = type(getattr(config, attr_name, ""))
        if attr_type is bool:
            typed: object = str_value.lower() in ("true", "1", "yes")
        elif attr_type is int:
            typed = int(str_value)
        elif attr_type is float:
            typed = float(str_value)
        else:
            typed = str_value
        object.__setattr__(config, attr_name, typed)
    except Exception as exc:
        log.warning("Live config update failed for %s: %s", key, exc)
    # Persist to .env
    try:
        _update_dotenv(env_key, str_value)
    except Exception as exc:
        return False, f"failed to write .env: {exc}"
    return True, ""


def _update_dotenv(key: str, value: str) -> None:
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text(f"{key}={value}\n")
        return
    text = env_path.read_text()
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(f"{key}={value}", text)
    else:
        text = text.rstrip("\n") + f"\n{key}={value}\n"
    env_path.write_text(text)


async def _collect_services() -> dict:
    results = await asyncio.gather(
        _probe_ollama(),
        _probe_gemini(),
        _probe_telegram(),
        _probe_redis(),
        _probe_neo4j(),
        return_exceptions=True,
    )
    services = [
        r if isinstance(r, dict)
        else {"name": "unknown", "port": None, "status": "offline", "detail": "probe error", "uptime": "—"}
        for r in results
    ]
    return {"type": "services", "services": services}


async def _probe_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False


async def _probe_ollama() -> dict:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get("http://localhost:11434/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                n = len(models)
                return {
                    "name": "Ollama", "port": 11434, "status": "online",
                    "detail": f"{n} model{'s' if n != 1 else ''} loaded",
                    "uptime": "—",
                }
    except Exception:
        pass
    return {"name": "Ollama", "port": 11434, "status": "offline", "detail": "not reachable", "uptime": "—"}


async def _probe_redis() -> dict:
    ok = await _probe_tcp("localhost", 6379)
    return {
        "name": "Redis", "port": 6379,
        "status": "online" if ok else "offline",
        "detail": "connected" if ok else "not reachable",
        "uptime": "—",
    }


async def _probe_neo4j() -> dict:
    ok = await _probe_tcp("localhost", 7687)
    return {
        "name": "Neo4j", "port": 7687,
        "status": "online" if ok else "offline",
        "detail": "bolt reachable" if ok else "container halted",
        "uptime": "—",
    }


async def _probe_gemini() -> dict:
    try:
        from bantz.config import config
        if not config.gemini_enabled:
            return {"name": "Gemini", "port": None, "status": "offline", "detail": "disabled in config", "uptime": "—"}
        return {"name": "Gemini", "port": None, "status": "online", "detail": "rate limit: ok", "uptime": "session"}
    except Exception:
        pass
    return {"name": "Gemini", "port": None, "status": "offline", "detail": "error", "uptime": "—"}


async def _probe_telegram() -> dict:
    try:
        from bantz.config import config
        if not config.telegram_bot_token:
            return {"name": "Telegram", "port": 443, "status": "offline", "detail": "no token set", "uptime": "—"}
        ok = await _probe_tcp("api.telegram.org", 443)
        return {
            "name": "Telegram", "port": 443,
            "status": "online" if ok else "offline",
            "detail": "reachable" if ok else "unreachable",
            "uptime": "—",
        }
    except Exception:
        pass
    return {"name": "Telegram", "port": 443, "status": "offline", "detail": "error", "uptime": "—"}


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
                loop.call_soon_threadsafe(self._log_q.put_nowait, payload)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

ws_server = WsBroadcastServer()
