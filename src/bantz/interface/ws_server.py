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
import time
from collections import deque
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

# Rolling buffer of the most recent log payloads, scanned by _compute_anomalies
# for ERROR/CRITICAL entries. Filled by _log_queue_loop as logs stream through.
_recent_logs: deque[dict] = deque(maxlen=50)

# Anomaly detection thresholds (percent). Tuned for a personal machine.
_CPU_CRIT = 80.0
_RAM_CRIT = 85.0
_DISK_CRIT = 85.0
_SWAP_WARN = 60.0
_SWAP_CRIT = 85.0
# Combined memory-pressure: RAM and swap both elevated at once.
_MEM_PRESSURE_RAM = 80.0
_MEM_PRESSURE_SWAP = 50.0

# ── Config key → (env alias, python attr) map ────────────────────────────────

_CONFIG_KEY_MAP: dict[str, tuple[str, str]] = {
    "llm_provider":               ("BANTZ_LLM_PROVIDER",               "llm_provider"),
    "ollama_model":               ("BANTZ_OLLAMA_MODEL",               "ollama_model"),
    "anthropic_api_key":          ("BANTZ_ANTHROPIC_API_KEY",          "anthropic_api_key"),
    "anthropic_model":            ("BANTZ_ANTHROPIC_MODEL",            "anthropic_model"),
    "gemini_enabled":             ("BANTZ_GEMINI_ENABLED",             "gemini_enabled"),
    "gemini_api_key":             ("BANTZ_GEMINI_API_KEY",             "gemini_api_key"),
    "language":                   ("BANTZ_LANGUAGE",                   "language"),
    "tts_enabled":                ("BANTZ_TTS_ENABLED",                "tts_enabled"),
    "stt_enabled":                ("BANTZ_STT_ENABLED",                "stt_enabled"),
    "wake_word_enabled":          ("BANTZ_WAKE_WORD_ENABLED",          "wake_word_enabled"),
    "distillation_enabled":       ("BANTZ_DISTILLATION_ENABLED",       "distillation_enabled"),
    "shell_confirm_destructive":  ("BANTZ_SHELL_CONFIRM_DESTRUCTIVE",  "shell_confirm_destructive"),
    "observer_enabled":           ("BANTZ_OBSERVER_ENABLED",           "observer_enabled"),
    "verbosity":                  ("BANTZ_VERBOSITY",                  "verbosity"),
    "autonomy":                   ("BANTZ_AUTONOMY",                   "autonomy"),
    "mood_bias":                  ("BANTZ_MOOD_BIAS",                  "mood_bias"),
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
        bus.on("research_progress", self._on_research_progress)

        self._tasks = [
            asyncio.create_task(self._vitals_loop(),        name="ws-vitals"),
            asyncio.create_task(self._log_queue_loop(),      name="ws-log-queue"),
            asyncio.create_task(self._services_loop(),       name="ws-services"),
            asyncio.create_task(self._preload_translation(), name="ws-translation-preload"),
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

        elif mtype == "cancel_research":
            self._handle_cancel_research()
            await _send(ws, {"type": "research_cancelled"})

        else:
            await _send(ws, {"type": "error", "msg": f"unknown type: {mtype}"})

    # ── chat processing ────────────────────────────────────────────────────

    async def _handle_chat(self, ws: ServerConnection, text: str) -> None:
        log.debug("chat received | raw=%r", text)
        from bantz.core.brain import brain
        from bantz.core.event_bus import bus as _event_bus, Event as _Event

        # Forward planning-phase events as butler-voice narration tokens so the
        # UI shows live progress instead of dead silence during LLM calls.

        async def _on_planning_start(event: _Event) -> None:
            # Structured event for the Operations Center plan view
            await _send(ws, {"type": "plan_start"})
            tier = _butler_tier()
            phrase = await _to_tr(_PLANNING_START[tier])
            await _send(ws, {"type": "token", "text": phrase + "\n"})

        async def _on_planner_step(event: _Event) -> None:
            d = event.data
            # Structured per-step event (start/done/failed) so the desktop
            # UI can render a live checklist, not just narration prose.
            await _send(ws, {
                "type": "plan_step",
                "step": d.get("step", 0),
                "total": d.get("total", 0),
                "tool": d.get("tool", ""),
                "description": d.get("description", ""),
                "status": d.get("status", "start"),
            })
            if d.get("status") != "start":
                return
            tier   = _butler_tier()
            tool   = d.get("tool", "")
            phrase = await _to_tr(_STEP_PHRASES.get(tool, _STEP_FALLBACK)[tier])
            await _send(ws, {"type": "token", "text": phrase + "\n"})

        _event_bus.on("planning_start", _on_planning_start)
        _event_bus.on("planner_step",   _on_planner_step)
        try:
            result = await brain.process(text)
        except Exception as exc:
            await _send(ws, {"type": "error", "msg": str(exc)})
            await _send(ws, {"type": "done"})
            return
        finally:
            _event_bus.off("planning_start", _on_planning_start)
            _event_bus.off("planner_step",   _on_planner_step)

        if result.stream is not None:
            # finalize_stream() now translates sentence-by-sentence when the
            # language bridge is enabled (#422), so tokens arrive already in
            # Turkish.  When the bridge is disabled the tokens are plain
            # English and no translation is needed either way.
            parts: list[str] = []
            try:
                async for token in result.stream:
                    parts.append(token)
            except Exception as exc:
                await _send(ws, {"type": "error", "msg": f"stream error: {exc}"})
                return
            response = "".join(parts)
            if response.strip():
                await _send(ws, {"type": "token", "text": response})
            await _send(ws, {"type": "done"})
        else:
            # Non-streaming path (tool results, planner output, etc.).
            response = await _to_tr(result.response or "")
            if response.strip():
                await _send(ws, {"type": "token", "text": response})
            await _send(ws, {"type": "done"})

        # ── TTS: speak the response aloud when enabled ────────────────────
        if response.strip():
            _maybe_speak(response)

    # ── tasks ──────────────────────────────────────────────────────────────

    async def _handle_get_tasks(self, ws: ServerConnection) -> None:
        await _send(ws, await _collect_tasks())

    async def _handle_new_task(self, ws: ServerConnection, msg: dict) -> None:
        text = str(msg.get("text", "")).strip()
        if not text:
            await _send(ws, {"type": "error", "msg": "empty text"})
            return

        rid: object = None
        title = text

        # Try to parse the directive into a proper scheduled job (cron/interval/once).
        parsed = await _parse_directive(text)
        if parsed:
            try:
                from bantz.agent.job_scheduler import job_scheduler
                from datetime import datetime
                st = parsed.get("schedule_type")
                if job_scheduler.started and st in ("cron", "interval", "once"):
                    title = parsed.get("title") or text
                    fire_at = None
                    if st == "once" and parsed.get("fire_at"):
                        try:
                            fire_at = datetime.fromisoformat(parsed["fire_at"])
                        except (ValueError, TypeError):
                            fire_at = None
                    rid = job_scheduler.add_dynamic_job(
                        title, st,
                        cron_expr=str(parsed.get("cron_expr", "")),
                        interval_seconds=int(parsed.get("interval_seconds") or 0),
                        fire_at=fire_at,
                        priority=str(parsed.get("priority", "medium")),
                    )
            except Exception as exc:
                log.warning("directive scheduling failed, falling back: %s", exc)
                rid = None

        # Fallback: one-time reminder firing in 1 hour (legacy behavior).
        if not rid:
            try:
                from bantz.core.scheduler import scheduler
                from datetime import datetime, timedelta
                if scheduler._initialized:
                    rid = scheduler.add(text, datetime.now() + timedelta(hours=1))
                    title = text
                else:
                    await _send(ws, {"type": "error", "msg": "scheduler not initialized"})
                    return
            except Exception as exc:
                await _send(ws, {"type": "error", "msg": str(exc)})
                return

        await _send(ws, {"type": "task_created", "id": str(rid), "title": title})
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

    # ── translation model preload ──────────────────────────────────────────

    async def _preload_translation(self) -> None:
        """Load MarianMT models into memory at startup so the first user
        message doesn't pay the 20-30s disk-load cost."""
        try:
            from bantz.i18n.bridge import bridge
            if not bridge.is_enabled():
                return
            log.info("Preloading translation models in background…")
            await asyncio.get_running_loop().run_in_executor(None, bridge.preload)
            log.info("Translation models ready.")
        except Exception as exc:
            log.warning("Translation model preload failed: %s", exc)

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
                _recent_logs.append(payload)  # keep for anomaly scanning
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

    def _on_research_progress(self, event: Event) -> None:
        """Bridge a 'research_progress' bus event to a structured frame (#490).

        The Broadcast Channel renders these as a compact progress indicator
        instead of interleaving raw text with the chat transcript. ``state`` is
        "running" | "done" | "cancelled"; the UI clears the indicator on the
        terminal states."""
        d = event.data
        asyncio.create_task(self._broadcast({
            "type": "research_progress",
            "stage": d.get("stage", ""),
            "detail": d.get("detail", ""),
            "elapsed": int(d.get("elapsed", 0) or 0),
            "state": d.get("state", "running"),
        }))

    def _handle_cancel_research(self) -> None:
        """Set the web_research tool's cancel flag (WS 'cancel_research')."""
        try:
            from bantz.tools import registry
            tool = registry.get("web_research")
            if tool is not None and hasattr(tool, "_research_cancelled"):
                tool._research_cancelled.set()
        except Exception as exc:
            log.debug("cancel_research failed: %s", exc)

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

# ── Butler-voice planning narration ───────────────────────────────────────────
# Phrases are keyed by affinity tier: "formal" | "warm" | "wry".
# Formal = new/neutral relationship; warm = established; wry = close.

_PLANNING_START: dict[str, str] = {
    "formal": "Very well. One moment.",
    "warm":   "Of course — allow me a moment.",
    "wry":    "Right away. Leave it with me.",
}

_STEP_PHRASES: dict[str, dict[str, str]] = {
    "web_search":     {"formal": "Searching, as requested.",          "warm": "Searching for what you've asked about.",       "wry": "Let me see what I can turn up."},
    "read_url":       {"formal": "Reading the article.",              "warm": "Reading the full article now.",                "wry": "Having a thorough read through this one."},
    "summarizer":     {"formal": "Preparing a summary.",              "warm": "Pulling together the key points.",             "wry": "Distilling this into something useful."},
    "process_text":   {"formal": "Preparing a summary.",              "warm": "Pulling together the key points.",             "wry": "Distilling this into something useful."},
    "filesystem":     {"formal": "Saving to file.",                   "warm": "Saving the results now.",                      "wry": "Filing that away for you."},
    "shell":          {"formal": "Running the command.",              "warm": "Running that through the terminal.",           "wry": "Firing up the terminal — back in a moment."},
    "browser_control":{"formal": "Opening the browser.",             "warm": "Navigating the browser now.",                  "wry": "Taking the wheel, so to speak."},
    "gmail":          {"formal": "Checking your correspondence.",     "warm": "Checking your messages now.",                  "wry": "Having a look at the post."},
    "calendar":       {"formal": "Consulting the calendar.",          "warm": "Checking your schedule.",                      "wry": "Having a look at the diary."},
    "weather":        {"formal": "Checking the weather.",             "warm": "Checking conditions outside.",                 "wry": "Consulting the forecast."},
    "news":           {"formal": "Checking the headlines.",           "warm": "Scanning the latest news.",                   "wry": "Having a look at what's in the papers."},
    "delegate_task":  {"formal": "Delegating the task.",              "warm": "Handing this off to a specialist.",            "wry": "Putting the right person on this."},
    "run_workflow":   {"formal": "Running the workflow.",             "warm": "Running the appropriate workflow.",            "wry": "Setting the machinery in motion."},
    "system":         {"formal": "Checking system status.",           "warm": "Checking the system's vitals.",                "wry": "Having a look under the bonnet."},
    "document":       {"formal": "Reading the document.",             "warm": "Going through the document.",                  "wry": "Having a thorough look at this one."},
}

_STEP_FALLBACK: dict[str, str] = {
    "formal": "Working on it.",
    "warm":   "Working on the next part now.",
    "wry":    "Right, on to the next bit.",
}


def _butler_tier() -> str:
    """Return 'formal', 'warm', or 'wry' based on current affinity score."""
    try:
        from bantz.agent.affinity_engine import affinity_engine
        if affinity_engine.initialized:
            score = affinity_engine.get_score()
            if score >= 60:
                return "wry"
            if score >= 20:
                return "warm"
    except Exception:
        pass
    return "formal"


async def _to_tr(text: str) -> str:
    """Translate EN → TR if the language bridge is enabled; no-op otherwise.

    This is the single point where the output side of the translation
    pipeline runs.  Brain and all internal modules stay in English —
    only the final user-facing text is translated here before sending.
    """
    if not text.strip():
        return text
    try:
        from bantz.i18n.bridge import bridge
        if bridge.is_enabled():
            return await asyncio.wait_for(bridge.to_turkish(text), timeout=60)
    except asyncio.TimeoutError:
        log.warning("EN→TR translation timed out — returning English text")
    except Exception as exc:
        log.debug("EN→TR translation failed: %s", exc)
    return text


def _maybe_speak(text: str) -> None:
    """Fire-and-forget TTS for a chat response when speak_all_responses is on.

    Called after the response has been sent to the UI so TTS never delays
    the WebSocket reply.  Skipped silently if TTS is unavailable.
    """
    try:
        from bantz.config import config as _cfg
        if not _cfg.tts_enabled or not _cfg.tts_speak_all_responses:
            return
        from bantz.agent.tts import tts_engine
        if tts_engine.available():
            asyncio.create_task(tts_engine.speak_background(text))
    except Exception as exc:
        log.debug("TTS speak skipped: %s", exc)


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
        "anomalies": _compute_anomalies(cpu, mem.percent),
    }


def _compute_anomalies(cpu: float, ram_pct: float) -> list[dict]:
    """Derive current anomalies from resource pressure + recent error logs.

    Stable ids per condition so the UI replaces (not duplicates) them each
    tick. Resource breaches → CRITICAL; recurring log errors → one WARNING
    per source.
    """
    now_ms = int(time.time() * 1000)
    out: list[dict] = []

    # Swap state (used by the swap + combined-pressure checks below).
    try:
        swap = psutil.swap_memory()
        swap_pct = swap.percent
        swap_used_gb = swap.used / (1024 ** 3)
        swap_total_gb = swap.total / (1024 ** 3)
    except Exception:
        swap_pct = swap_used_gb = swap_total_gb = 0.0

    # Debug: confirm what's actually being read each vitals push (debug level so
    # it doesn't spam the live log stream every 2s).
    log.debug("anomaly scan: ram=%.1f%% cpu=%.1f%% swap=%.1f%%", ram_pct, cpu, swap_pct)

    if cpu > _CPU_CRIT:
        out.append({
            "id": "cpu-high", "title": "CPU saturated", "severity": "critical",
            "description": f"CPU at {cpu:.1f}% — threshold {_CPU_CRIT:.0f}%.",
            "source": "system", "timestamp": now_ms,
        })
    if ram_pct > _RAM_CRIT:
        out.append({
            "id": "ram-high", "title": "Memory pressure", "severity": "critical",
            "description": f"RAM at {ram_pct:.1f}% — threshold {_RAM_CRIT:.0f}%.",
            "source": "system", "timestamp": now_ms,
        })

    # Swap pressure — high swap means the system is memory-starved even when
    # RAM % looks only borderline.
    if swap_total_gb > 0 and swap_pct > _SWAP_WARN:
        crit = swap_pct > _SWAP_CRIT
        out.append({
            "id": "swap-high",
            "title": "Swap saturated" if crit else "Swap in heavy use",
            "severity": "critical" if crit else "warning",
            "description": (
                f"Swap {swap_used_gb:.1f} GB / {swap_total_gb:.0f} GB used "
                f"({swap_pct:.0f}%) — system is paging heavily."
            ),
            "source": "memory-monitor", "timestamp": now_ms,
        })

    # Combined memory pressure — RAM and swap elevated simultaneously.
    if ram_pct > _MEM_PRESSURE_RAM and swap_pct > _MEM_PRESSURE_SWAP:
        out.append({
            "id": "memory-pressure", "title": "MEMORY PRESSURE", "severity": "critical",
            "description": (
                f"RAM and swap both elevated (RAM {ram_pct:.0f}%, swap {swap_pct:.0f}%) "
                "— system is under sustained memory stress."
            ),
            "source": "memory-monitor", "timestamp": now_ms,
        })
    # every mounted partition, not just "/"
    seen_mounts: set[str] = set()
    for part in psutil.disk_partitions(all=False):
        mount = part.mountpoint
        if mount in seen_mounts:
            continue
        seen_mounts.add(mount)
        try:
            pct = psutil.disk_usage(mount).percent
        except (PermissionError, OSError):
            continue
        if pct > _DISK_CRIT:
            out.append({
                "id": f"disk-{mount}", "title": "Disk almost full", "severity": "critical",
                "description": f"{mount} at {pct:.0f}% (threshold {_DISK_CRIT:.0f}%).",
                "source": "disk", "timestamp": now_ms,
            })

    # Recent ERROR/CRITICAL logs → one WARNING per unique source.
    by_source: dict[str, int] = {}
    last_msg: dict[str, str] = {}
    for entry in _recent_logs:
        if entry.get("level") not in ("error", "critical"):
            continue
        msg = str(entry.get("msg", ""))
        source = msg.split(":", 1)[0].strip() if ":" in msg else "bantz"
        by_source[source] = by_source.get(source, 0) + 1
        last_msg[source] = msg
    for source, count in by_source.items():
        out.append({
            "id": f"log-{source}", "title": f"{count} error{'s' if count != 1 else ''} from {source}",
            "severity": "warning",
            "description": last_msg[source][:200],
            "source": source, "timestamp": now_ms,
        })

    return out


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


_DIRECTIVE_PROMPT = """Parse this directive into a scheduled job. Return JSON only:
{{
  "title": "short title",
  "schedule_type": "cron" | "interval" | "once",
  "cron_expr": "hour=7,minute=0" (if cron),
  "interval_seconds": 1800 (if interval),
  "fire_at": "ISO datetime" (if once),
  "priority": "low" | "medium" | "high" | "critical"
}}
User said: "{user_text}"
Today is {now}. If they say "every morning at 7am" -> cron hour=7,minute=0. \
If they say "every 30 minutes" -> interval 1800. If they say "at 7am tomorrow" -> once with fire_at.
Return ONLY the JSON object. No explanation, no code, no markdown fences."""


def _extract_directive_json(raw: str) -> dict | None:
    """Pull the schedule JSON out of a (possibly verbose) LLM response.

    Weak models wrap the answer in prose/code and emit multiple ``{...}``
    blocks (e.g. an empty template + the filled one), so a greedy match
    fails. Scan fenced blocks then flat objects, and return the first that
    parses to a dict with a real ``schedule_type``."""
    candidates: list[str] = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidates += re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
    fallback: dict | None = None
    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            if data.get("schedule_type"):
                return data
            if fallback is None and "schedule_type" in data:
                fallback = data
    return fallback


async def _parse_directive(text: str) -> dict | None:
    """Ask the LLM to turn a natural-language directive into a schedule spec.
    Returns the parsed dict, or None if anything fails (caller falls back)."""
    from datetime import datetime
    prompt = _DIRECTIVE_PROMPT.format(user_text=text, now=datetime.now().isoformat())
    try:
        from bantz.llm.router import get_llm
        raw = await get_llm().chat([{"role": "user", "content": prompt}])
        log.info("directive raw LLM response: %r", raw[:500])
        data = _extract_directive_json(raw)
        if data and data.get("schedule_type"):
            return data
        log.warning("directive parse: no usable JSON in response")
    except Exception as exc:
        log.debug("directive parse failed: %s", exc)
    return None


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
                "llm_provider":               config.llm_provider,
                "ollama_model":               config.ollama_model,
                "ollama_base_url":            config.ollama_base_url,
                "anthropic_api_key":          config.anthropic_api_key,
                "anthropic_model":            config.anthropic_model,
                "gemini_enabled":             config.gemini_enabled,
                "gemini_api_key":             config.gemini_api_key,
                "language":                   config.language,
                "tts_enabled":                config.tts_enabled,
                "stt_enabled":                config.stt_enabled,
                "wake_word_enabled":          config.wake_word_enabled,
                "distillation_enabled":       config.distillation_enabled,
                "shell_confirm_destructive":  config.shell_confirm_destructive,
                "observer_enabled":           config.observer_enabled,
                "verbosity":                  config.verbosity,
                "autonomy":                   config.autonomy,
                "mood_bias":                  config.mood_bias,
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
        text = f"{key}={value}\n"
    else:
        text = env_path.read_text()
        pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(f"{key}={value}", text)
        else:
            text = text.rstrip("\n") + f"\n{key}={value}\n"

    fd = os.open(str(env_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):  # unavailable on Windows
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


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
                loop.call_soon_threadsafe(self._q.put_nowait, payload)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

ws_server = WsBroadcastServer()
