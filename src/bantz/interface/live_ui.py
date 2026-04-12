"""
Bantz v3 — Rich Live TUI  (#296)

Replaces the Textual-based TUI with a **Rich Live** layout.

Architecture
────────────
•  ``Layout`` split:  header | main (stats ＋ logs) | chat
•  ``Live(layout, refresh_per_second=4)`` — 4 fps ceiling, diff-only
   terminal redraws via Rich.
•  Stats panel  — CPU / RAM / VRAM / DISK via ``psutil``, updated 2 s.
•  Log panel    — fed by ``asyncio.Queue``; newest at bottom; last 200
   lines.  Mouse-scroll via SGR escape sequences (``\\033[?1006h``).
•  Chat input   — ``aioconsole.ainput()``; non-blocking.
•  Chat output  — ``rich.markdown.Markdown`` for long / code responses.
•  Header bar   — live service-status dots (Ollama, Neo4j, Redis,
   Gemini, Telegram).

Design decisions (from issue #296 discussion)
──────────────────────────────────────────────
•  ``_dry_run`` / ``DRY_RUN`` not relevant here — this is UI-only.
•  ``call_from_thread()`` replaced entirely by ``asyncio.create_task()``.
•  All ``@work`` / ``Worker`` replaced with ``asyncio.create_task()``.
•  ``log_bus.emit(msg)``  → ``await emit_log(msg)`` or
   ``emit_log_threadsafe(msg)``  — shared global ``asyncio.Queue``.
•  No Textual imports anywhere in this module.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any

try:
    import select as _select_mod
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  # Windows / non-POSIX
    _HAS_TERMIOS = False

import psutil
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from bantz.config import config
from bantz.core.event_bus import bus, Event

logger = logging.getLogger("bantz.live_ui")


# ═══════════════════════════════════════════════════════════════════════════
# Module-level log queue — any module can ``await emit_log(msg)``
# ═══════════════════════════════════════════════════════════════════════════

_log_queue: asyncio.Queue[str] = asyncio.Queue()
_event_loop: asyncio.AbstractEventLoop | None = None


async def emit_log(msg: str) -> None:
    """Push *msg* to the log panel (call from inside the event loop)."""
    await _log_queue.put(msg)


def emit_log_threadsafe(msg: str) -> None:
    """Push *msg* to the log panel from **any** thread."""
    loop = _event_loop
    if loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(_log_queue.put_nowait, msg)


class _QueueLogHandler(logging.Handler):
    """Bridges stdlib ``logging`` → the Live UI log panel."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            emit_log_threadsafe(msg)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Service status
# ═══════════════════════════════════════════════════════════════════════════

class ServiceDot(str, Enum):
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNCONFIGURED = "unconfigured"


_DOT_STYLE: dict[ServiceDot, str] = {
    ServiceDot.UP: "[green]●[/]",
    ServiceDot.DOWN: "[red]●[/]",
    ServiceDot.DEGRADED: "[yellow]●[/]",
    ServiceDot.UNCONFIGURED: "[dim]○[/]",
}


# ═══════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ═══════════════════════════════════════════════════════════════════════════

def _bar(value: float, max_val: float = 100.0, width: int = 10) -> str:
    """Colored bar using block characters."""
    pct = min(value / max_val * 100, 100) if max_val else 0
    filled = int(pct / 100 * width)
    color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
    return f"[{color}]{'█' * filled}{'░' * (width - filled)}[/]"


# ═══════════════════════════════════════════════════════════════════════════
# Mouse scroll reader  (Linux only — SGR 1006 mode)
# ═══════════════════════════════════════════════════════════════════════════

_SGR_MOUSE_RE = re.compile(rb"\x1b\[<(\d+);\d+;\d+[Mm]")


class _MouseReader:
    """Background thread that reads SGR mouse-scroll events from stdin.

    Starts / stops mouse reporting via ``\\033[?1000h`` /
    ``\\033[?1006h`` escape sequences, then reads stdin in ``cbreak``
    mode.  Only scroll-up (button 64) and scroll-down (button 65) are
    handled — all other mouse events are silently discarded.
    """

    def __init__(self, on_scroll: Any) -> None:
        self._on_scroll = on_scroll
        self._active = False
        self._thread: threading.Thread | None = None
        self._old_attrs: list | None = None

    @property
    def active(self) -> bool:
        return self._active

    @staticmethod
    def _raw_write(data: bytes) -> None:
        """Write bytes directly to fd 1, bypassing Rich FileProxy."""
        try:
            os.write(1, data)
        except OSError:
            pass

    def start(self) -> None:
        if not _HAS_TERMIOS or self._active:
            return
        fd = sys.stdin.fileno()
        try:
            self._old_attrs = termios.tcgetattr(fd)
        except termios.error:
            return
        self._active = True
        self._raw_write(b"\033[?1000h\033[?1006h")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        self._raw_write(b"\033[?1000l\033[?1006l")
        if self._thread is not None:
            self._thread.join(timeout=0.3)
            self._thread = None
        if self._old_attrs is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs,
                )
            except (termios.error, ValueError):
                pass
            self._old_attrs = None

    def _run(self) -> None:
        fd = sys.stdin.fileno()
        try:
            tty.setcbreak(fd)
            while self._active:
                r, _, _ = _select_mod.select([fd], [], [], 0.05)
                if not r:
                    continue
                data = os.read(fd, 256)
                if data:
                    self._parse(data)
        except Exception:
            pass
        # Terminal is restored in stop()

    def _parse(self, data: bytes) -> None:
        for m in _SGR_MOUSE_RE.finditer(data):
            btn = int(m.group(1))
            if btn == 64:  # scroll up
                self._on_scroll(1)
            elif btn == 65:  # scroll down
                self._on_scroll(-1)


# ═══════════════════════════════════════════════════════════════════════════
# LiveUI — main application class
# ═══════════════════════════════════════════════════════════════════════════

class LiveUI:
    """Rich Live-based terminal UI for Bantz v3."""

    REFRESH_FPS: int = 4
    STATS_INTERVAL: float = 2.0
    LOG_MAX: int = 200
    CHAT_MAX: int = 100

    def __init__(self) -> None:
        self.console = Console()

        # ── data stores ───────────────────────────────────────────
        self._log_lines: deque[str] = deque(maxlen=self.LOG_MAX)
        self._chat_lines: deque[tuple[str, str]] = deque(maxlen=self.CHAT_MAX)

        # ── service health ────────────────────────────────────────
        self._services: dict[str, ServiceDot] = {
            "Ollama": ServiceDot.UNCONFIGURED,
            "Neo4j": ServiceDot.UNCONFIGURED,
            "Redis": ServiceDot.UNCONFIGURED,
            "Gemini": ServiceDot.UNCONFIGURED,
            "Telegram": ServiceDot.UNCONFIGURED,
        }

        # ── system stats ──────────────────────────────────────────
        self._cpu: float = 0.0
        self._ram_pct: float = 0.0
        self._ram_used_gb: float = 0.0
        self._ram_total_gb: float = 0.0
        self._disk_pct: float = 0.0
        self._disk_used_gb: float = 0.0
        self._disk_total_gb: float = 0.0
        self._vram_available: bool = False
        self._vram_pct: float = 0.0
        self._vram_used_mb: float = 0.0
        self._vram_total_mb: float = 0.0

        # ── UI state ─────────────────────────────────────────────
        self._scroll_offset: int = 0
        self._chat_scroll_offset: int = 0
        self._running: bool = True
        self._live: Live | None = None
        self._busy: bool = False
        self._streaming_text: str | None = None
        self._pending: Any = None

        # ── mouse ─────────────────────────────────────────────────
        self._mouse: _MouseReader | None = (
            _MouseReader(self._on_scroll) if _HAS_TERMIOS else None
        )

    # ── Scroll callback ───────────────────────────────────────────

    def _on_scroll(self, direction: int) -> None:
        """Adjust chat-panel scroll offset.  +1 = up, −1 = down."""
        if direction > 0:
            self._chat_scroll_offset = min(
                self._chat_scroll_offset + 3,
                max(0, len(self._chat_lines) - 1),
            )
        else:
            self._chat_scroll_offset = max(self._chat_scroll_offset - 3, 0)

    # ── Layout ────────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="chat", ratio=3),
            Layout(name="bottom", ratio=1, minimum_size=7),
        )
        layout["bottom"].split_row(
            Layout(name="stats", size=28),
            Layout(name="logs", ratio=1),
        )
        return layout

    # ── Panel renderers ───────────────────────────────────────────

    def _render_header(self) -> Panel:
        dots = "  ".join(
            f"{_DOT_STYLE[s]} {n}" for n, s in self._services.items()
        )
        now = datetime.now().strftime("%H:%M:%S")
        content = Text.from_markup(
            f"  [bold]BANTZ // OPERATIONS CENTER[/]"
            f"      {dots}  [dim]│[/]  [bold]{now}[/]"
        )
        return Panel(content, style="bold blue", height=3)

    def _render_stats(self) -> Panel:
        lines: list[str] = [
            f" [dim]CPU [/]{_bar(self._cpu)} {self._cpu:4.0f}%",
            f" [dim]RAM [/]{_bar(self._ram_pct)} "
            f"{self._ram_used_gb:.1f}/{self._ram_total_gb:.0f}G",
            f" [dim]DISK[/]{_bar(self._disk_pct)} "
            f"{self._disk_used_gb:.0f}/{self._disk_total_gb:.0f}G",
        ]
        if self._vram_available:
            lines.append(
                f" [dim]VRAM[/]{_bar(self._vram_pct)} "
                f"{self._vram_used_mb:.0f}/{self._vram_total_mb:.0f}M"
            )
        return Panel(
            Text.from_markup("\n".join(lines)),
            title="[bold cyan]SYS[/]",
            border_style="cyan",
        )

    def _render_logs(self) -> Panel:
        lines = list(self._log_lines)
        total = len(lines)
        visible_count = 15

        if self._scroll_offset > 0 and total > 0:
            end = max(0, total - self._scroll_offset)
            start = max(0, end - visible_count)
            visible = lines[start:end]
        else:
            visible = lines[-visible_count:]

        text = Text()
        for line in visible:
            try:
                text.append_text(Text.from_markup(line))
            except Exception:
                text.append(line)
            text.append("\n")

        extra = ""
        if self._scroll_offset > 0:
            extra = f" [dim](↑{self._scroll_offset})[/]"

        return Panel(
            text,
            title=f"[bold cyan]LOG STREAM[/]{extra}",
            border_style="cyan",
        )

    def _render_chat(self) -> Panel:
        parts: list[Any] = []
        all_msgs = list(self._chat_lines)
        total = len(all_msgs)
        visible_count = 40

        if self._chat_scroll_offset > 0 and total > 0:
            end = max(0, total - self._chat_scroll_offset)
            start = max(0, end - visible_count)
            recent = all_msgs[start:end]
        else:
            recent = all_msgs[-visible_count:]

        for i, (role, msg) in enumerate(recent):
            if role == "user":
                if i > 0:
                    parts.append(Text(""))
                parts.append(
                    Text.from_markup(f"[bold green]▶ You[/]  {escape(msg)}")
                )
            elif role == "bantz":
                if i > 0:
                    parts.append(Text(""))
                if "```" in msg or len(msg) > 120:
                    parts.append(Text.from_markup("[bold cyan]◆ Bantz[/]"))
                    parts.append(Markdown(msg))
                else:
                    parts.append(
                        Text.from_markup(
                            f"[bold cyan]◆ Bantz[/]  {escape(msg)}"
                        )
                    )
            elif role == "system":
                parts.append(Text.from_markup(f"  [dim]{escape(msg)}[/]"))
            elif role == "error":
                parts.append(
                    Text.from_markup(f"  [bold red]✗ {escape(msg)}[/]")
                )
            elif role == "tool":
                parts.append(
                    Text.from_markup(
                        f"  [dim magenta]⚙ \\[{escape(msg)}][/]"
                    )
                )

        if self._streaming_text is not None:
            parts.append(Text(""))
            parts.append(
                Text.from_markup(
                    f"[bold cyan]◆ Bantz[/]  {escape(self._streaming_text)}▌"
                )
            )
        elif self._busy:
            parts.append(Text("  ⟳ thinking...", style="dim cyan"))

        scroll_hint = ""
        if self._chat_scroll_offset > 0:
            scroll_hint = f" [dim](↑{self._chat_scroll_offset})[/]"

        content = Group(*parts) if parts else Text("")
        return Panel(
            content,
            title=f"[bold cyan]CHAT[/]{scroll_hint}",
            border_style="cyan",
            subtitle="[dim]scroll ↑↓ to browse history[/]" if total > visible_count else None,
            subtitle_align="right",
        )

    def _update_panels(self, layout: Layout) -> None:
        layout["header"].update(self._render_header())
        layout["bottom"]["stats"].update(self._render_stats())
        layout["bottom"]["logs"].update(self._render_logs())
        layout["chat"].update(self._render_chat())

    # ── Data helpers ──────────────────────────────────────────────

    def add_chat(self, role: str, msg: str) -> None:
        """Append a message to the chat panel."""
        self._chat_lines.append((role, msg))
        self._chat_scroll_offset = 0  # auto-scroll to newest

    def add_log(self, msg: str) -> None:
        """Append a timestamped line to the log panel (auto-scroll)."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_lines.append(f"[dim]{ts}[/] {msg}")
        self._scroll_offset = 0

    # ══════════════════════════════════════════════════════════════
    # Background tasks (all launched with asyncio.create_task)
    # ══════════════════════════════════════════════════════════════

    async def _log_consumer(self) -> None:
        """Drain the global ``_log_queue`` into ``_log_lines``."""
        while self._running:
            try:
                msg = await asyncio.wait_for(_log_queue.get(), timeout=1.0)
                self.add_log(msg)
            except asyncio.TimeoutError:
                continue

    async def _stats_collector(self) -> None:
        """Refresh CPU / RAM / DISK / VRAM every ``STATS_INTERVAL``."""
        while self._running:
            try:
                self._cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                self._ram_pct = mem.percent
                self._ram_used_gb = mem.used / (1024 ** 3)
                self._ram_total_gb = mem.total / (1024 ** 3)
                disk = psutil.disk_usage("/")
                self._disk_pct = disk.percent
                self._disk_used_gb = disk.used / (1024 ** 3)
                self._disk_total_gb = disk.total / (1024 ** 3)
                # VRAM via nvidia-smi  (best-effort)
                self._collect_vram()
            except Exception:
                pass
            await asyncio.sleep(self.STATS_INTERVAL)

    def _collect_vram(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,nounits,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) == 2:
                    used = float(parts[0].strip())
                    total = float(parts[1].strip())
                    self._vram_used_mb = used
                    self._vram_total_mb = total
                    self._vram_pct = (used / total * 100) if total else 0
                    self._vram_available = True
                    return
            self._vram_available = False
        except (FileNotFoundError, Exception):
            self._vram_available = False

    async def _panel_updater(self, layout: Layout) -> None:
        """Re-render all panels at ``REFRESH_FPS``."""
        while self._running:
            self._update_panels(layout)
            await asyncio.sleep(1 / self.REFRESH_FPS)

    async def _probe_services(self) -> None:
        """One-shot health probes for all monitored services."""
        import httpx

        # ── Ollama ────────────────────────────────────────────────
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{config.ollama_base_url}/api/tags")
                self._services["Ollama"] = (
                    ServiceDot.UP if r.status_code == 200
                    else ServiceDot.DEGRADED
                )
                self.add_log(
                    f"✓ Ollama connected → {config.ollama_model}"
                )
        except Exception:
            self._services["Ollama"] = ServiceDot.DOWN
            self.add_log(f"✗ Ollama unreachable: {config.ollama_base_url}")

        # ── Neo4j / Palace ────────────────────────────────────────
        try:
            if getattr(config, "mempalace_enabled", False):
                from bantz.memory.bridge import palace_bridge
                if palace_bridge and palace_bridge.enabled:
                    self._services["Neo4j"] = ServiceDot.UP
                else:
                    self._services["Neo4j"] = ServiceDot.DOWN
            else:
                self._services["Neo4j"] = ServiceDot.UNCONFIGURED
        except Exception:
            self._services["Neo4j"] = ServiceDot.DOWN

        # ── Redis ─────────────────────────────────────────────────
        try:
            redis_url = getattr(config, "redis_url", None)
            if redis_url:
                import redis.asyncio as aioredis
                rc = aioredis.from_url(redis_url)
                await rc.ping()
                await rc.aclose()
                self._services["Redis"] = ServiceDot.UP
            else:
                self._services["Redis"] = ServiceDot.UNCONFIGURED
        except Exception:
            self._services["Redis"] = ServiceDot.DOWN

        # ── Gemini ────────────────────────────────────────────────
        try:
            if (
                getattr(config, "gemini_enabled", False)
                and getattr(config, "gemini_api_key", None)
            ):
                async with httpx.AsyncClient(timeout=3.0) as c:
                    r = await c.get(
                        "https://generativelanguage.googleapis.com"
                        f"/v1beta/models?key={config.gemini_api_key}"
                    )
                    self._services["Gemini"] = (
                        ServiceDot.UP if r.status_code == 200
                        else ServiceDot.DOWN
                    )
            else:
                self._services["Gemini"] = ServiceDot.UNCONFIGURED
        except Exception:
            self._services["Gemini"] = ServiceDot.DOWN

        # ── Telegram ──────────────────────────────────────────────
        try:
            token = getattr(config, "telegram_bot_token", None)
            if token:
                async with httpx.AsyncClient(timeout=3.0) as c:
                    r = await c.get(
                        f"https://api.telegram.org/bot{token}/getMe"
                    )
                    self._services["Telegram"] = (
                        ServiceDot.UP if r.status_code == 200
                        else ServiceDot.DOWN
                    )
            else:
                self._services["Telegram"] = ServiceDot.UNCONFIGURED
        except Exception:
            self._services["Telegram"] = ServiceDot.DOWN

        self.add_log("Service probes complete")

    async def _warm_ollama(self) -> None:
        try:
            from bantz.llm.ollama import ollama
            await ollama.chat([{"role": "user", "content": "hi"}])
        except Exception:
            pass

    async def _enrich_greeting(self) -> None:
        try:
            from bantz.core.session import session_tracker
            from bantz.core.butler import butler
            from bantz.core.memory import memory

            session_info = session_tracker.on_launch()
            text = await butler.greet(session_info)
            if text:
                self.add_chat("bantz", text)
                try:
                    memory.add("assistant", text, tool_used="startup")
                except Exception:
                    pass
        except Exception:
            pass

    async def _scheduler_loop(self) -> None:
        """Periodic checks: reminders, morning-briefing, digest."""
        while self._running:
            # ── Reminders ─────────────────────────────────────────
            try:
                from bantz.core.scheduler import scheduler
                from bantz.core.memory import memory

                due = scheduler.check_due()
                for r in due:
                    repeat = (
                        f" (repeats {r['repeat']})"
                        if r.get("repeat", "none") != "none"
                        else ""
                    )
                    text = f"⏰ Reminder: {r['title']}{repeat}"
                    self.add_chat("bantz", text)
                    try:
                        memory.add("assistant", text, tool_used="reminder")
                    except Exception:
                        pass
            except Exception:
                pass

            # ── Morning briefing ──────────────────────────────────
            try:
                from bantz.personality.greeting import greeting_manager

                text = await greeting_manager.morning_briefing_if_due()
                if text:
                    self.add_chat("bantz", text)
            except Exception:
                pass

            # ── Digest ────────────────────────────────────────────
            try:
                from bantz.core.digest import digest_manager

                for fn in (
                    digest_manager.daily_if_due,
                    digest_manager.weekly_if_due,
                ):
                    text = await fn()
                    if text:
                        self.add_chat("bantz", text)
            except Exception:
                pass

            await asyncio.sleep(60)

    # ══════════════════════════════════════════════════════════════
    # Event-bus integration
    # ══════════════════════════════════════════════════════════════

    def _subscribe_bus(self) -> None:
        bus.bind_loop()
        bus.on("voice_input", self._on_bus_voice_input)
        bus.on("health_alert", self._on_bus_health_alert)
        bus.on("thinking_start", self._on_bus_thinking_start)
        bus.on("thinking_done", self._on_bus_thinking_done)
        bus.on("planner_step", self._on_bus_planner_step)
        bus.on("pre_tool_message", self._on_bus_pre_tool)
        bus.on("ghost_loop_listening", self._on_bus_ghost_listening)
        bus.on("ghost_loop_idle", self._on_bus_ghost_idle)
        bus.on("stt_model_ready", self._on_bus_stt_ready)
        bus.on("stt_model_failed", self._on_bus_stt_failed)
        bus.on("delegation_start", self._on_bus_delegation_start)
        bus.on("delegation_done", self._on_bus_delegation_done)
        logger.debug("EventBus → LiveUI bridge active")

    def _on_bus_voice_input(self, event: Event) -> None:
        text = event.data.get("text", "").strip()
        if text and not self._busy:
            self.add_chat("user", f"🎤 {text}")
            asyncio.create_task(self._process_input(text))

    def _on_bus_health_alert(self, event: Event) -> None:
        title = event.data.get("title", "Health Alert")
        self.add_chat("system", f"⚠️ {title}")

    def _on_bus_thinking_start(self, event: Event) -> None:
        self._busy = True

    def _on_bus_thinking_done(self, event: Event) -> None:
        self._busy = False

    def _on_bus_planner_step(self, event: Event) -> None:
        step = event.data.get("step", 0)
        total = event.data.get("total", 0)
        desc = event.data.get("description", "")[:60]
        status = event.data.get("status", "start")
        if status == "done":
            self.add_log(f"✓ Step {step}/{total}: {desc}")
        elif status == "failed":
            self.add_log(f"✗ Step {step}/{total}: {desc}")
        else:
            self.add_log(f"⚙ Step {step}/{total}: {desc}")

    def _on_bus_pre_tool(self, event: Event) -> None:
        msg = event.data.get("message", "")
        if msg:
            self.add_chat("bantz", msg)

    def _on_bus_ghost_listening(self, event: Event) -> None:
        self.add_log("🎙️  Listening…")

    def _on_bus_ghost_idle(self, event: Event) -> None:
        pass  # no visual change needed

    def _on_bus_stt_ready(self, event: Event) -> None:
        self.add_chat("system", "✅ Speech recognition ready.")

    def _on_bus_stt_failed(self, event: Event) -> None:
        error = event.data.get("error", "unknown")
        self.add_chat("error", f"Speech recognition failed: {error}")

    def _on_bus_delegation_start(self, event: Event) -> None:
        name = event.data.get("display_name", "Agent")
        task = event.data.get("task", "")[:60]
        self.add_log(f"🤖 [bold cyan]{name}[/] agent started: {task}")

    def _on_bus_delegation_done(self, event: Event) -> None:
        name = event.data.get("display_name", "Agent")
        ok = event.data.get("success", False)
        dur = event.data.get("duration_s", 0)
        tools = event.data.get("tools_used", [])
        if ok:
            t_str = f" (tools: {', '.join(tools)})" if tools else ""
            self.add_log(f"✅ [bold green]{name}[/] finished in {dur}s{t_str}")
        else:
            err = event.data.get("error", "unknown")[:60]
            self.add_log(f"❌ [bold red]{name}[/] failed: {err}")

    # ══════════════════════════════════════════════════════════════
    # Chat loop
    # ══════════════════════════════════════════════════════════════

    async def _chat_loop(self) -> None:
        import aioconsole

        self.add_chat("system", "Bantz v3 started.")
        self.add_chat("system", f"Model: {config.ollama_model}")
        self.add_chat("system", "─" * 38)

        try:
            from bantz.core.time_context import time_ctx

            self.add_chat("bantz", time_ctx.greeting_line())
        except Exception:
            pass

        while self._running:
            # Pause UI + mouse for text input
            if self._mouse:
                self._mouse.stop()
            if self._live:
                self._live.stop()

            try:
                text = await aioconsole.ainput("› ")
            except (EOFError, KeyboardInterrupt):
                self._running = False
                return
            finally:
                if self._live:
                    self._live.start()
                if self._mouse:
                    self._mouse.start()

            text = text.strip()
            if not text:
                continue

            self.add_chat("user", text)
            await self._process_input(text)

    async def _process_input(self, text: str) -> None:
        from bantz.core.brain import brain
        from bantz.core.memory import memory

        # Pending confirmation?
        if self._pending is not None:
            await self._handle_confirm(text)
            return

        self._busy = True
        try:
            result = await brain.process(text)
        except Exception as exc:
            self._busy = False
            self.add_chat("error", f"{type(exc).__name__}: {exc}")
            return

        # ── Streaming response ────────────────────────────────────
        if result.stream is not None:
            if result.tool_used:
                self.add_chat("tool", result.tool_used)

            accumulated = ""
            try:
                async for token in result.stream:
                    accumulated += token
                    self._streaming_text = accumulated
            except Exception as exc:
                self._streaming_text = None
                self._busy = False
                self.add_chat("error", f"Stream error: {exc}")
                return

            self._streaming_text = None
            self._busy = False
            self.add_chat("bantz", accumulated)

            try:
                from bantz.core.finalizer import strip_markdown

                cleaned = strip_markdown(accumulated)
                memory.add("assistant", cleaned, tool_used=result.tool_used)
            except Exception:
                pass
            try:
                await brain._graph_store(text, accumulated, result.tool_used)
            except Exception:
                pass
            return

        # ── Non-streaming response ────────────────────────────────
        self._busy = False

        if result.needs_confirm:
            self._pending = result
            self.add_chat("bantz", result.response)
        elif result.response and result.response.strip():
            if result.tool_used:
                self.add_chat("tool", result.tool_used)
            self.add_chat("bantz", result.response)
        else:
            self.add_chat(
                "system",
                "🤔 I processed your message but had nothing to say.",
            )

    async def _handle_confirm(self, text: str) -> None:
        pending = self._pending
        self._pending = None
        confirmed = text.lower().strip() in (
            "yes", "y", "ok", "evet", "e", "tamam",
        )
        if not confirmed:
            self.add_chat("system", "Cancelled.")
            return

        self._busy = True
        try:
            from bantz.core.brain import brain

            if pending.pending_tool and pending.pending_args:
                from bantz.tools import registry

                tool = registry.get(pending.pending_tool)
                if tool:
                    tr = await tool.execute(**pending.pending_args)
                    self.add_chat("tool", pending.pending_tool)
                    self.add_chat(
                        "bantz",
                        tr.output if tr.success else f"Error: {tr.error}",
                    )
                    self._busy = False
                    return

            result = await brain.process(
                pending.pending_command, confirmed=True,
            )
            if result.tool_used:
                self.add_chat("tool", result.tool_used)
            self.add_chat("bantz", result.response)
        except Exception as exc:
            self.add_chat("error", f"{type(exc).__name__}: {exc}")
        self._busy = False

    # ══════════════════════════════════════════════════════════════
    # Main entry
    # ══════════════════════════════════════════════════════════════

    async def run(self) -> None:
        global _event_loop
        _event_loop = asyncio.get_running_loop()

        layout = self._build_layout()
        self._update_panels(layout)

        # Install log handler
        handler = _QueueLogHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logging.getLogger("bantz").addHandler(handler)

        # Event bus
        self._subscribe_bus()

        # Background tasks
        tasks = [
            asyncio.create_task(self._log_consumer()),
            asyncio.create_task(self._stats_collector()),
            asyncio.create_task(self._panel_updater(layout)),
            asyncio.create_task(self._scheduler_loop()),
        ]
        asyncio.create_task(self._probe_services())
        asyncio.create_task(self._warm_ollama())
        asyncio.create_task(self._enrich_greeting())

        try:
            with Live(
                layout,
                console=self.console,
                refresh_per_second=self.REFRESH_FPS,
                screen=True,
            ) as live:
                self._live = live
                if self._mouse:
                    self._mouse.start()
                await self._chat_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            if self._mouse:
                self._mouse.stop()
            for t in tasks:
                t.cancel()
            try:
                await bus.shutdown()
            except Exception:
                pass
            try:
                logging.getLogger("bantz").removeHandler(handler)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# Module entry point
# ═══════════════════════════════════════════════════════════════════════════

def run() -> None:
    """Launch the Bantz Rich Live TUI."""
    ui = LiveUI()
    asyncio.run(ui.run())
