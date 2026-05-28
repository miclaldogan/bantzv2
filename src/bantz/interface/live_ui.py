"""
Bantz v3 — Rich Live TUI

Design
──────
• Single ``Live`` context that never stops while bantz runs.
• ``auto_refresh=False`` — panels are refreshed explicitly so the
  terminal is never redrawn while the user is typing.
• User input via a daemon thread calling ``sys.stdin.readline()``
  (blocking, instant terminal echo) → pushed to an asyncio queue via
  ``loop.call_soon_threadsafe``.
• After the user presses Enter the "› prompt" line is erased with a
  cursor-escape sequence before the next Live refresh so the render
  position stays correct.
• ``screen=False, transient=False`` — TUI renders inline and stays
  visible after bantz exits.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any

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
    """Colored ASCII bar: green < 60 %, yellow < 85 %, red otherwise."""
    pct = min(value / max_val * 100, 100) if max_val else 0
    filled = int(pct / 100 * width)
    color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
    return f"[{color}]{'█' * filled}{'░' * (width - filled)}[/]"


# ═══════════════════════════════════════════════════════════════════════════
# Compat stubs — kept so existing tests / imports continue to work
# ═══════════════════════════════════════════════════════════════════════════

_SGR_MOUSE_RE = re.compile(rb"\x1b\[<(\d+);\d+;\d+[Mm]")


class _MouseReader:
    """Scroll-event stub (kept for tests; not used in main UI flow)."""

    def __init__(self, on_scroll: Any) -> None:
        self._on_scroll = on_scroll
        self._active = False
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def _parse(self, data: bytes) -> None:
        for m in _SGR_MOUSE_RE.finditer(data):
            btn = int(m.group(1))
            if btn == 64:
                self._on_scroll(1)
            elif btn == 65:
                self._on_scroll(-1)


class _StdinReader:
    """Unified stdin stub (kept for tests; not used in main UI flow)."""

    def __init__(
        self, on_scroll: Any, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._on_scroll = on_scroll
        self._loop = loop
        self._active = False
        self._buf: str = ""
        self.line_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def buf(self) -> str:
        return self._buf

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# LiveUI
# ═══════════════════════════════════════════════════════════════════════════

class LiveUI:
    """Rich Live-based terminal UI for Bantz v3.

    The Live context is opened once in ``run()`` and never stopped.
    User input is read by a daemon thread (``sys.stdin.readline()``),
    giving instant terminal echo with zero lag.  Lines are forwarded
    to an asyncio queue via ``call_soon_threadsafe``.

    While waiting for input, auto-refresh is suspended so that the
    terminal cursor stays below the panels and the display is stable.
    After the user presses Enter a cursor-escape clears the prompt
    line, then the panels resume updating.
    """

    REFRESH_FPS: int = 4
    STATS_INTERVAL: float = 2.0
    LOG_MAX: int = 200
    CHAT_MAX: int = 100

    def __init__(self) -> None:
        self.console = Console()

        # ── data ──────────────────────────────────────────────────
        self._log_lines: deque[str] = deque(maxlen=self.LOG_MAX)
        self._chat_lines: deque[tuple[str, str]] = deque(maxlen=self.CHAT_MAX)

        # ── service dots ──────────────────────────────────────────
        self._services: dict[str, ServiceDot] = {
            "Ollama": ServiceDot.UNCONFIGURED,
            "MemPalace": ServiceDot.UNCONFIGURED,  # #437
            "TTS": ServiceDot.UNCONFIGURED,         # #437
            "STT": ServiceDot.UNCONFIGURED,         # #437
            "Voice": ServiceDot.UNCONFIGURED,       # #437
            "Neo4j": ServiceDot.UNCONFIGURED,
            "Redis": ServiceDot.UNCONFIGURED,
            "Gemini": ServiceDot.UNCONFIGURED,
            "Telegram": ServiceDot.UNCONFIGURED,
        }

        # ── status bar info (#437) ─────────────────────────────────
        self._memory_count: int = -1      # -1 = not yet fetched
        self._persona_state: str = ""     # from AffinityEngine

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

        # ── UI state ──────────────────────────────────────────────
        self._scroll_offset: int = 0
        self._chat_scroll_offset: int = 0
        self._running: bool = True
        self._live: Live | None = None
        self._layout: Layout | None = None
        self._busy: bool = False
        self._streaming_text: str | None = None
        self._pending: Any = None

        # ── input pipeline: thread → asyncio ──────────────────────
        self._input_queue: asyncio.Queue[str | None] = asyncio.Queue()
        # True while we are waiting for user input (suppress auto-refresh)
        self._waiting_input: bool = False

    # ─────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=5),   # #437: was 3; now shows dots + info
            Layout(name="chat", ratio=3),
            Layout(name="bottom", ratio=1, minimum_size=7),
        )
        layout["bottom"].split_row(
            Layout(name="stats", size=28),
            Layout(name="logs", ratio=1),
        )
        return layout

    # ─────────────────────────────────────────────────────────────
    # Scroll (for external callers / tests)
    # ─────────────────────────────────────────────────────────────

    def _on_scroll(self, direction: int) -> None:
        if direction > 0:
            self._chat_scroll_offset = min(
                self._chat_scroll_offset + 3,
                max(0, len(self._chat_lines) - 1),
            )
        else:
            self._chat_scroll_offset = max(self._chat_scroll_offset - 3, 0)

    # ─────────────────────────────────────────────────────────────
    # Panel renderers
    # ─────────────────────────────────────────────────────────────

    def _render_header(self) -> Panel:
        dots = "  ".join(
            f"{_DOT_STYLE[s]} {n}" for n, s in self._services.items()
        )
        now = datetime.now().strftime("%H:%M:%S")
        # ── info line: model name, persona state, memory drawer count (#437) ──
        info_parts: list[str] = [
            f"[dim]model:[/][bold cyan]{config.ollama_model}[/]",
        ]
        if self._memory_count >= 0:
            info_parts.append(
                f"[dim]mem:[/][bold]{self._memory_count}[/]"
            )
        if self._persona_state:
            info_parts.append(
                f"[dim]persona:[/][bold]{self._persona_state}[/]"
            )
        info_parts.append(f"[bold]{now}[/]")
        info_line = "  [dim]│[/]  ".join(info_parts)
        content = Text.from_markup(
            f"  [bold]BANTZ // OPERATIONS CENTER[/]\n"
            f"  {dots}\n"
            f"  {info_line}"
        )
        return Panel(content, style="bold blue", height=5)

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

        return Panel(
            text,
            title="[bold cyan]LOG STREAM[/]",
            border_style="cyan",
        )

    def _render_chat(self) -> Panel:
        parts: list[Any] = []
        all_msgs = list(self._chat_lines)
        total = len(all_msgs)

        # Dynamically limit visible messages so newest ones (bottom of Group)
        # are never clipped by Rich. header=3, bottom≈term_h//4 (min 7),
        # panel borders=2, ~2 lines per message on average.
        term_h = self.console.height or 24
        bottom_h = max(7, (term_h - 3) // 4)
        chat_h = max(5, term_h - 3 - bottom_h - 2)
        visible_count = max(4, chat_h // 2)

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

        content = Group(*parts) if parts else Text("")
        return Panel(
            content,
            title="[bold cyan]CHAT[/]",
            border_style="cyan",
        )

    def _update_panels(self, layout: Layout) -> None:
        layout["header"].update(self._render_header())
        layout["bottom"]["stats"].update(self._render_stats())
        layout["bottom"]["logs"].update(self._render_logs())
        layout["chat"].update(self._render_chat())

    # ─────────────────────────────────────────────────────────────
    # Data helpers
    # ─────────────────────────────────────────────────────────────

    def add_chat(self, role: str, msg: str) -> None:
        self._chat_lines.append((role, msg))
        self._chat_scroll_offset = 0

    def add_log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_lines.append(f"[dim]{ts}[/] {msg}")
        self._scroll_offset = 0

    # ─────────────────────────────────────────────────────────────
    # Input thread
    # ─────────────────────────────────────────────────────────────

    def _start_input_thread(
        self, loop: asyncio.AbstractEventLoop
    ) -> threading.Thread:
        """Spawn a daemon thread that reads lines from stdin and forwards
        them to the asyncio input queue via call_soon_threadsafe."""

        def _reader() -> None:
            while self._running:
                try:
                    raw = sys.stdin.readline()
                    if not raw:  # EOF (Ctrl+D)
                        loop.call_soon_threadsafe(
                            self._input_queue.put_nowait, None
                        )
                        break
                    loop.call_soon_threadsafe(
                        self._input_queue.put_nowait, raw.rstrip("\n")
                    )
                except (EOFError, KeyboardInterrupt):
                    loop.call_soon_threadsafe(
                        self._input_queue.put_nowait, None
                    )
                    break
                except Exception:
                    pass

        t = threading.Thread(target=_reader, daemon=True, name="bantz-input")
        t.start()
        return t

    def _erase_prompt_line(self, typed: str) -> None:
        """After readline() returns, the terminal has echoed '› {typed}\\n'
        and moved the cursor one line below the live-render area.

        We move the cursor back up by the number of lines that text
        occupied, then clear to end-of-screen so the next live.refresh()
        starts from the correct position without any residual prompt text.
        """
        try:
            width = self.console.width or 80
            # "› " is 2 chars; typed text follows on the same line
            chars = 2 + len(typed)
            lines_used = max(1, (chars + width - 1) // width)
            # Go up lines_used lines, go to column 0, clear to end of screen
            os.write(1, f"\033[{lines_used}A\r\033[J".encode())
        except OSError:
            pass

    # ─────────────────────────────────────────────────────────────
    # Background tasks
    # ─────────────────────────────────────────────────────────────

    async def _log_consumer(self) -> None:
        """Drain the global log queue into _log_lines."""
        while self._running:
            try:
                msg = await asyncio.wait_for(_log_queue.get(), timeout=1.0)
                self.add_log(msg)
            except asyncio.TimeoutError:
                continue

    async def _stats_collector(self) -> None:
        """Refresh CPU / RAM / DISK / VRAM every STATS_INTERVAL seconds."""
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
        """Re-render all panels at REFRESH_FPS.

        Suspended while _waiting_input is True so that the terminal
        cursor stays in place below the panels as the user types.
        """
        while self._running:
            if not self._waiting_input and self._live is not None:
                try:
                    self._update_panels(layout)
                    self._live.refresh()
                except Exception:
                    pass
            await asyncio.sleep(1 / self.REFRESH_FPS)

    def _refresh_now(self, layout: Layout) -> None:
        """Convenience: update panels and force a single Live refresh."""
        if self._live is not None:
            try:
                self._update_panels(layout)
                self._live.refresh()
            except Exception:
                pass

    async def _probe_services(self) -> None:
        """One-shot health probes for all monitored services."""
        import httpx

        async def check_ollama(client: httpx.AsyncClient) -> None:
            try:
                r = await client.get(f"{config.ollama_base_url}/api/tags")
                self._services["Ollama"] = (
                    ServiceDot.UP if r.status_code == 200
                    else ServiceDot.DEGRADED
                )
                self.add_log(f"✓ Ollama connected → {config.ollama_model}")
            except Exception:
                self._services["Ollama"] = ServiceDot.DOWN
                self.add_log(
                    f"✗ Ollama unreachable: {config.ollama_base_url}"
                )
                self.add_chat(
                    "error",
                    f"Cannot reach Ollama at {config.ollama_base_url}. "
                    f"Start it with: ollama serve  "
                    f"(model needed: {config.ollama_model})",
                )

        async def check_mempalace() -> None:  # #437
            try:
                if getattr(config, "mempalace_enabled", False):
                    from bantz.memory.bridge import palace_bridge
                    if not palace_bridge.enabled:
                        await palace_bridge.init()
                    self._services["MemPalace"] = (
                        ServiceDot.UP if palace_bridge.enabled
                        else ServiceDot.DOWN
                    )
                else:
                    self._services["MemPalace"] = ServiceDot.UNCONFIGURED
            except Exception:
                self._services["MemPalace"] = ServiceDot.DOWN

        async def check_tts() -> None:  # #437
            try:
                if getattr(config, "tts_enabled", False):
                    from bantz.agent.tts import tts_engine as _tts
                    self._services["TTS"] = (
                        ServiceDot.UP if _tts.available()
                        else ServiceDot.DOWN
                    )
                else:
                    self._services["TTS"] = ServiceDot.UNCONFIGURED
            except Exception:
                self._services["TTS"] = ServiceDot.DOWN

        async def check_stt() -> None:  # #437
            try:
                if getattr(config, "stt_enabled", False):
                    try:
                        import faster_whisper  # noqa: F401
                        self._services["STT"] = ServiceDot.UP
                    except ImportError:
                        self._services["STT"] = ServiceDot.DOWN
                else:
                    self._services["STT"] = ServiceDot.UNCONFIGURED
            except Exception:
                self._services["STT"] = ServiceDot.DOWN

        async def check_voice() -> None:  # #437
            try:
                voice_on = getattr(config, "voice_enabled", False)
                any_voice = any([
                    getattr(config, attr, False)
                    for attr in (
                        "tts_enabled", "wake_word_enabled",
                        "stt_enabled", "ghost_loop_enabled",
                    )
                ])
                if voice_on or any_voice:
                    up = (
                        self._services["TTS"] == ServiceDot.UP
                        or self._services["STT"] == ServiceDot.UP
                    )
                    self._services["Voice"] = (
                        ServiceDot.UP if up else ServiceDot.DEGRADED
                    )
                else:
                    self._services["Voice"] = ServiceDot.UNCONFIGURED
            except Exception:
                self._services["Voice"] = ServiceDot.DOWN

        async def check_neo4j() -> None:
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

        async def check_redis() -> None:
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

        async def check_gemini(client: httpx.AsyncClient) -> None:
            try:
                if (
                    getattr(config, "gemini_enabled", False)
                    and getattr(config, "gemini_api_key", None)
                ):
                    r = await client.get(
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

        async def check_telegram(client: httpx.AsyncClient) -> None:
            try:
                token = getattr(config, "telegram_bot_token", None)
                if token:
                    r = await client.get(
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

        async with httpx.AsyncClient(timeout=3.0) as client:
            await asyncio.gather(
                check_ollama(client),
                check_mempalace(),
                check_tts(),
                check_stt(),
                check_neo4j(),
                check_redis(),
                check_gemini(client),
                check_telegram(client),
            )
        # voice depends on TTS/STT state — run after the gather
        await check_voice()
        self.add_log("Service probes complete")

    async def _warm_ollama(self) -> None:
        try:
            from bantz.llm.ollama import ollama
            await ollama.chat([{"role": "user", "content": "hi"}])
        except Exception:
            pass

    async def _service_poller(self) -> None:
        """Re-run health probes every 30 seconds (#437)."""
        while self._running:
            await asyncio.sleep(30)
            if not self._running:
                break
            try:
                await self._probe_services()
            except Exception:
                pass

    async def _status_updater(self) -> None:
        """Refresh memory count + persona state every 30 seconds (#437)."""
        while self._running:
            try:
                from bantz.core.memory import memory as _mem
                stats = _mem.stats()
                self._memory_count = stats.get("total_conversations", 0)
            except Exception:
                pass
            try:
                from bantz.agent.affinity_engine import affinity_engine
                state = affinity_engine.get_persona_state()
                # Keep it short — strip the period and take first 20 chars
                self._persona_state = state.rstrip(".")[:20] if state else ""
            except Exception:
                pass
            await asyncio.sleep(30)

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
        """Check reminders, morning briefing, and digests every 60 s."""
        while self._running:
            try:
                from bantz.core.scheduler import scheduler
                from bantz.core.memory import memory

                for r in scheduler.check_due():
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

            try:
                from bantz.personality.greeting import greeting_manager
                text = await greeting_manager.morning_briefing_if_due()
                if text:
                    self.add_chat("bantz", text)
            except Exception:
                pass

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

    # ─────────────────────────────────────────────────────────────
    # Event-bus integration
    # ─────────────────────────────────────────────────────────────

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
        pass

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

    # ─────────────────────────────────────────────────────────────
    # Chat loop — the main interactive loop
    # ─────────────────────────────────────────────────────────────

    async def _chat_loop(self) -> None:
        assert self._layout is not None
        layout = self._layout

        loop = asyncio.get_running_loop()
        self._start_input_thread(loop)

        self.add_chat("system", "Bantz v3 started.")
        self.add_chat("system", f"Model: {config.ollama_model}")
        self.add_chat("system", "─" * 38)

        # ── First-run welcome banner ───────────────────────────────
        try:
            from bantz.memory.onboarding import is_onboarding_done
            from pathlib import Path as _Path
            _palace_parent = str(_Path(config.resolved_palace_path).parent)
            if not is_onboarding_done(_palace_parent):
                self.add_chat("system", "")
                self.add_chat("system", "╔══ Welcome to Bantz — First Run ══╗")
                self.add_chat("system", "  ✅  Text chat (English & Turkish)")
                self.add_chat("system", "  ✅  Turkish translation (MarianMT)")
                tts_ok = getattr(config, "tts_enabled", False)
                self.add_chat("system",
                    "  ✅  TTS voice responses" if tts_ok
                    else "  ⚙   TTS (set BANTZ_TTS_ENABLED=true to enable)")
                self.add_chat("system", "  ⚙   Voice input — needs setup")
                self.add_chat("system", "╚══════════════════════════════════╝")
                self.add_chat("system",
                    "Run [bold]bantz --setup onboarding[/] to personalise Bantz "
                    "and confirm Ollama is running.")
        except Exception:
            pass

        try:
            from bantz.core.time_context import time_ctx
            self.add_chat("bantz", time_ctx.greeting_line())
        except Exception:
            pass

        while self._running:
            # ── 1. Render current state and pause auto-refresh ────
            self._waiting_input = True
            self._refresh_now(layout)

            # ── 2. Print "› " prompt below the panels ─────────────
            sys.stdout.write("› ")
            sys.stdout.flush()

            # ── 3. Wait for a line from the input thread ──────────
            try:
                text = await self._input_queue.get()
            except asyncio.CancelledError:
                self._running = False
                return

            # ── 4. Erase the "› {text}\n" line so the next refresh ─
            #       lands at the correct cursor position
            self._erase_prompt_line(text or "")
            self._waiting_input = False

            if text is None:  # EOF / Ctrl+D
                self._running = False
                return

            text = text.strip()
            if not text:
                continue

            # ── 5. Show user message + thinking indicator ─────────
            self.add_chat("user", text)
            self._busy = True
            self._refresh_now(layout)

            # ── 6. Process ────────────────────────────────────────
            try:
                await self._process_input(text)
            except asyncio.CancelledError:
                self._running = False
                return
            except Exception as exc:
                self._busy = False
                self._streaming_text = None
                log.error("Chat loop error: %s", exc)
                self.add_chat("error", "I'm afraid I encountered a slight mechanical difficulty, ma'am.")

    # ─────────────────────────────────────────────────────────────
    # Process user input
    # ─────────────────────────────────────────────────────────────

    async def _process_input(self, text: str) -> None:
        assert self._layout is not None
        layout = self._layout

        from bantz.core.brain import brain
        from bantz.core.memory import memory

        if self._pending is not None:
            await self._handle_confirm(text)
            return

        self._busy = True
        try:
            result = await brain.process(text)
        except Exception as exc:
            self._busy = False
            log.error("Process input error: %s", exc)
            self.add_chat("error", "I'm afraid I encountered a slight mechanical difficulty, ma'am.")
            return

        # ── Streaming ─────────────────────────────────────────────
        if result.stream is not None:
            if result.tool_used:
                self.add_chat("tool", result.tool_used)

            accumulated = ""
            try:
                async for token in result.stream:
                    accumulated += token
                    self._streaming_text = accumulated
                    # Each token triggers an immediate panel refresh so
                    # the user sees text appear in real time.
                    self._refresh_now(layout)
            except Exception as exc:
                self._streaming_text = None
                self._busy = False
                log.error("Stream error: %s", exc)
                self.add_chat("error", "I'm afraid the stream encountered a slight difficulty, ma'am.")
                return

            self._streaming_text = None
            self._busy = False

            if accumulated.strip():
                self.add_chat("bantz", accumulated)
            else:
                self.add_chat(
                    "error",
                    "Empty response from model — is Ollama running and "
                    f"is '{config.ollama_model}' pulled? "
                    "Try: ollama pull " + config.ollama_model,
                )

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

        # ── Non-streaming ─────────────────────────────────────────
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
                pending.pending_command, confirmed=True
            )
            if result.tool_used:
                self.add_chat("tool", result.tool_used)
            self.add_chat("bantz", result.response)
        except Exception as exc:
            self.add_chat("error", f"{type(exc).__name__}: {exc}")
        self._busy = False

    # ─────────────────────────────────────────────────────────────
    # Main entry
    # ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        global _event_loop
        _event_loop = asyncio.get_running_loop()

        self._layout = self._build_layout()
        self._update_panels(self._layout)

        # Silence stray log writes to stderr while Live owns the terminal
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            root_logger.addHandler(logging.NullHandler())

        handler = _QueueLogHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        bantz_logger = logging.getLogger("bantz")
        bantz_logger.addHandler(handler)
        bantz_logger.propagate = False

        self._subscribe_bus()

        tasks = [
            asyncio.create_task(self._log_consumer()),
            asyncio.create_task(self._stats_collector()),
            asyncio.create_task(self._panel_updater(self._layout)),
            asyncio.create_task(self._scheduler_loop()),
            asyncio.create_task(self._service_poller()),   # #437: re-probe every 30s
            asyncio.create_task(self._status_updater()),   # #437: refresh model/persona/mem
        ]
        asyncio.create_task(self._probe_services())
        asyncio.create_task(self._warm_ollama())
        asyncio.create_task(self._enrich_greeting())
        asyncio.create_task(self._start_ws())
        asyncio.create_task(self._start_ambient_sampler())

        try:
            with Live(
                self._layout,
                console=self.console,
                refresh_per_second=self.REFRESH_FPS,
                screen=False,
                transient=False,
                auto_refresh=False,
            ) as live:
                self._live = live
                await self._chat_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            for t in tasks:
                t.cancel()
            try:
                await bus.shutdown()
            except Exception:
                pass
            try:
                bantz_logger.removeHandler(handler)
                bantz_logger.propagate = True
            except Exception:
                pass

    async def _start_ws(self) -> None:
        try:
            from bantz.interface.ws_server import ws_server
            await ws_server.start()
        except Exception as exc:
            logger.warning("WS server failed to start: %s", exc)

    async def _start_ambient_sampler(self) -> None:
        """Start the standalone ambient sampler when wake word is disabled (#441)."""
        try:
            from bantz.agent.ambient import maybe_start_standalone
            maybe_start_standalone()
        except Exception as exc:
            logger.warning("Ambient sampler failed to start: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Module entry point
# ═══════════════════════════════════════════════════════════════════════════

def run() -> None:
    """Launch the Bantz Rich Live TUI."""
    ui = LiveUI()
    asyncio.run(ui.run())
