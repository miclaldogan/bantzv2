"""
Bantz — Operations Header (#136)

Custom header replacing Textual's default Header widget.
Two-row command center showing live service indicators, uptime, mood,
model name, memory count, and session count.

Visual:
  ╭─── BANTZ // OPERATIONS CENTER ────── (◕‿◕) chill ───╮
  │ ● Ollama  ● Palace  ● Gemini  ● Telegram │ qwen3:8b  │
  │ Uptime: 3h 42m │ Memory: 847 │ Sessions: 4          │
  ╰──────────────────────────────────────────────────────╯

Architecture:
  - Health checks are EVENT-DRIVEN: startup probe + error/success tracking
  - No periodic API pinging (zero-cost health)
  - DB counts fetched once at startup, updated via Textual messages
  - All network probes use @work(thread=True) + timeout=3.0
"""
from __future__ import annotations

import logging
import time
from enum import Enum

from textual import work
from textual.app import ComposeResult
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from bantz.config import config

log = logging.getLogger("bantz.header")


# ═══════════════════════════════════════════════════════════════════════════
# Service status enum
# ═══════════════════════════════════════════════════════════════════════════

class ServiceStatus(str, Enum):
    UP = "up"          # ● green  — responding
    DEGRADED = "degraded"  # ● yellow — slow / warning
    DOWN = "down"      # ● red    — unreachable
    UNCONFIGURED = "unconfigured"  # ○ gray — not set up

STATUS_DOTS: dict[ServiceStatus, str] = {
    ServiceStatus.UP:           "[#00ff88]●[/]",
    ServiceStatus.DEGRADED:     "[#ffaa00]●[/]",
    ServiceStatus.DOWN:         "[#ff4444]●[/]",
    ServiceStatus.UNCONFIGURED: "[#666666]○[/]",
}


# ═══════════════════════════════════════════════════════════════════════════
# Messages — event-driven health updates
# ═══════════════════════════════════════════════════════════════════════════

class ServiceHealthChanged(Message):
    """Fired when a service status changes (success/failure in brain or probe)."""

    def __init__(self, service: str, status: ServiceStatus, detail: str = "") -> None:
        super().__init__()
        self.service = service
        self.status = status
        self.detail = detail


class MemoryCountUpdated(Message):
    """Fired when conversation/message count changes."""

    def __init__(self, total_messages: int, total_sessions: int) -> None:
        super().__init__()
        self.total_messages = total_messages
        self.total_sessions = total_sessions


class VoiceStatusChanged(Message):
    """Fired to update the transient voice pipeline status in the header.

    Pass an empty string to clear the indicator (idle state).
    """

    def __init__(self, status: str) -> None:
        super().__init__()
        self.status = status


# ═══════════════════════════════════════════════════════════════════════════
# Operations Header Widget
# ═══════════════════════════════════════════════════════════════════════════

class OperationsHeader(Static):
    """Two-row command center header with live service dots.

    Health is event-driven:
      - On startup: probe each service once
      - After that: brain.py / integrations fire ServiceHealthChanged
        on every success/failure → header updates the dot
      - No periodic polling — zero API cost
    """

    # Reactive state
    ollama_status: reactive[ServiceStatus] = reactive(ServiceStatus.UNCONFIGURED)
    palace_status: reactive[ServiceStatus] = reactive(ServiceStatus.UNCONFIGURED)
    gemini_status: reactive[ServiceStatus] = reactive(ServiceStatus.UNCONFIGURED)
    telegram_status: reactive[ServiceStatus] = reactive(ServiceStatus.UNCONFIGURED)
    uptime_text: reactive[str] = reactive("0m")
    memory_count: reactive[int] = reactive(0)
    session_count: reactive[int] = reactive(0)
    # Transient voice pipeline status — cleared after voice_input is processed
    voice_status: reactive[str] = reactive("")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._boot_time = time.monotonic()

    def compose(self) -> ComposeResult:
        return []  # Static renders via render()

    def on_mount(self) -> None:
        # ── Startup probes (one-time, background threaded) ────────
        self._probe_ollama()
        self._probe_palace()
        self._probe_gemini()
        self._probe_telegram()
        self._fetch_db_counts()

        # ── Uptime ticker — every 60s ────────────────────────────
        self.set_interval(60, self._tick_uptime)
        self._tick_uptime()  # first render

    # ── Rendering ─────────────────────────────────────────────────

    def render(self) -> str:
        from bantz.interface.tui.mood import mood_machine, MOOD_FACES

        face = MOOD_FACES.get(mood_machine.current, "(◕‿◕)")
        label = mood_machine.label

        dots = (
            f"{STATUS_DOTS[self.ollama_status]} Ollama  "
            f"{STATUS_DOTS[self.palace_status]} Palace  "
            f"{STATUS_DOTS[self.gemini_status]} Gemini  "
            f"{STATUS_DOTS[self.telegram_status]} Telegram"
        )

        model = config.ollama_model
        uptime = self.uptime_text
        mem = self.memory_count
        sess = self.session_count

        line1 = f"  BANTZ // OPERATIONS CENTER    {face} {label}"
        line2 = f"  {dots}  │  {model}"
        voice_part = f"  │  {self.voice_status}" if self.voice_status else ""
        line3 = f"  Uptime: {uptime}  │  Memory: {mem}  │  Sessions: {sess}{voice_part}"

        return f"{line1}\n{line2}\n{line3}"

    # ── Reactive watchers — re-render on any change ───────────────

    def watch_ollama_status(self) -> None:
        self.refresh()

    def watch_palace_status(self) -> None:
        self.refresh()

    def watch_gemini_status(self) -> None:
        self.refresh()

    def watch_telegram_status(self) -> None:
        self.refresh()

    def watch_uptime_text(self) -> None:
        self.refresh()

    def watch_memory_count(self) -> None:
        self.refresh()

    def watch_session_count(self) -> None:
        self.refresh()

    def watch_voice_status(self) -> None:
        self.refresh()

    # ── Event handler: ServiceHealthChanged ───────────────────────

    def on_service_health_changed(self, msg: ServiceHealthChanged) -> None:
        """React to health events from brain/integrations."""
        if msg.service == "ollama":
            self.ollama_status = msg.status
        elif msg.service == "palace":
            self.palace_status = msg.status
        elif msg.service == "gemini":
            self.gemini_status = msg.status
        elif msg.service == "telegram":
            self.telegram_status = msg.status

    def on_memory_count_updated(self, msg: MemoryCountUpdated) -> None:
        """React to DB count updates (event-driven, not polling)."""
        self.memory_count = msg.total_messages
        self.session_count = msg.total_sessions

    def on_voice_status_changed(self, msg: VoiceStatusChanged) -> None:
        """React to voice pipeline state changes — updates header status bar."""
        self.voice_status = msg.status

    # ── Uptime ticker ─────────────────────────────────────────────

    def _tick_uptime(self) -> None:
        elapsed = time.monotonic() - self._boot_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        if hours > 0:
            self.uptime_text = f"{hours}h {minutes:02d}m"
        else:
            self.uptime_text = f"{minutes}m"

    # ══════════════════════════════════════════════════════════════
    # One-time startup probes — all @work(thread=True) + timeout=3
    # ══════════════════════════════════════════════════════════════

    @work(thread=True)
    def _probe_ollama(self) -> None:
        """Startup probe: GET /api/tags with 3s timeout."""
        import httpx
        try:
            resp = httpx.get(
                f"{config.ollama_base_url}/api/tags", timeout=3.0,
            )
            if resp.status_code == 200:
                self.app.call_from_thread(
                    self._set_status, "ollama", ServiceStatus.UP,
                )
            else:
                self.app.call_from_thread(
                    self._set_status, "ollama", ServiceStatus.DEGRADED,
                )
        except Exception:
            self.app.call_from_thread(
                self._set_status, "ollama", ServiceStatus.DOWN,
            )

    @work(thread=True)
    def _probe_palace(self) -> None:
        """Startup probe: check MemPalace bridge enabled + initialized."""
        if not config.mempalace_enabled:
            self.app.call_from_thread(
                self._set_status, "palace", ServiceStatus.UNCONFIGURED,
            )
            return
        try:
            from bantz.memory.bridge import palace_bridge
            if palace_bridge and palace_bridge.enabled:
                self.app.call_from_thread(
                    self._set_status, "palace", ServiceStatus.UP,
                )
            else:
                self.app.call_from_thread(
                    self._set_status, "palace", ServiceStatus.DOWN,
                )
        except Exception:
            self.app.call_from_thread(
                self._set_status, "palace", ServiceStatus.DOWN,
            )

    @work(thread=True)
    def _probe_gemini(self) -> None:
        """Startup probe: lightweight model list if API key configured."""
        if not config.gemini_enabled or not config.gemini_api_key:
            self.app.call_from_thread(
                self._set_status, "gemini", ServiceStatus.UNCONFIGURED,
            )
            return
        import httpx
        try:
            resp = httpx.get(
                f"https://generativelanguage.googleapis.com/v1beta/models"
                f"?key={config.gemini_api_key}",
                timeout=3.0,
            )
            if resp.status_code == 200:
                self.app.call_from_thread(
                    self._set_status, "gemini", ServiceStatus.UP,
                )
            else:
                self.app.call_from_thread(
                    self._set_status, "gemini", ServiceStatus.DOWN,
                )
        except Exception:
            self.app.call_from_thread(
                self._set_status, "gemini", ServiceStatus.DOWN,
            )

    @work(thread=True)
    def _probe_telegram(self) -> None:
        """Startup probe: bot.get_me() if token configured."""
        if not config.telegram_bot_token:
            self.app.call_from_thread(
                self._set_status, "telegram", ServiceStatus.UNCONFIGURED,
            )
            return
        import httpx
        try:
            resp = httpx.get(
                f"https://api.telegram.org/bot{config.telegram_bot_token}/getMe",
                timeout=3.0,
            )
            if resp.status_code == 200:
                self.app.call_from_thread(
                    self._set_status, "telegram", ServiceStatus.UP,
                )
            else:
                self.app.call_from_thread(
                    self._set_status, "telegram", ServiceStatus.DOWN,
                )
        except Exception:
            self.app.call_from_thread(
                self._set_status, "telegram", ServiceStatus.DOWN,
            )

    @work(thread=True)
    def _fetch_db_counts(self) -> None:
        """Fetch memory/session counts once at startup."""
        try:
            from bantz.data import data_layer
            if data_layer.conversations:
                stats = data_layer.conversations.stats()
                self.app.call_from_thread(
                    self._set_counts,
                    stats.get("total_messages", 0),
                    stats.get("total_conversations", 0),
                )
        except Exception:
            pass

    # ── Thread-safe setters ───────────────────────────────────────

    def _set_status(self, service: str, status: ServiceStatus) -> None:
        if service == "ollama":
            self.ollama_status = status
        elif service == "palace":
            self.palace_status = status
        elif service == "gemini":
            self.gemini_status = status
        elif service == "telegram":
            self.telegram_status = status

    def _set_counts(self, messages: int, sessions: int) -> None:
        self.memory_count = messages
        self.session_count = sessions
