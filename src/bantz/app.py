"""
Bantz v2 — Textual UI
Main application. Left: chat. Right: system status + clock.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
)
from textual.reactive import reactive
from textual import work

from bantz.core.brain import brain, BrainResult
from bantz.config import config


# ── Widgets ───────────────────────────────────────────────────────────────────

class SystemStatus(Static):
    """Right panel: live CPU/RAM/Disk/Clock updates."""

    cpu: reactive[float] = reactive(0.0)
    ram: reactive[float] = reactive(0.0)
    disk: reactive[float] = reactive(0.0)
    clock: reactive[str] = reactive("")

    def on_mount(self) -> None:
        self.update_stats()
        self.set_interval(3, self.update_stats)
        self.set_interval(1, self.update_clock)

    def update_clock(self) -> None:
        self.clock = datetime.now().strftime("%H:%M:%S")
        self.refresh()

    @work(thread=True)
    def update_stats(self) -> None:
        import psutil, time
        self.cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        self.ram = mem.percent
        disk = psutil.disk_usage("/")
        self.disk = disk.percent

    def render(self) -> str:
        def bar(pct: float, width: int = 14) -> str:
            filled = int(pct / 100 * width)
            color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
            return f"[{color}]{'█' * filled}{'░' * (width - filled)}[/] {pct:.0f}%"

        now = self.clock or datetime.now().strftime("%H:%M:%S")
        return (
            f"[bold cyan]── SYSTEM ──────────[/]\n"
            f"[dim]CPU [/]  {bar(self.cpu)}\n"
            f"[dim]RAM [/]  {bar(self.ram)}\n"
            f"[dim]DISK[/]  {bar(self.disk)}\n\n"
            f"[bold cyan]── CLOCK ───────────[/]\n"
            f"[bold white]  {now}[/]\n"
            f"[dim]  {datetime.now().strftime('%A, %d %B %Y')}[/]"
        )


class ChatLog(RichLog):
    """Chat history — Extends RichLog."""

    def add_user(self, text: str) -> None:
        self.write(f"[bold green]▶ You[/]  {text}")

    def add_bantz(self, text: str) -> None:
        self.write(f"[bold cyan]◆ Bantz[/]  {text}")

    def add_system(self, text: str) -> None:
        self.write(f"[dim]  {text}[/]")

    def add_error(self, text: str) -> None:
        self.write(f"[bold red]✗[/]  {text}")


# ── Main App ───────────────────────────────────────────────────────────────────

class BantzApp(App):
    """Bantz v2 — Terminal Host"""

    CSS = """
    Screen {
        background: #0d0d0d;
    }
    #main-layout {
        height: 1fr;
    }
    #chat-panel {
        width: 3fr;
        border: solid #1e3a1e;
        padding: 0 1;
    }
    #right-panel {
        width: 1fr;
        border: solid #1a2a3a;
        padding: 1;
    }
    ChatLog {
        height: 1fr;
        scrollbar-color: #2a4a2a;
    }
    #input-bar {
        height: 3;
        border-top: solid #1e3a1e;
        padding: 0 1;
    }
    Input {
        background: #111;
        border: solid #2a4a2a;
        color: #00ff88;
    }
    Input:focus {
        border: solid #00ff88;
    }
    SystemStatus {
        height: auto;
        color: #aaccaa;
    }
    Header {
        background: #0a1a0a;
        color: #00ff88;
    }
    Footer {
        background: #0a1a0a;
        color: #446644;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Exit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("escape", "focus_input", "Focus Input"),
    ]

    TITLE = "BANTZ v2"
    SUB_TITLE = "your terminal host"

    def __init__(self) -> None:
        super().__init__()
        self._pending_confirm: BrainResult | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-layout"):
            with Vertical(id="chat-panel"):
                yield ChatLog(id="chat-log", highlight=True, markup=True)
                with Horizontal(id="input-bar"):
                    yield Input(placeholder="Say something to Bantz... (Ctrl+Q to exit)", id="chat-input")
        with Vertical(id="right-panel"):
            yield SystemStatus()
        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_system("Bantz v2 started.")
        chat.add_system(f"Model: {config.ollama_model}")
        chat.add_system("─" * 40)
        self.check_ollama()
        self.query_one("#chat-input", Input).focus()

    @work(exclusive=False)
    async def check_ollama(self) -> None:
        from bantz.llm.ollama import ollama
        chat = self.query_one("#chat-log", ChatLog)
        ok = await ollama.is_available()
        if ok:
            chat.add_system(f"✓ Ollama connected → {config.ollama_model}")
        else:
            chat.add_error(f"Ollama connection failed: {config.ollama_base_url}")
            chat.add_system("  Is Ollama running? `ollama serve`")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = ""

        chat = self.query_one("#chat-log", ChatLog)
        chat.add_user(text)

        # Confirmation flow
        if self._pending_confirm is not None:
            await self._handle_confirm(text, chat)
            return

        self.process_input(text, chat)

    @work(exclusive=False)
    async def process_input(self, text: str, chat: ChatLog) -> None:
        chat.add_system("thinking...")
        result = await brain.process(text)

        # Clear the previous "thinking..." line — simple solution
        if result.needs_confirm:
            self._pending_confirm = result
            chat.add_bantz(result.response)
        else:
            if result.tool_used:
                chat.add_system(f"[tool: {result.tool_used}]")
            chat.add_bantz(result.response)

    async def _handle_confirm(self, text: str, chat: ChatLog) -> None:
        pending = self._pending_confirm
        self._pending_confirm = None

        evet = text.lower().strip() in ("evet", "e", "yes", "y", "ok", "tamam")
        if evet:
            chat.add_system("Confirmed, executing...")
            result = await brain.process(pending.pending_command, confirmed=True)
            chat.add_bantz(result.response)
        else:
            chat.add_system("Cancelled.")

    def action_clear_chat(self) -> None:
        self.query_one("#chat-log", ChatLog).clear()

    def action_focus_input(self) -> None:
        self.query_one("#chat-input", Input).focus()


def run() -> None:
    BantzApp().run()