"""
Bantz v2 — Textual UI
Left: chat panel. Right: system status + clock.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static
from textual.reactive import reactive
from textual import work

from bantz.core.brain import brain, BrainResult
from bantz.config import config


# ── SystemStatus widget ───────────────────────────────────────────────────────

class SystemStatus(Static):
    cpu: reactive[float] = reactive(0.0)
    ram: reactive[float] = reactive(0.0)
    disk: reactive[float] = reactive(0.0)
    clock: reactive[str] = reactive("")

    def on_mount(self) -> None:
        self._refresh_stats()
        self.set_interval(3, self._refresh_stats)
        self.set_interval(1, self._tick_clock)

    def _tick_clock(self) -> None:
        self.clock = datetime.now().strftime("%H:%M:%S")
        self.refresh()

    @work(thread=True)
    def _refresh_stats(self) -> None:
        import psutil
        self.cpu = psutil.cpu_percent(interval=0.5)
        self.ram = psutil.virtual_memory().percent
        self.disk = psutil.disk_usage("/").percent

    def render(self) -> str:
        def bar(pct: float, width: int = 12) -> str:
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


# ── Thinking indicator — separated Label, removable ─────────────────────────────

class ThinkingLabel(Static):
    """ While Bantz is thinking, show an animated indicator."""

    _frame: reactive[int] = reactive(0)
    FRAMES = ["⠋", "⠙", "⠸", "⠴", "⠦", "⠇"]

    def on_mount(self) -> None:
        self.set_interval(0.12, self._spin)

    def _spin(self) -> None:
        self._frame = (self._frame + 1) % len(self.FRAMES)
        self.refresh()

    def render(self) -> str:
        f = self.FRAMES[self._frame % len(self.FRAMES)]
        return f"[dim cyan]  {f} thinking...[/]"


# ── ChatLog ───────────────────────────────────────────────────────────────────

class ChatLog(RichLog):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._streaming_buffer: str = ""
        self._stream_started: bool = False

    def add_user(self, text: str) -> None:
        self.write(f"[bold green]▶ You[/]   {text}")

    def add_bantz(self, text: str) -> None:
        # Multi-line response → indent each line
        lines = text.strip().splitlines()
        if not lines:
            return
        self.write(f"[bold cyan]◆ Bantz[/]  {lines[0]}")
        for line in lines[1:]:
            self.write(f"         {line}")

    def stream_start(self) -> None:
        """Begin a streaming response — write the prefix, prepare buffer."""
        self._streaming_buffer = ""
        self._stream_started = True
        self.write("[bold cyan]◆ Bantz[/]  ...")

    def stream_token(self, token: str) -> None:
        """Append a token to the current streaming response."""
        self._streaming_buffer += token
        # Split into completed lines vs current partial line
        parts = self._streaming_buffer.split("\n")
        # Replace the last written line with current partial text
        if self.lines:
            # Remove the last placeholder/partial line
            self.lines.pop()
        # Write any completed lines
        if len(parts) > 1:
            for completed_line in parts[:-1]:
                self.write(f"         {completed_line}")
        # Write current partial line
        partial = parts[-1] if parts else ""
        self.write(f"         {partial}")
        self.scroll_end(animate=False)

    def stream_end(self) -> str:
        """Finish streaming — return the full accumulated text."""
        text = self._streaming_buffer
        self._streaming_buffer = ""
        self._stream_started = False
        return text

    def add_system(self, text: str) -> None:
        self.write(f"[dim]  {text}[/]")

    def add_error(self, text: str) -> None:
        self.write(f"[bold red]  ✗ {text}[/]")

    def add_tool(self, tool_name: str) -> None:
        self.write(f"[dim magenta]  ⚙ [{tool_name}][/]")


# ── Main App ───────────────────────────────────────────────────────────────────

class BantzApp(App):
    TITLE = "BANTZ v2"
    SUB_TITLE = "your terminal host"

    CSS = """
    Screen { background: #0d0d0d; }

    #main-layout { height: 1fr; }

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

    #thinking-area {
        height: 1;
        padding: 0 1;
    }

    #input-row {
        height: 3;
        border-top: solid #1e3a1e;
        padding: 0 1;
    }

    Input {
        background: #111;
        border: solid #2a4a2a;
        color: #00ff88;
    }
    Input:focus { border: solid #00ff88; }

    SystemStatus { height: auto; color: #aaccaa; }

    Header { background: #0a1a0a; color: #00ff88; }
    Footer { background: #0a1a0a; color: #446644; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+c", "copy_selection", "Copy"),
        Binding("escape", "focus_input", "Focus"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._pending: BrainResult | None = None
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-layout"):
            with Vertical(id="chat-panel"):
                yield ChatLog(id="chat-log", highlight=True, markup=True)
                yield Static("", id="thinking-area")   # thinking indicator buraya mount/unmount
                with Horizontal(id="input-row"):
                    yield Input(
                        placeholder="Tell Bantz something... (Ctrl+Q to quit, Ctrl+L to clear)",
                        id="chat-input",
                    )
            with Vertical(id="right-panel"):
                yield SystemStatus()
        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_system("Bantz v2 started.")
        chat.add_system(f"Model: {config.ollama_model}")
        chat.add_system("─" * 38)

        # Show immediate time-based greeting before Google API calls
        from bantz.core.time_context import time_ctx
        chat.add_bantz(time_ctx.greeting_line())

        self._check_ollama()
        self._warm_up_ollama()
        self._enrich_butler_greeting()
        self._start_gps_server()
        self._start_stationary_checker()
        self.query_one("#chat-input", Input).focus()

    async def action_quit(self) -> None:
        """Send GPS stop signal before exiting (#74)."""
        try:
            from bantz.core.gps_server import gps_server
            await gps_server.stop()
        except Exception:
            pass
        self.exit()

    @work(exclusive=False)
    async def _start_gps_server(self) -> None:
        """Start the GPS receiver in the background."""
        try:
            from bantz.core.gps_server import gps_server
            ok = await gps_server.start()
            if ok:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_system(f"GPS: {gps_server.url}")
                chat.add_system(f"Relay: {gps_server.relay_topic}")
        except Exception:
            pass

    def _start_stationary_checker(self) -> None:
        """Run stationary check every 5 minutes."""
        self.set_interval(300, self._check_stationary)

    @work(exclusive=False)
    async def _check_stationary(self) -> None:
        """Periodic check: if user is stationary at unknown place, show prompt."""
        try:
            from bantz.core.places import places
            notice = places.check_stationary()
            if notice:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_bantz(notice)
                chat.scroll_end()
        except Exception:
            pass

    @work(exclusive=False)
    async def _warm_up_ollama(self) -> None:
        """Fire a hidden dummy request to warm up Ollama while greeting assembles."""
        try:
            from bantz.llm.ollama import ollama
            await ollama.chat([{"role": "user", "content": "hi"}])
        except Exception:
            pass

    @work(exclusive=False)
    async def _enrich_butler_greeting(self) -> None:
        """Try to get a richer butler greeting with live data (mail, calendar, etc.)."""
        from bantz.core.memory import memory
        try:
            from bantz.core.session import session_tracker
            from bantz.core.butler import butler

            session_info = session_tracker.on_launch()

            def _run_greet():
                return asyncio.run(butler.greet(session_info))

            loop = asyncio.get_event_loop()
            text = await asyncio.wait_for(
                loop.run_in_executor(None, _run_greet), timeout=8
            )
            if text:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_bantz(text)
                chat.scroll_end()
                try:
                    memory.add("assistant", text, tool_used="startup")
                except Exception:
                    pass
        except (asyncio.CancelledError, Exception):
            pass


    @work(exclusive=False)
    async def _check_ollama(self) -> None:
        from bantz.llm.ollama import ollama
        chat = self.query_one("#chat-log", ChatLog)
        ok = await ollama.is_available()
        if ok:
            chat.add_system(f"✓ Ollama connected → {config.ollama_model}")
        else:
            chat.add_error(f"Ollama unreachable: {config.ollama_base_url}")
            chat.add_system("  → Is `ollama serve` running?")

    # ── Input handler ──────────────────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._busy:
            return

        self.query_one("#chat-input", Input).value = ""
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_user(text)

        # Confirmation flow
        if self._pending is not None:
            await self._handle_confirm(text, chat)
            return

        self._start_processing(text, chat)

    @work(exclusive=False)
    async def _start_processing(self, text: str, chat: ChatLog) -> None:
        self._busy = True
        self._show_thinking(True)

        try:
            result = await brain.process(text)
        except Exception as exc:
            self._show_thinking(False)
            self._busy = False
            err_name = type(exc).__name__
            chat.add_error(f"Network/tool error: {err_name} — {exc}")
            chat.add_system("  Check your internet connection and try again.")
            return

        # ── Streaming response (#67) ──
        if result.stream is not None:
            self._show_thinking(False)
            if result.tool_used:
                chat.add_tool(result.tool_used)
            chat.stream_start()

            accumulated = ""
            try:
                async for token in result.stream:
                    accumulated += token
                    chat.stream_token(token)
            except Exception as exc:
                chat.stream_end()
                self._busy = False
                chat.add_error(f"Stream error: {exc}")
                return

            full_text = chat.stream_end()
            self._busy = False

            # Post-processing on accumulated text
            from bantz.core.brain import strip_markdown
            cleaned = strip_markdown(full_text)

            # Save to memory + graph
            from bantz.core.memory import memory as _mem
            _mem.add("assistant", cleaned, tool_used=result.tool_used)
            try:
                from bantz.core.brain import brain as _brain
                await _brain._graph_store(text, cleaned, result.tool_used)
            except Exception:
                pass

            return

        # ── Non-streaming response ──
        self._show_thinking(False)
        self._busy = False

        if result.needs_confirm:
            self._pending = result
            chat.add_bantz(result.response)
        else:
            if result.tool_used:
                chat.add_tool(result.tool_used)
            chat.add_bantz(result.response)

    async def _handle_confirm(self, text: str, chat: ChatLog) -> None:
        pending = self._pending
        self._pending = None
        evet = text.lower().strip() in ("evet", "e", "yes", "y", "ok", "tamam")
        if evet:
            self._busy = True
            self._show_thinking(True)
            try:
                # Direct tool execution for compose/reply drafts
                if pending.pending_tool and pending.pending_args:
                    from bantz.tools import registry as _reg
                    tool = _reg.get(pending.pending_tool)
                    if tool:
                        tr = await tool.execute(**pending.pending_args)
                        self._show_thinking(False)
                        self._busy = False
                        chat.add_tool(pending.pending_tool)
                        chat.add_bantz(tr.output if tr.success else f"Error: {tr.error}")
                        return
                # Fallback: shell command confirmation
                result = await brain.process(
                    pending.pending_command,
                    confirmed=True,
                )
                self._show_thinking(False)
                self._busy = False
                if result.tool_used:
                    chat.add_tool(result.tool_used)
                chat.add_bantz(result.response)
            except Exception as exc:
                self._show_thinking(False)
                self._busy = False
                err_name = type(exc).__name__
                chat.add_error(f"Network/tool error: {err_name} — {exc}")
                chat.add_system("  Check your internet connection and try again.")
        else:
            chat.add_system("Cancelled.")

    # ── Thinking indicator ─────────────────────────────────────────────────

    def _show_thinking(self, show: bool) -> None:
        area = self.query_one("#thinking-area", Static)
        if show:
            # Inline spinner — Static doesn't support animation but it's enough
            area.update("[dim cyan]  ⟳ thinking...[/]")
        else:
            area.update("")

    # ── Actions ────────────────────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        self.query_one("#chat-log", ChatLog).clear()

    def action_copy_selection(self) -> None:
        import pyperclip
        chat = self.query_one("#chat-log", ChatLog)
        # RichLog has no built-in selection; copy last Bantz message
        try:
            lines = chat.lines
            # find last Bantz line
            for line in reversed(lines):
                text = line.text if hasattr(line, 'text') else str(line)
                if 'Bantz' in text or '◆' in text:
                    clean = text.replace('◆ Bantz', '').strip()
                    pyperclip.copy(clean)
                    chat.add_system('Copied ✓')
                    return
        except Exception:
            pass

    def action_focus_input(self) -> None:
        self.query_one("#chat-input", Input).focus()


def run() -> None:
    BantzApp().run()