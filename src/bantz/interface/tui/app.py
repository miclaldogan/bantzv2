"""
Bantz v3 — Textual TUI Application

Entry point for the terminal user interface.
Left panel: chat.  Right panel: system status + clock.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Static
from textual import work

from bantz.core.brain import brain, BrainResult
from bantz.config import config
from bantz.interface.tui.panels.system import SystemStatus
from bantz.interface.tui.panels.chat import ChatLog, ThinkingLabel

_STYLES_PATH = Path(__file__).parent / "styles.tcss"


class BantzApp(App):
    TITLE = "BANTZ v3"
    SUB_TITLE = "your terminal host"

    CSS_PATH = _STYLES_PATH

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
                yield Static("", id="thinking-area")
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
        chat.add_system("Bantz v3 started.")
        chat.add_system(f"Model: {config.ollama_model}")
        chat.add_system("─" * 38)

        from bantz.core.time_context import time_ctx
        chat.add_bantz(time_ctx.greeting_line())

        self._check_ollama()
        self._warm_up_ollama()
        self._enrich_butler_greeting()
        self._start_gps_server()
        self._start_stationary_checker()
        self._start_morning_briefing_timer()
        self._start_reminder_checker()
        self._start_digest_checker()
        self.query_one("#chat-input", Input).focus()

    async def action_quit(self) -> None:
        try:
            from bantz.core.gps_server import gps_server
            await gps_server.stop()
        except Exception:
            pass
        self.exit()

    @work(exclusive=False)
    async def _start_gps_server(self) -> None:
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
        self.set_interval(300, self._check_stationary)

    @work(exclusive=False)
    async def _check_stationary(self) -> None:
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
        try:
            from bantz.llm.ollama import ollama
            await ollama.chat([{"role": "user", "content": "hi"}])
        except Exception:
            pass

    @work(exclusive=False)
    async def _enrich_butler_greeting(self) -> None:
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

    def _start_morning_briefing_timer(self) -> None:
        self.set_interval(60, self._check_morning_briefing)

    def _start_reminder_checker(self) -> None:
        self.set_interval(config.reminder_check_interval, self._check_reminders)

    @work(exclusive=False)
    async def _check_reminders(self) -> None:
        try:
            from bantz.core.scheduler import scheduler
            from bantz.core.memory import memory

            due = scheduler.check_due()
            if not due:
                return

            chat = self.query_one("#chat-log", ChatLog)
            for r in due:
                repeat_tag = f" (repeats {r['repeat']})" if r['repeat'] != 'none' else ''
                text = f"\u23f0 Reminder: {r['title']}{repeat_tag}"
                chat.add_bantz(text)
                try:
                    memory.add("assistant", text, tool_used="reminder")
                except Exception:
                    pass
            chat.scroll_end()
        except Exception:
            pass

    @work(exclusive=False)
    async def _check_morning_briefing(self) -> None:
        try:
            from bantz.personality.greeting import greeting_manager
            from bantz.core.memory import memory

            text = await greeting_manager.morning_briefing_if_due()
            if text:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_bantz(text)
                chat.scroll_end()
                try:
                    memory.add("assistant", text, tool_used="briefing")
                except Exception:
                    pass
        except Exception:
            pass

    def _start_digest_checker(self) -> None:
        self.set_interval(60, self._check_digest)

    @work(exclusive=False)
    async def _check_digest(self) -> None:
        try:
            from bantz.core.digest import digest_manager
            from bantz.core.memory import memory

            # Check daily digest
            text = await digest_manager.daily_if_due()
            if text:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_bantz(text)
                chat.scroll_end()
                try:
                    memory.add("assistant", text, tool_used="digest")
                except Exception:
                    pass

            # Check weekly digest
            text = await digest_manager.weekly_if_due()
            if text:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_bantz(text)
                chat.scroll_end()
                try:
                    memory.add("assistant", text, tool_used="digest")
                except Exception:
                    pass
        except Exception:
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

        # ── Streaming response ──
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

            from bantz.core.finalizer import strip_markdown
            cleaned = strip_markdown(full_text)

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
        confirmed = text.lower().strip() in ("yes", "y", "ok", "evet", "e", "tamam")
        if confirmed:
            self._busy = True
            self._show_thinking(True)
            try:
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
            area.update("[dim cyan]  ⟳ thinking...[/]")
        else:
            area.update("")

    # ── Actions ────────────────────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        self.query_one("#chat-log", ChatLog).clear()

    def action_copy_selection(self) -> None:
        import pyperclip
        chat = self.query_one("#chat-log", ChatLog)
        try:
            lines = chat.lines
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
    """Launch the Bantz TUI."""
    BantzApp().run()
