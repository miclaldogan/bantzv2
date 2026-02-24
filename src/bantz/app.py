"""
Bantz v3 — Operations Center TUI

Multi-panel Textual UI:
  Left  (1/4): SystemPanel — CPU/RAM/VRAM/DISK + graph stats + today snapshot
  Right (3/4): ChatPanel (top 2/3) + TaskQueue + LogPanel (bottom 1/3 split)
"""
from __future__ import annotations

from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static
from textual.reactive import reactive
from textual import work

from bantz.core.brain import brain, BrainResult
from bantz.config import config


# ── System Panel ──────────────────────────────────────────────────────────────

class SystemPanel(Static):
    """Left-side panel: hardware stats, graph memory, today's snapshot."""

    cpu:         reactive[float] = reactive(0.0)
    ram_used:    reactive[float] = reactive(0.0)
    ram_total:   reactive[float] = reactive(0.0)
    vram_used:   reactive[float] = reactive(0.0)
    vram_total:  reactive[float] = reactive(0.0)
    disk_free:   reactive[float] = reactive(0.0)
    graph_nodes: reactive[int]   = reactive(0)
    graph_rels:  reactive[int]   = reactive(0)
    graph_ok:    reactive[bool]  = reactive(False)
    today_lines: reactive[list]  = reactive([])
    clock:       reactive[str]   = reactive("")

    def on_mount(self) -> None:
        self._refresh_hw()
        self._refresh_graph()
        self._refresh_today()
        self.set_interval(5,  self._refresh_hw)
        self.set_interval(1,  self._tick_clock)
        self.set_interval(30, self._refresh_graph)
        self.set_interval(60, self._refresh_today)

    def _tick_clock(self) -> None:
        self.clock = datetime.now().strftime("%H:%M:%S")
        self.refresh()

    @work(thread=True)
    def _refresh_hw(self) -> None:
        import psutil
        self.cpu = psutil.cpu_percent(interval=0.5)
        vm = psutil.virtual_memory()
        self.ram_used  = vm.used  / 1024 ** 3
        self.ram_total = vm.total / 1024 ** 3
        self.disk_free = psutil.disk_usage("/").free / 1024 ** 3
        # VRAM via pynvml (optional — graceful fallback)
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            self.vram_used  = info.used  / 1024 ** 3
            self.vram_total = info.total / 1024 ** 3
        except Exception:
            self.vram_used = self.vram_total = 0.0

    @work(exclusive=False)
    async def _refresh_graph(self) -> None:
        try:
            from bantz.memory.graph import graph_memory
            stats = graph_memory.stats()
            self.graph_ok    = stats["available"]
            self.graph_nodes = stats["nodes"]
            self.graph_rels  = stats["relations"]
        except Exception:
            self.graph_ok = False

    @work(exclusive=False)
    async def _refresh_today(self) -> None:
        lines: list[str] = []
        try:
            from bantz.core.schedule import Schedule
            sched = Schedule.load()
            nc = sched.next_class()
            if nc:
                lines.append(f"{nc['time']} — {nc['course']}")
        except Exception:
            pass
        self.today_lines = lines

    def render(self) -> str:
        def bar(pct: float, width: int = 10) -> str:
            filled = int(pct / 100 * width)
            color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
            return f"[{color}]{'█' * filled}{'░' * (width - filled)}[/] {pct:4.0f}%"

        cpu_bar = bar(self.cpu)

        ram_pct = (self.ram_used / self.ram_total * 100) if self.ram_total else 0
        ram_color = "green" if ram_pct < 60 else "yellow" if ram_pct < 85 else "red"
        ram_line = f"[dim]RAM [/]  [{ram_color}]{self.ram_used:.1f}[/]/{self.ram_total:.0f} GB\n"

        vram_line = ""
        if self.vram_total > 0:
            vram_pct = (self.vram_used / self.vram_total * 100) if self.vram_total else 0
            vram_color = "green" if vram_pct < 60 else "yellow" if vram_pct < 85 else "red"
            vram_line = f"[dim]VRAM[/]  [{vram_color}]{self.vram_used:.1f}[/]/{self.vram_total:.0f} GB\n"

        disk_line = f"[dim]DISK[/]  {self.disk_free:.0f} GB free\n"

        # Graph memory
        if self.graph_ok:
            graph_section = (
                f"[dim]Nodes    [/] {self.graph_nodes:,}\n"
                f"[dim]Relations[/] {self.graph_rels:,}\n"
            )
        else:
            graph_section = "[dim]Neo4j offline[/]\n"

        # Today's snapshot
        today_str = ""
        for line in self.today_lines[:3]:
            today_str += f"  {line}\n"
        if not today_str:
            today_str = "  [dim]No classes[/]\n"

        now      = self.clock or datetime.now().strftime("%H:%M:%S")
        date_str = datetime.now().strftime("%a %d %b %Y")

        return (
            f"[bold cyan]// SYSTEM[/]\n"
            f"[dim]CPU [/]  {cpu_bar}\n"
            f"{ram_line}"
            f"{vram_line}"
            f"{disk_line}\n"
            f"[bold cyan]// MEMORY GRAPH[/]\n"
            f"{graph_section}\n"
            f"[bold cyan]// TODAY[/]\n"
            f"{today_str}\n"
            f"[bold cyan]// CLOCK[/]\n"
            f"[bold white]  {now}[/]\n"
            f"[dim]  {date_str}[/]"
        )


# ── Task Queue Panel ──────────────────────────────────────────────────────────

class TaskQueuePanel(RichLog):
    """Displays scheduled / recently completed jobs."""

    def on_mount(self) -> None:
        self.write("[bold cyan]// TASK QUEUE[/]")
        self.set_interval(15, self._refresh_jobs)

    @work(exclusive=False)
    async def _refresh_jobs(self) -> None:
        try:
            from bantz.core.scheduler import scheduler
            jobs = scheduler.list_jobs()
            self.clear()
            self.write("[bold cyan]// TASK QUEUE[/]")
            for j in jobs[:8]:
                self.write(f"[dim cyan]⟳[/] {j['name']} · {j['next_run']}")
        except Exception:
            pass


# ── System Log Panel ──────────────────────────────────────────────────────────

class LogPanel(RichLog):
    """Tailing log — tool calls, errors, events."""

    def on_mount(self) -> None:
        self.write("[bold cyan]// SYSTEM LOG[/]")

    def log_event(self, tag: str, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.write(f"[dim]{ts}[/] [{tag}] {message}")


# ── Chat Log ──────────────────────────────────────────────────────────────────

class ChatLog(RichLog):
    def add_user(self, text: str) -> None:
        self.write(f"[bold green]▶ You[/]    {text}")

    def add_bantz(self, text: str) -> None:
        lines = text.strip().splitlines()
        if not lines:
            return
        self.write(f"[bold cyan]◆ Bantz[/]  {lines[0]}")
        for line in lines[1:]:
            self.write(f"          {line}")

    def add_system(self, text: str) -> None:
        self.write(f"[dim]  {text}[/]")

    def add_error(self, text: str) -> None:
        self.write(f"[bold red]  ✗ {text}[/]")

    def add_tool(self, tool_name: str) -> None:
        self.write(f"[dim magenta]  ⚙ [{tool_name}][/]")


# ── Main App ──────────────────────────────────────────────────────────────────

class BantzApp(App):
    TITLE     = "BANTZ // OPERATIONS CENTER"
    SUB_TITLE = "v3"

    CSS = """
    Screen { background: #0d0d0d; }

    #main-layout { height: 1fr; }

    #left-panel {
        width: 1fr;
        border: solid #1a2a1a;
        padding: 1;
    }
    #right-panel { width: 3fr; }

    #chat-panel {
        height: 2fr;
        border: solid #1e3a1e;
        padding: 0 1;
    }
    #bottom-strip { height: 1fr; }

    #task-panel {
        width: 1fr;
        border: solid #1a2a3a;
        padding: 0 1;
    }
    #log-panel {
        width: 1fr;
        border: solid #2a1a2a;
        padding: 0 1;
    }

    ChatLog        { height: 1fr; scrollbar-color: #2a4a2a; }
    TaskQueuePanel { height: 1fr; scrollbar-color: #1a2a3a; }
    LogPanel       { height: 1fr; scrollbar-color: #2a1a2a; }

    #thinking-area { height: 1; padding: 0 1; }

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

    SystemPanel { height: auto; color: #aaccaa; }
    Header { background: #0a1a0a; color: #00ff88; }
    Footer { background: #0a1a0a; color: #446644; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit",        "Quit"),
        Binding("ctrl+l", "clear_chat",  "Clear"),
        Binding("ctrl+c", "copy_last",   "Copy"),
        Binding("escape", "focus_input", "Focus"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._pending: BrainResult | None = None
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-layout"):
            with Vertical(id="left-panel"):
                yield SystemPanel()
            with Vertical(id="right-panel"):
                with Vertical(id="chat-panel"):
                    yield ChatLog(id="chat-log", highlight=True, markup=True)
                    yield Static("", id="thinking-area")
                    with Horizontal(id="input-row"):
                        yield Input(
                            placeholder="Tell Bantz something... (Ctrl+Q to quit, Ctrl+L to clear)",
                            id="chat-input",
                        )
                with Horizontal(id="bottom-strip"):
                    yield TaskQueuePanel(id="task-panel", highlight=True, markup=True)
                    yield LogPanel(id="log-panel",  highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_system("Bantz v3 starting...")
        chat.add_system(f"Model: {config.ollama_model}")
        chat.add_system("─" * 40)
        self._check_ollama()
        self._warm_up_ollama()
        self._show_butler_greeting()
        self.query_one("#chat-input", Input).focus()

    @work(exclusive=False)
    async def _warm_up_ollama(self) -> None:
        """Fire a hidden request to warm up Ollama while greeting assembles."""
        try:
            from bantz.llm.ollama import ollama
            await ollama.chat([{"role": "user", "content": "hi"}])
        except Exception:
            pass

    @work(exclusive=False)
    async def _show_butler_greeting(self) -> None:
        """Butler-style context-aware greeting on app launch."""
        from bantz.core.session import session_tracker
        from bantz.core.butler import butler
        from bantz.core.memory import memory

        try:
            session_info = session_tracker.on_launch()
            text = await butler.greet(session_info)
        except Exception:
            from bantz.core.time_context import time_ctx
            text = time_ctx.greeting_line()

        chat = self.query_one("#chat-log", ChatLog)
        chat.add_bantz(text)

        try:
            memory.add("assistant", text, tool_used="startup")
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
            chat.add_error(f"Ollama not reachable: {config.ollama_base_url}")
            chat.add_system("  → is `ollama serve` running?")

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

        result = await brain.process(text)

        self._show_thinking(False)
        self._busy = False

        if result.needs_confirm:
            self._pending = result
            chat.add_bantz(result.response)
        else:
            if result.tool_used:
                chat.add_tool(result.tool_used)
                self._log(
                    result.tool_used,
                    result.response[:60] + "..." if len(result.response) > 60 else result.response,
                )
            chat.add_bantz(result.response)

    async def _handle_confirm(self, text: str, chat: ChatLog) -> None:
        pending       = self._pending
        self._pending = None
        confirmed     = text.lower().strip() in ("yes", "y", "ok")

        if confirmed:
            self._busy = True
            self._show_thinking(True)

            if pending.pending_tool and pending.pending_args:
                from bantz.tools import registry as _reg
                tool = _reg.get(pending.pending_tool)
                if tool:
                    tr = await tool.execute(**pending.pending_args)
                    self._show_thinking(False)
                    self._busy = False
                    chat.add_tool(pending.pending_tool)
                    msg = tr.output if tr.success else f"Error: {tr.error}"
                    chat.add_bantz(msg)
                    self._log(pending.pending_tool, msg[:60])
                    return

            result = await brain.process(pending.pending_command, confirmed=True)
            self._show_thinking(False)
            self._busy = False

            if result.tool_used:
                chat.add_tool(result.tool_used)
            chat.add_bantz(result.response)
        else:
            chat.add_system("Cancelled.")

    # ── Thinking indicator ─────────────────────────────────────────────────

    def _show_thinking(self, show: bool) -> None:
        area = self.query_one("#thinking-area", Static)
        area.update("[dim cyan]  ⟳ thinking...[/]" if show else "")

    # ── System log helper ──────────────────────────────────────────────────

    def _log(self, tag: str, message: str) -> None:
        try:
            self.query_one("#log-panel", LogPanel).log_event(tag, message)
        except Exception:
            pass

    # ── Actions ────────────────────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        self.query_one("#chat-log", ChatLog).clear()

    def action_copy_last(self) -> None:
        try:
            import pyperclip
            chat  = self.query_one("#chat-log", ChatLog)
            lines = chat.lines
            for line in reversed(lines):
                text = line.text if hasattr(line, "text") else str(line)
                if "Bantz" in text or "◆" in text:
                    clean = text.replace("◆ Bantz", "").strip()
                    pyperclip.copy(clean)
                    chat.add_system("Copied ✓")
                    return
        except Exception:
            pass

    def action_focus_input(self) -> None:
        self.query_one("#chat-input", Input).focus()


def run() -> None:
    BantzApp().run()
