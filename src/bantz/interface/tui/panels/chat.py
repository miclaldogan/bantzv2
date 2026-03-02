"""
Bantz — Chat Panel Widgets

ChatLog: scrollable rich-text chat log with streaming support.
ThinkingLabel: animated spinner shown while Bantz is processing.
"""
from __future__ import annotations

from textual.widgets import RichLog, Static
from textual.reactive import reactive


class ThinkingLabel(Static):
    """Animated spinner shown while Bantz is thinking."""

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


class ChatLog(RichLog):
    """Scrollable rich-text chat log with streaming support."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._streaming_buffer: str = ""
        self._stream_started: bool = False

    def add_user(self, text: str) -> None:
        self.write(f"[bold green]▶ You[/]   {text}")

    def add_bantz(self, text: str) -> None:
        lines = text.strip().splitlines()
        if not lines:
            return
        self.write(f"[bold cyan]◆ Bantz[/]  {lines[0]}")
        for line in lines[1:]:
            self.write(f"         {line}")

    def stream_start(self) -> None:
        """Begin a streaming response."""
        self._streaming_buffer = ""
        self._stream_started = True
        self.write("[bold cyan]◆ Bantz[/]  ...")

    def stream_token(self, token: str) -> None:
        """Append a token to the current streaming response."""
        self._streaming_buffer += token
        parts = self._streaming_buffer.split("\n")
        if self.lines:
            self.lines.pop()
        if len(parts) > 1:
            for completed_line in parts[:-1]:
                self.write(f"         {completed_line}")
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
