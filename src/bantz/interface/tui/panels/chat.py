"""
Bantz — Chat Panel Widgets

ChatLog: scrollable rich-text chat log with streaming support.
ThinkingLabel: animated spinner shown while Bantz is processing.
ThinkingPanel: collapsible panel showing real-time <thinking> stream (#273).
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


class ThinkingPanel(Static):
    """Collapsible panel showing real-time ``<thinking>`` stream (#273).

    Displays the LLM's chain-of-thought reasoning in dim gray text
    as tokens arrive via EventBus ``thinking_token`` events.  Auto-hides
    when thinking completes (``thinking_done`` event).

    The panel is invisible by default and only appears when there is
    active thinking content to show.
    """

    DEFAULT_CSS = """
    ThinkingPanel {
        height: auto;
        max-height: 6;
        overflow-y: auto;
        padding: 0 1;
        display: none;
    }
    ThinkingPanel.visible {
        display: block;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._buffer: str = ""

    def start(self) -> None:
        """Begin a new thinking session — clear and show."""
        self._buffer = ""
        self.update("[dim]  🧠 Thinking...[/]")
        self.add_class("visible")

    def append_token(self, token: str) -> None:
        """Append a thinking token and refresh display."""
        self._buffer += token
        # Truncate display to last ~300 chars for readability
        display = self._buffer[-300:]
        if len(self._buffer) > 300:
            display = "…" + display
        self.update(f"[dim]  🧠 {display}[/]")

    def finish(self) -> None:
        """Thinking complete — hide after a brief moment."""
        if self._buffer:
            display = self._buffer[-200:]
            if len(self._buffer) > 200:
                display = "…" + display
            self.update(f"[dim]  ✓ {display}[/]")
        # Auto-hide after 2 seconds
        self.set_timer(2.0, self._hide)

    def _hide(self) -> None:
        """Remove visibility."""
        self.remove_class("visible")
        self._buffer = ""
        self.update("")


class ChatLog(RichLog):
    """Scrollable rich-text chat log with streaming support."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._streaming_buffer: str = ""   # only the current incomplete line (delta)
        self._full_text: str = ""          # full accumulated text for stream_end()
        self._stream_started: bool = False

    def add_user(self, text: str) -> None:
        self.write(f"[bold green]▶ You[/]   {text}")
        self.scroll_end(animate=False)

    def add_bantz(self, text: str) -> None:
        lines = text.strip().splitlines()
        if not lines:
            return
        self.write(f"[bold cyan]◆ Bantz[/]  {lines[0]}")
        for line in lines[1:]:
            self.write(f"         {line}")
        self.scroll_end(animate=False)

    def stream_start(self) -> None:
        """Begin a streaming response."""
        self._streaming_buffer = ""
        self._full_text = ""
        self._stream_started = True
        self.write("[bold cyan]◆ Bantz[/]  ...")

    def stream_token(self, token: str) -> None:
        """Append a token — delta buffering (Option C).

        Only the current incomplete line lives in ``_streaming_buffer``.
        When a ``\\n`` arrives the completed line is written permanently
        and the buffer is reset to the trailing fragment.  The last
        (temporary) line is popped and re-written each call so the
        terminal only touches one line per token — zero flicker.
        """
        # 1. Pop the previous temporary partial line
        if self.lines:
            self.lines.pop()

        self._streaming_buffer += token
        self._full_text += token

        # 2. Completed lines → write permanently
        if "\n" in self._streaming_buffer:
            parts = self._streaming_buffer.split("\n")
            for completed_line in parts[:-1]:
                self.write(f"         {completed_line}")
            # Only keep the last incomplete fragment
            self._streaming_buffer = parts[-1]

        # 3. Current partial → temporary line (will be popped next call)
        self.write(f"         {self._streaming_buffer}")
        self.scroll_end(animate=False)

    def stream_end(self) -> str:
        """Finish streaming — return the full accumulated text."""
        text = self._full_text
        self._streaming_buffer = ""
        self._full_text = ""
        self._stream_started = False
        return text

    def add_system(self, text: str) -> None:
        self.write(f"[dim]  {text}[/]")
        self.scroll_end(animate=False)

    def add_error(self, text: str) -> None:
        self.write(f"[bold red]  ✗ {text}[/]")
        self.scroll_end(animate=False)

    def add_tool(self, tool_name: str) -> None:
        self.write(f"[dim magenta]  ⚙ [{tool_name}][/]")
        self.scroll_end(animate=False)
