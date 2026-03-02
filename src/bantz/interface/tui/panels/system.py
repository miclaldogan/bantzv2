"""
Bantz — System Status Panel

Displays CPU, RAM, disk usage and a live clock in the TUI right panel.
"""
from __future__ import annotations

from datetime import datetime

from textual.widgets import Static
from textual.reactive import reactive
from textual import work


class SystemStatus(Static):
    """Live system metrics widget."""

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
