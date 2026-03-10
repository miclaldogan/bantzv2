"""
Bantz — System Status Panel (#133)

Real-time hardware telemetry with sparkline history in the TUI right panel.
All metric collection runs in a background thread via @work(thread=True)
so the Textual event loop is never blocked.
"""
from __future__ import annotations

from datetime import datetime

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Label, Sparkline, Static

from bantz.interface.tui.telemetry import TelemetryCollector

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_REFRESH_INTERVAL = 2  # seconds — matches telemetry collector


# ═══════════════════════════════════════════════════════════════════════════
# Metric Row — label + bar + sparkline
# ═══════════════════════════════════════════════════════════════════════════

class MetricRow(Static):
    """Single metric: label, current value bar, sparkline."""

    value: reactive[float] = reactive(0.0)

    def __init__(
        self,
        label: str,
        unit: str = "%",
        max_value: float = 100.0,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._label = label
        self._unit = unit
        self._max_value = max_value

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="metric-label")
        yield Sparkline([], id=f"spark-{self.id}" if self.id else None)

    def update_data(self, value: float, history: list[float]) -> None:
        self.value = value
        spark = self.query_one(Sparkline)
        spark.data = list(history)
        self.refresh()

    def render(self) -> str:
        # Return the current value as a compact bar
        pct = min(self.value / self._max_value * 100, 100) if self._max_value else 0
        filled = int(pct / 100 * 10)
        color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
        if self._unit == "%":
            return f"[dim]{self._label:<4}[/] [{color}]{'█' * filled}{'░' * (10 - filled)}[/] {self.value:.0f}{self._unit}"
        return f"[dim]{self._label:<4}[/] [{color}]{'█' * filled}{'░' * (10 - filled)}[/] {self.value:.1f}{self._unit}"


# ═══════════════════════════════════════════════════════════════════════════
# System Status Panel
# ═══════════════════════════════════════════════════════════════════════════

class SystemStatus(Vertical):
    """Live hardware telemetry panel with sparkline history.

    Metrics: CPU%, RAM%, Disk%, Net I/O (MB/s), CPU Temp, GPU Temp, VRAM.
    All collection happens in a background thread (never blocks the event loop).
    Ring buffers: 60 readings × 2s = 2 min sliding window.
    """

    clock: reactive[str] = reactive("")

    def __init__(self, *, collector: TelemetryCollector | None = None) -> None:
        super().__init__()
        self._collector = collector or TelemetryCollector()

    def compose(self) -> ComposeResult:
        yield Static("[bold cyan]── TELEMETRY ───────[/]", classes="section-head")
        yield MetricRow("CPU", "%", id="metric-cpu")
        yield MetricRow("RAM", "%", id="metric-ram")
        yield MetricRow("DISK", "%", id="metric-disk")

        yield Static("[bold cyan]── NETWORK ─────────[/]", classes="section-head")
        yield MetricRow("↑ TX", " MB/s", max_value=10.0, id="metric-net-tx")
        yield MetricRow("↓ RX", " MB/s", max_value=10.0, id="metric-net-rx")

        yield Static("[bold cyan]── THERMAL ─────────[/]", classes="section-head")
        yield MetricRow("CPU°", "°C", max_value=100.0, id="metric-cpu-temp")
        yield Static("", id="thermal-alert")

        yield Static("[bold cyan]── GPU ─────────────[/]", id="gpu-section-head", classes="section-head")
        yield MetricRow("GPU°", "°C", max_value=100.0, id="metric-gpu-temp")
        yield Static("", id="vram-bar")

        yield Static("", classes="section-head")
        yield Static("", id="clock-display")

    def on_mount(self) -> None:
        self._collector.start()
        self._collect_telemetry()
        self.set_interval(_REFRESH_INTERVAL, self._collect_telemetry)
        self.set_interval(1, self._tick_clock)

        # Hide GPU section if no GPU detected
        if not self._collector.gpu_available:
            for wid in ("gpu-section-head", "metric-gpu-temp", "vram-bar"):
                try:
                    self.query_one(f"#{wid}").display = False
                except Exception:
                    pass

    def on_unmount(self) -> None:
        self._collector.stop()

    # ── Clock ───────────────────────────────────────────────────────────

    def _tick_clock(self) -> None:
        self.clock = datetime.now().strftime("%H:%M:%S")
        try:
            clock_w = self.query_one("#clock-display", Static)
            now = self.clock
            date = datetime.now().strftime("%A, %d %B %Y")
            clock_w.update(f"[bold white]  {now}[/]\n[dim]  {date}[/]")
        except Exception:
            pass

    # ── Telemetry (background thread) ───────────────────────────────────

    @work(thread=True)
    def _collect_telemetry(self) -> None:
        """Runs in a thread — never blocks the Textual event loop."""
        snap = self._collector.collect()
        self.app.call_from_thread(self._apply_snapshot, snap)

    def _apply_snapshot(self, snap) -> None:
        """Apply collected data to widgets — runs on the main thread."""
        c = self._collector

        # Core metrics
        self.query_one("#metric-cpu", MetricRow).update_data(
            snap.cpu_pct, list(c.cpu_history)
        )
        self.query_one("#metric-ram", MetricRow).update_data(
            snap.ram_pct, list(c.ram_history)
        )
        self.query_one("#metric-disk", MetricRow).update_data(
            snap.disk_pct, list(c.disk_history)
        )

        # Network
        self.query_one("#metric-net-tx", MetricRow).update_data(
            snap.net_send_mbps, list(c.net_send_history)
        )
        self.query_one("#metric-net-rx", MetricRow).update_data(
            snap.net_recv_mbps, list(c.net_recv_history)
        )

        # Thermal
        self.query_one("#metric-cpu-temp", MetricRow).update_data(
            snap.cpu_temp, list(c.cpu_temp_history)
        )
        alert = self.query_one("#thermal-alert", Static)
        if snap.thermal_alert:
            alert.update("[bold red blink]⚠ THERMAL THROTTLE[/]")
        else:
            alert.update("")

        # GPU
        if c.gpu_available:
            self.query_one("#metric-gpu-temp", MetricRow).update_data(
                snap.gpu_temp, list(c.gpu_temp_history)
            )
            vram = self.query_one("#vram-bar", Static)
            pct = c.vram_pct()
            used = snap.vram_used_mb
            total = snap.vram_total_mb
            color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
            filled = int(pct / 100 * 10)
            vram.update(
                f"[dim]VRAM[/] [{color}]{'█' * filled}{'░' * (10 - filled)}[/] "
                f"{used:.0f}/{total:.0f}MB"
            )
