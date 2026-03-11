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
from bantz.interface.tui.mood import (
    mood_machine, Mood, ALL_MOOD_CLASSES,
)

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

    peak: reactive[float] = reactive(0.0)

    def update_data(self, value: float, history: list[float], peak: float = 0.0) -> None:
        self.value = value
        self.peak = peak
        spark = self.query_one(Sparkline)
        spark.data = list(history)
        self.refresh()

    def render(self) -> str:
        # Return the current value as a compact bar with peak label
        pct = min(self.value / self._max_value * 100, 100) if self._max_value else 0
        filled = int(pct / 100 * 10)
        color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
        peak_str = ""
        if self.peak > 0 and self.peak != self.value:
            if self._unit == "%":
                peak_str = f" [dim](↑{self.peak:.0f})[/]"
            else:
                peak_str = f" [dim](↑{self.peak:.1f})[/]"
        if self._unit == "%":
            return f"[dim]{self._label:<4}[/] [{color}]{'█' * filled}{'░' * (10 - filled)}[/] {self.value:.0f}{self._unit}{peak_str}"
        return f"[dim]{self._label:<4}[/] [{color}]{'█' * filled}{'░' * (10 - filled)}[/] {self.value:.1f}{self._unit}{peak_str}"


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
        # Init mood history with app DB
        try:
            from bantz.config import config
            if not mood_machine.history.initialized:
                mood_machine.history.init(config.db_path)
        except Exception:
            pass
        self._collect_telemetry()
        self.set_interval(_REFRESH_INTERVAL, self._collect_telemetry)
        self.set_interval(1, self._tick_clock)
        # Initial mood CSS class
        self._apply_mood_class(mood_machine.current)

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
            snap.cpu_pct, list(c.cpu_history), c.peak_cpu
        )
        self.query_one("#metric-ram", MetricRow).update_data(
            snap.ram_pct, list(c.ram_history), c.peak_ram
        )
        self.query_one("#metric-disk", MetricRow).update_data(
            snap.disk_pct, list(c.disk_history), c.peak_disk
        )

        # Network
        self.query_one("#metric-net-tx", MetricRow).update_data(
            snap.net_send_mbps, list(c.net_send_history), c.peak_net_send
        )
        self.query_one("#metric-net-rx", MetricRow).update_data(
            snap.net_recv_mbps, list(c.net_recv_history), c.peak_net_recv
        )

        # Thermal
        self.query_one("#metric-cpu-temp", MetricRow).update_data(
            snap.cpu_temp, list(c.cpu_temp_history), c.peak_cpu_temp
        )
        alert = self.query_one("#thermal-alert", Static)
        if snap.thermal_alert:
            alert.update("[bold red blink]⚠ THERMAL THROTTLE[/]")
        else:
            alert.update("")

        # GPU
        if c.gpu_available:
            self.query_one("#metric-gpu-temp", MetricRow).update_data(
                snap.gpu_temp, list(c.gpu_temp_history), c.peak_gpu_temp
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

        # ── Mood evaluation (#135) ──────────────────────────────────
        self._evaluate_mood(snap)

    def _evaluate_mood(self, snap) -> None:
        """Run mood state machine and update TUI."""
        # Get activity from AppDetector (graceful fallback)
        activity = "idle"
        try:
            from bantz.agent.app_detector import app_detector
            activity = app_detector.get_activity_category().value
        except Exception:
            pass

        # Get observer error count (graceful fallback)
        obs_errors = 0
        try:
            from bantz.agent.observer import observer
            if observer.running:
                obs_errors = observer.stats().get("total_events", 0)
        except Exception:
            pass

        prev = mood_machine.current
        mood_machine.evaluate(
            cpu_pct=snap.cpu_pct,
            ram_pct=snap.ram_pct,
            thermal_alert=snap.thermal_alert,
            activity=activity,
            observer_error_count=obs_errors,
        )

        # Swap CSS class on app if mood changed
        if mood_machine.current != prev:
            self._apply_mood_class(mood_machine.current)

    def _apply_mood_class(self, mood: Mood) -> None:
        """Swap mood CSS class on the app screen — flicker-free."""
        try:
            for cls in ALL_MOOD_CLASSES:
                self.app.remove_class(cls)
            self.app.add_class(mood_machine.css_class)
        except Exception:
            pass
