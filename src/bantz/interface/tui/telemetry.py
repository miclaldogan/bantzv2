"""
Bantz — Hardware Telemetry Collector (#133)

Collects system metrics every 2 seconds in a background thread:
  CPU%, RAM%, Disk%, Net I/O (MB/s delta), CPU Temp, GPU Temp, VRAM.

Design decisions:
  - pynvml (NVML C-level bindings) instead of nvidia-smi subprocess
    → zero process-spawn overhead, works on both desktop and Jetson
  - psutil for CPU/RAM/Disk/Net — native C extension, near-zero overhead
  - collections.deque(maxlen=60) ring buffers → 2 min sliding window
  - All I/O in a dedicated thread via asyncio.to_thread() or @work(thread=True)
    → never blocks the Textual async event loop
  - Net I/O uses delta math: (new_bytes - old_bytes) / interval → MB/s
  - Graceful GPU: if no NVIDIA GPU → gpu_available=False, all GPU metrics=0
  - Thermal throttle alert: CPU > 90°C
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import psutil

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_HISTORY_LEN = 60        # 60 readings × 2s = 2 min
_INTERVAL = 2.0          # seconds
_THERMAL_ALERT_C = 90.0  # CPU thermal throttle threshold


# ═══════════════════════════════════════════════════════════════════════════
# GPU wrapper (pynvml)
# ═══════════════════════════════════════════════════════════════════════════

class _GPUMonitor:
    """Thin wrapper around pynvml — graceful when no GPU."""

    def __init__(self) -> None:
        self._available = False
        self._handle = None
        self._init_done = False

    def init(self) -> bool:
        """Try to initialize NVML. Returns True if GPU is accessible."""
        if self._init_done:
            return self._available
        self._init_done = True
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode()
            log.info("GPU detected: %s", name)
            self._available = True
            self._name = name
        except Exception as exc:
            log.debug("No NVIDIA GPU (pynvml): %s", exc)
            self._available = False
            self._name = ""
        return self._available

    @property
    def available(self) -> bool:
        return self._available

    @property
    def name(self) -> str:
        return self._name if self._available else ""

    def read(self) -> tuple[float, float, float]:
        """Read GPU metrics: (temp_C, vram_used_MB, vram_total_MB).

        Returns (0, 0, 0) if GPU is unavailable.
        """
        if not self._available or self._handle is None:
            return 0.0, 0.0, 0.0
        try:
            import pynvml
            temp = float(pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            ))
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            used_mb = mem.used / (1024 * 1024)
            total_mb = mem.total / (1024 * 1024)
            return temp, used_mb, total_mb
        except Exception:
            return 0.0, 0.0, 0.0

    def shutdown(self) -> None:
        if self._available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._available = False


# ═══════════════════════════════════════════════════════════════════════════
# Snapshot dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TelemetrySnapshot:
    """Single point-in-time reading of all metrics."""
    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    disk_pct: float = 0.0
    net_send_mbps: float = 0.0
    net_recv_mbps: float = 0.0
    cpu_temp: float = 0.0
    gpu_temp: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    thermal_alert: bool = False
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Telemetry Collector
# ═══════════════════════════════════════════════════════════════════════════

class TelemetryCollector:
    """Background hardware telemetry with ring buffer history.

    All reads happen in a separate thread — call collect() from
    asyncio.to_thread() or Textual's @work(thread=True).

    Usage:
        tc = TelemetryCollector()
        tc.start()         # init GPU, take baseline net reading
        snap = tc.collect() # blocking — call from thread
        tc.stop()           # cleanup NVML
    """

    def __init__(self, history_len: int = _HISTORY_LEN) -> None:
        self._gpu = _GPUMonitor()
        self._started = False

        # Ring buffers for sparkline data
        self.cpu_history: deque[float] = deque(maxlen=history_len)
        self.ram_history: deque[float] = deque(maxlen=history_len)
        self.disk_history: deque[float] = deque(maxlen=history_len)
        self.net_send_history: deque[float] = deque(maxlen=history_len)
        self.net_recv_history: deque[float] = deque(maxlen=history_len)
        self.cpu_temp_history: deque[float] = deque(maxlen=history_len)
        self.gpu_temp_history: deque[float] = deque(maxlen=history_len)

        # Peak values for session (#134)
        self.peak_cpu: float = 0.0
        self.peak_ram: float = 0.0
        self.peak_disk: float = 0.0
        self.peak_net_send: float = 0.0
        self.peak_net_recv: float = 0.0
        self.peak_cpu_temp: float = 0.0
        self.peak_gpu_temp: float = 0.0

        # Net I/O delta tracking
        self._last_net_bytes_sent: int = 0
        self._last_net_bytes_recv: int = 0
        self._last_net_time: float = 0.0

        # Latest snapshot
        self.latest: TelemetrySnapshot = TelemetrySnapshot()

    @property
    def gpu_available(self) -> bool:
        return self._gpu.available

    @property
    def gpu_name(self) -> str:
        return self._gpu.name

    def start(self) -> None:
        """Initialize GPU and take net I/O baseline."""
        if self._started:
            return
        self._gpu.init()

        # Baseline for net delta
        net = psutil.net_io_counters()
        self._last_net_bytes_sent = net.bytes_sent
        self._last_net_bytes_recv = net.bytes_recv
        self._last_net_time = time.monotonic()
        self._started = True

    def stop(self) -> None:
        """Shutdown NVML."""
        self._gpu.shutdown()
        self._started = False

    def collect(self) -> TelemetrySnapshot:
        """Collect all metrics — BLOCKING, call from thread.

        Returns a TelemetrySnapshot and appends to ring buffers.
        """
        now = time.monotonic()
        snap = TelemetrySnapshot(timestamp=now)

        # CPU (interval=0 for non-blocking when called frequently)
        snap.cpu_pct = psutil.cpu_percent(interval=0)

        # RAM
        snap.ram_pct = psutil.virtual_memory().percent

        # Disk
        try:
            snap.disk_pct = psutil.disk_usage("/").percent
        except Exception:
            snap.disk_pct = 0.0

        # Network I/O — delta math
        try:
            net = psutil.net_io_counters()
            dt = now - self._last_net_time if self._last_net_time else _INTERVAL
            if dt > 0:
                snap.net_send_mbps = (
                    (net.bytes_sent - self._last_net_bytes_sent) / dt / (1024 * 1024)
                )
                snap.net_recv_mbps = (
                    (net.bytes_recv - self._last_net_bytes_recv) / dt / (1024 * 1024)
                )
            self._last_net_bytes_sent = net.bytes_sent
            self._last_net_bytes_recv = net.bytes_recv
            self._last_net_time = now
        except Exception:
            snap.net_send_mbps = 0.0
            snap.net_recv_mbps = 0.0

        # CPU temperature
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Try common sensor names
                for name in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
                    if name in temps and temps[name]:
                        snap.cpu_temp = temps[name][0].current
                        break
                else:
                    # Fallback: first available sensor
                    first = next(iter(temps.values()))
                    if first:
                        snap.cpu_temp = first[0].current
        except Exception:
            snap.cpu_temp = 0.0

        # Thermal throttle alert
        snap.thermal_alert = snap.cpu_temp > _THERMAL_ALERT_C

        # GPU (pynvml — zero subprocess cost)
        snap.gpu_temp, snap.vram_used_mb, snap.vram_total_mb = self._gpu.read()

        # Append to ring buffers
        self.cpu_history.append(snap.cpu_pct)
        self.ram_history.append(snap.ram_pct)
        self.disk_history.append(snap.disk_pct)
        self.net_send_history.append(snap.net_send_mbps)
        self.net_recv_history.append(snap.net_recv_mbps)
        self.cpu_temp_history.append(snap.cpu_temp)
        self.gpu_temp_history.append(snap.gpu_temp)

        # Update session peaks (#134)
        self.peak_cpu = max(self.peak_cpu, snap.cpu_pct)
        self.peak_ram = max(self.peak_ram, snap.ram_pct)
        self.peak_disk = max(self.peak_disk, snap.disk_pct)
        self.peak_net_send = max(self.peak_net_send, snap.net_send_mbps)
        self.peak_net_recv = max(self.peak_net_recv, snap.net_recv_mbps)
        self.peak_cpu_temp = max(self.peak_cpu_temp, snap.cpu_temp)
        self.peak_gpu_temp = max(self.peak_gpu_temp, snap.gpu_temp)

        self.latest = snap
        return snap

    # ── Convenience ─────────────────────────────────────────────────────

    def vram_pct(self) -> float:
        """VRAM usage as percentage (0-100)."""
        if self.latest.vram_total_mb > 0:
            return (self.latest.vram_used_mb / self.latest.vram_total_mb) * 100
        return 0.0

    def net_total_mbps(self) -> float:
        """Combined upload + download in MB/s."""
        return self.latest.net_send_mbps + self.latest.net_recv_mbps

    def stats(self) -> dict:
        """Summary dict for diagnostics."""
        s = self.latest
        return {
            "cpu": f"{s.cpu_pct:.1f}%",
            "ram": f"{s.ram_pct:.1f}%",
            "disk": f"{s.disk_pct:.1f}%",
            "net": f"↑{s.net_send_mbps:.2f} ↓{s.net_recv_mbps:.2f} MB/s",
            "cpu_temp": f"{s.cpu_temp:.0f}°C" if s.cpu_temp else "n/a",
            "gpu": self._gpu.name or "none",
            "gpu_temp": f"{s.gpu_temp:.0f}°C" if s.gpu_temp else "n/a",
            "vram": f"{s.vram_used_mb:.0f}/{s.vram_total_mb:.0f} MB" if s.vram_total_mb else "n/a",
            "thermal_alert": s.thermal_alert,
            "readings": len(self.cpu_history),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

telemetry = TelemetryCollector()
