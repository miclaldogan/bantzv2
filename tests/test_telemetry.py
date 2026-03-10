"""
Tests — Issue #133: Hardware Telemetry Collector + TUI Panel

Covers:
  - TelemetryCollector ring buffers, delta math, snapshot, GPU graceful
  - SystemStatus panel composition, threaded collect, GPU hide
  - MetricRow rendering, color thresholds
"""
from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# TelemetryCollector — unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTelemetryCollector:
    """Core telemetry collector tests."""

    def _make(self, history_len: int = 60):
        from bantz.interface.tui.telemetry import TelemetryCollector
        return TelemetryCollector(history_len=history_len)

    # ── Ring buffer ─────────────────────────────────────────────────────

    def test_ring_buffers_maxlen(self):
        tc = self._make(history_len=5)
        assert tc.cpu_history.maxlen == 5
        assert tc.ram_history.maxlen == 5
        assert tc.disk_history.maxlen == 5
        assert tc.net_send_history.maxlen == 5
        assert tc.net_recv_history.maxlen == 5
        assert tc.cpu_temp_history.maxlen == 5
        assert tc.gpu_temp_history.maxlen == 5

    def test_default_maxlen_is_60(self):
        tc = self._make()
        assert tc.cpu_history.maxlen == 60

    def test_ring_buffer_overflow(self):
        tc = self._make(history_len=3)
        tc.start()
        # Collect 5 times — only last 3 should remain
        for _ in range(5):
            tc.collect()
        assert len(tc.cpu_history) == 3
        assert len(tc.ram_history) == 3
        tc.stop()

    # ── Start / Stop ────────────────────────────────────────────────────

    def test_start_sets_baseline(self):
        tc = self._make()
        tc.start()
        assert tc._started is True
        assert tc._last_net_bytes_sent > 0 or tc._last_net_bytes_sent == 0
        assert tc._last_net_time > 0
        tc.stop()

    def test_double_start_is_noop(self):
        tc = self._make()
        tc.start()
        t1 = tc._last_net_time
        tc.start()  # should not reset
        assert tc._last_net_time == t1
        tc.stop()

    def test_stop_clears_started(self):
        tc = self._make()
        tc.start()
        tc.stop()
        assert tc._started is False

    # ── Collect ─────────────────────────────────────────────────────────

    def test_collect_returns_snapshot(self):
        from bantz.interface.tui.telemetry import TelemetrySnapshot
        tc = self._make()
        tc.start()
        snap = tc.collect()
        assert isinstance(snap, TelemetrySnapshot)
        assert 0 <= snap.cpu_pct <= 100
        assert 0 <= snap.ram_pct <= 100
        assert snap.timestamp > 0
        tc.stop()

    def test_collect_appends_to_history(self):
        tc = self._make()
        tc.start()
        tc.collect()
        assert len(tc.cpu_history) == 1
        assert len(tc.ram_history) == 1
        tc.collect()
        assert len(tc.cpu_history) == 2
        tc.stop()

    def test_latest_is_updated(self):
        tc = self._make()
        tc.start()
        tc.collect()
        assert tc.latest.timestamp > 0
        tc.stop()

    # ── Network delta math ──────────────────────────────────────────────

    def test_net_delta_positive(self):
        """Net rate should never be negative."""
        tc = self._make()
        tc.start()
        snap = tc.collect()
        assert snap.net_send_mbps >= 0
        assert snap.net_recv_mbps >= 0
        tc.stop()

    def test_net_delta_math_manual(self):
        """Verify delta math with known values."""
        tc = self._make()
        tc.start()
        # Manually set known baseline
        tc._last_net_bytes_sent = 0
        tc._last_net_bytes_recv = 0
        tc._last_net_time = time.monotonic() - 2.0  # 2 seconds ago

        with patch("psutil.net_io_counters") as mock_net:
            mock_net.return_value = SimpleNamespace(
                bytes_sent=2 * 1024 * 1024,  # 2 MB
                bytes_recv=4 * 1024 * 1024,  # 4 MB
            )
            snap = tc.collect()
        # Rate should be ~1 MB/s send, ~2 MB/s recv (over ~2s)
        assert 0.5 < snap.net_send_mbps < 1.5
        assert 1.5 < snap.net_recv_mbps < 2.5
        tc.stop()

    def test_net_total_mbps(self):
        tc = self._make()
        tc.start()
        tc.collect()
        total = tc.net_total_mbps()
        assert total >= 0
        tc.stop()

    # ── CPU temperature ─────────────────────────────────────────────────

    def test_cpu_temp_with_coretemp(self):
        tc = self._make()
        tc.start()
        with patch("psutil.sensors_temperatures") as mock_t:
            mock_t.return_value = {
                "coretemp": [SimpleNamespace(current=72.0, high=100.0, critical=110.0)],
            }
            snap = tc.collect()
        assert snap.cpu_temp == 72.0
        tc.stop()

    def test_cpu_temp_fallback_to_first_sensor(self):
        tc = self._make()
        tc.start()
        with patch("psutil.sensors_temperatures") as mock_t:
            mock_t.return_value = {
                "custom_sensor": [SimpleNamespace(current=55.0, high=90.0, critical=100.0)],
            }
            snap = tc.collect()
        assert snap.cpu_temp == 55.0
        tc.stop()

    def test_cpu_temp_empty_sensors(self):
        tc = self._make()
        tc.start()
        with patch("psutil.sensors_temperatures") as mock_t:
            mock_t.return_value = {}
            snap = tc.collect()
        assert snap.cpu_temp == 0.0
        tc.stop()

    def test_cpu_temp_no_sensors_function(self):
        """On macOS/Windows psutil may not have sensors_temperatures."""
        tc = self._make()
        tc.start()
        with patch("psutil.sensors_temperatures", side_effect=AttributeError):
            snap = tc.collect()
        assert snap.cpu_temp == 0.0
        tc.stop()

    # ── Thermal alert ───────────────────────────────────────────────────

    def test_thermal_alert_above_90(self):
        tc = self._make()
        tc.start()
        with patch("psutil.sensors_temperatures") as mock_t:
            mock_t.return_value = {
                "coretemp": [SimpleNamespace(current=95.0, high=100.0, critical=110.0)],
            }
            snap = tc.collect()
        assert snap.thermal_alert is True
        tc.stop()

    def test_no_thermal_alert_below_90(self):
        tc = self._make()
        tc.start()
        with patch("psutil.sensors_temperatures") as mock_t:
            mock_t.return_value = {
                "coretemp": [SimpleNamespace(current=70.0, high=100.0, critical=110.0)],
            }
            snap = tc.collect()
        assert snap.thermal_alert is False
        tc.stop()

    def test_thermal_at_exactly_90_no_alert(self):
        tc = self._make()
        tc.start()
        with patch("psutil.sensors_temperatures") as mock_t:
            mock_t.return_value = {
                "coretemp": [SimpleNamespace(current=90.0, high=100.0, critical=110.0)],
            }
            snap = tc.collect()
        assert snap.thermal_alert is False  # > 90, not >=
        tc.stop()

    # ── GPU graceful ────────────────────────────────────────────────────

    def test_gpu_not_available_by_default(self):
        """Without NVIDIA hardware, GPU should be unavailable."""
        tc = self._make()
        # Don't call start() — GPU init not attempted
        assert tc.gpu_available is False

    def test_gpu_graceful_when_pynvml_fails(self):
        tc = self._make()
        with patch.dict("sys.modules", {"pynvml": None}):
            tc.start()
        assert tc.gpu_available is False
        snap = tc.collect()
        assert snap.gpu_temp == 0.0
        assert snap.vram_used_mb == 0.0
        assert snap.vram_total_mb == 0.0
        tc.stop()

    def test_vram_pct_zero_when_no_gpu(self):
        tc = self._make()
        # Force GPU unavailable by disabling init
        with patch.object(tc._collector if hasattr(tc, '_collector') else tc, '_gpu') as mock_gpu:
            mock_gpu.available = False
            mock_gpu.init.return_value = False
            mock_gpu.read.return_value = (0.0, 0.0, 0.0)
            tc.start()
            tc.collect()
            assert tc.vram_pct() == 0.0
            tc.stop()

    # ── Stats dict ──────────────────────────────────────────────────────

    def test_stats_returns_dict(self):
        tc = self._make()
        tc.start()
        tc.collect()
        s = tc.stats()
        assert "cpu" in s
        assert "ram" in s
        assert "disk" in s
        assert "net" in s
        assert "thermal_alert" in s
        assert "readings" in s
        assert s["readings"] == 1
        tc.stop()

    def test_stats_readings_increment(self):
        tc = self._make()
        tc.start()
        for _ in range(3):
            tc.collect()
        assert tc.stats()["readings"] == 3
        tc.stop()


# ═══════════════════════════════════════════════════════════════════════════
# TelemetrySnapshot — unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTelemetrySnapshot:
    def test_defaults(self):
        from bantz.interface.tui.telemetry import TelemetrySnapshot
        s = TelemetrySnapshot()
        assert s.cpu_pct == 0.0
        assert s.ram_pct == 0.0
        assert s.thermal_alert is False
        assert s.timestamp == 0.0

    def test_custom_values(self):
        from bantz.interface.tui.telemetry import TelemetrySnapshot
        s = TelemetrySnapshot(cpu_pct=42.0, thermal_alert=True)
        assert s.cpu_pct == 42.0
        assert s.thermal_alert is True


# ═══════════════════════════════════════════════════════════════════════════
# _GPUMonitor — unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGPUMonitor:
    def _make(self):
        from bantz.interface.tui.telemetry import _GPUMonitor
        return _GPUMonitor()

    def test_not_available_before_init(self):
        g = self._make()
        assert g.available is False
        assert g.name == ""

    def test_read_returns_zeros_before_init(self):
        g = self._make()
        assert g.read() == (0.0, 0.0, 0.0)

    def test_init_only_runs_once(self):
        g = self._make()
        g.init()
        result = g.init()  # second call
        assert isinstance(result, bool)

    def test_shutdown_is_safe_when_not_available(self):
        g = self._make()
        g.shutdown()  # should not raise

    def test_mock_gpu_available(self):
        """Simulate a GPU being present via pynvml mocks."""
        g = self._make()
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")
        mock_pynvml.nvmlDeviceGetName = MagicMock(return_value="Test GPU")
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = g.init()
        assert result is True
        assert g.available is True
        assert g.name == "Test GPU"

    def test_mock_gpu_read(self):
        """Simulate GPU metric reading."""
        g = self._make()
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")
        mock_pynvml.nvmlDeviceGetName = MagicMock(return_value="Test GPU")
        mock_pynvml.NVML_TEMPERATURE_GPU = 0
        mock_pynvml.nvmlDeviceGetTemperature = MagicMock(return_value=65)
        mem_info = SimpleNamespace(
            used=2048 * 1024 * 1024,  # 2 GB
            total=8192 * 1024 * 1024,  # 8 GB
        )
        mock_pynvml.nvmlDeviceGetMemoryInfo = MagicMock(return_value=mem_info)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            g.init()
            temp, used, total = g.read()
        assert temp == 65.0
        assert abs(used - 2048.0) < 1
        assert abs(total - 8192.0) < 1

    def test_gpu_name_bytes_decoded(self):
        """pynvml sometimes returns bytes for device name."""
        g = self._make()
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")
        mock_pynvml.nvmlDeviceGetName = MagicMock(return_value=b"NVIDIA RTX 3090")
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            g.init()
        assert g.name == "NVIDIA RTX 3090"


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

class TestModuleSingleton:
    def test_singleton_exists(self):
        from bantz.interface.tui.telemetry import telemetry
        from bantz.interface.tui.telemetry import TelemetryCollector
        assert isinstance(telemetry, TelemetryCollector)


# ═══════════════════════════════════════════════════════════════════════════
# MetricRow — rendering
# ═══════════════════════════════════════════════════════════════════════════

class TestMetricRow:
    def test_render_green(self):
        from bantz.interface.tui.panels.system import MetricRow
        row = MetricRow("CPU", "%")
        row.value = 30.0
        text = row.render()
        assert "green" in text
        assert "30%" in text

    def test_render_yellow(self):
        from bantz.interface.tui.panels.system import MetricRow
        row = MetricRow("CPU", "%")
        row.value = 75.0
        text = row.render()
        assert "yellow" in text

    def test_render_red(self):
        from bantz.interface.tui.panels.system import MetricRow
        row = MetricRow("CPU", "%")
        row.value = 90.0
        text = row.render()
        assert "red" in text

    def test_render_mbps_unit(self):
        from bantz.interface.tui.panels.system import MetricRow
        row = MetricRow("↑ TX", " MB/s", max_value=10.0)
        row.value = 3.14
        text = row.render()
        assert "MB/s" in text
        assert "3.1" in text

    def test_clamps_at_100_percent(self):
        from bantz.interface.tui.panels.system import MetricRow
        row = MetricRow("CPU", "%")
        row.value = 150.0  # over 100
        text = row.render()
        # Should render full bar (10 filled) without crash
        assert "150%" in text

    def test_zero_max_value(self):
        from bantz.interface.tui.panels.system import MetricRow
        row = MetricRow("X", "%", max_value=0)
        row.value = 50.0
        text = row.render()
        # pct becomes 0 when max_value is 0 — no crash
        assert "50%" in text


# ═══════════════════════════════════════════════════════════════════════════
# VRAM percentage
# ═══════════════════════════════════════════════════════════════════════════

class TestVRAMPct:
    def test_vram_pct_with_data(self):
        from bantz.interface.tui.telemetry import TelemetryCollector, TelemetrySnapshot
        tc = TelemetryCollector()
        tc.latest = TelemetrySnapshot(vram_used_mb=2048, vram_total_mb=8192)
        assert abs(tc.vram_pct() - 25.0) < 0.1

    def test_vram_pct_zero_total(self):
        from bantz.interface.tui.telemetry import TelemetryCollector, TelemetrySnapshot
        tc = TelemetryCollector()
        tc.latest = TelemetrySnapshot(vram_used_mb=0, vram_total_mb=0)
        assert tc.vram_pct() == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Disk error resilience
# ═══════════════════════════════════════════════════════════════════════════

class TestDiskResilience:
    def test_disk_returns_zero_on_error(self):
        from bantz.interface.tui.telemetry import TelemetryCollector
        tc = TelemetryCollector()
        tc.start()
        with patch("psutil.disk_usage", side_effect=PermissionError("denied")):
            snap = tc.collect()
        assert snap.disk_pct == 0.0
        tc.stop()
