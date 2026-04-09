"""Tests for Rich Live TUI migration (#296)."""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.panel import Panel

from bantz.core.event_bus import Event
from bantz.interface.live_ui import (
    LiveUI,
    ServiceDot,
    _bar,
    _DOT_STYLE,
    _log_queue,
    _MouseReader,
    _QueueLogHandler,
    _SGR_MOUSE_RE,
    emit_log,
    emit_log_threadsafe,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ui():
    """Fresh LiveUI instance (no Live context)."""
    return LiveUI()


@pytest.fixture(autouse=True)
def drain_queue():
    """Ensure the global log queue is empty before and after each test."""
    while not _log_queue.empty():
        try:
            _log_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    yield
    while not _log_queue.empty():
        try:
            _log_queue.get_nowait()
        except asyncio.QueueEmpty:
            break


# ── Tests: _bar helper ───────────────────────────────────────────────────────

class TestBar:
    def test_zero(self):
        result = _bar(0)
        assert "░" * 10 in result

    def test_full(self):
        result = _bar(100)
        assert "█" * 10 in result

    def test_half(self):
        result = _bar(50)
        assert "█" * 5 in result

    def test_custom_max(self):
        result = _bar(5, max_val=10)
        assert "█" * 5 in result

    def test_color_green(self):
        assert "green" in _bar(30)

    def test_color_yellow(self):
        assert "yellow" in _bar(70)

    def test_color_red(self):
        assert "red" in _bar(95)

    def test_over_100_clamped(self):
        result = _bar(200)
        assert "█" * 10 in result

    def test_zero_max(self):
        result = _bar(50, max_val=0)
        assert "░" * 10 in result

    def test_custom_width(self):
        result = _bar(100, width=5)
        assert "█" * 5 in result


# ── Tests: ServiceDot ────────────────────────────────────────────────────────

class TestServiceDot:
    def test_all_dots_have_styles(self):
        for s in ServiceDot:
            assert s in _DOT_STYLE

    def test_up_is_green(self):
        assert "green" in _DOT_STYLE[ServiceDot.UP]

    def test_down_is_red(self):
        assert "red" in _DOT_STYLE[ServiceDot.DOWN]

    def test_degraded_is_yellow(self):
        assert "yellow" in _DOT_STYLE[ServiceDot.DEGRADED]


# ── Tests: Layout ────────────────────────────────────────────────────────────

class TestBuildLayout:
    def test_layout_has_header(self, ui):
        layout = ui._build_layout()
        assert layout["header"] is not None

    def test_layout_has_main(self, ui):
        layout = ui._build_layout()
        assert layout["main"] is not None

    def test_layout_has_chat(self, ui):
        layout = ui._build_layout()
        assert layout["chat"] is not None

    def test_main_has_stats_and_logs(self, ui):
        layout = ui._build_layout()
        assert layout["main"]["stats"] is not None
        assert layout["main"]["logs"] is not None


# ── Tests: Add chat / log ────────────────────────────────────────────────────

class TestAddChat:
    def test_add_single(self, ui):
        ui.add_chat("user", "hello")
        assert len(ui._chat_lines) == 1
        assert ui._chat_lines[0] == ("user", "hello")

    def test_add_multiple_roles(self, ui):
        ui.add_chat("user", "hi")
        ui.add_chat("bantz", "hey")
        ui.add_chat("system", "ok")
        ui.add_chat("error", "bad")
        ui.add_chat("tool", "grep")
        assert len(ui._chat_lines) == 5

    def test_max_capacity(self, ui):
        for i in range(150):
            ui.add_chat("user", f"msg {i}")
        assert len(ui._chat_lines) == ui.CHAT_MAX


class TestAddLog:
    def test_add_log_has_timestamp(self, ui):
        ui.add_log("test message")
        assert len(ui._log_lines) == 1
        assert "test message" in ui._log_lines[0]

    def test_add_log_resets_scroll(self, ui):
        ui._scroll_offset = 5
        ui.add_log("new msg")
        assert ui._scroll_offset == 0

    def test_max_capacity(self, ui):
        for i in range(250):
            ui.add_log(f"line {i}")
        assert len(ui._log_lines) == ui.LOG_MAX


# ── Tests: Scrolling ─────────────────────────────────────────────────────────

class TestScrolling:
    def test_scroll_up(self, ui):
        ui._log_lines.extend(["line"] * 50)
        ui._on_scroll(1)
        assert ui._scroll_offset == 3

    def test_scroll_down(self, ui):
        ui._scroll_offset = 10
        ui._on_scroll(-1)
        assert ui._scroll_offset == 7

    def test_scroll_floor_zero(self, ui):
        ui._scroll_offset = 1
        ui._on_scroll(-1)
        assert ui._scroll_offset == 0

    def test_scroll_not_below_zero(self, ui):
        ui._on_scroll(-1)
        assert ui._scroll_offset == 0

    def test_scroll_ceiling(self, ui):
        for _ in range(5):
            ui._log_lines.append("x")
        ui._on_scroll(1)
        ui._on_scroll(1)
        ui._on_scroll(1)
        # should not exceed total lines - 1
        assert ui._scroll_offset <= len(ui._log_lines) - 1


# ── Tests: Render panels ────────────────────────────────────────────────────

class TestRenderHeader:
    def test_returns_panel(self, ui):
        panel = ui._render_header()
        assert isinstance(panel, Panel)

    def test_header_changes_with_services(self, ui):
        ui._services["Ollama"] = ServiceDot.UP
        panel = ui._render_header()
        assert isinstance(panel, Panel)


class TestRenderStats:
    def test_returns_panel(self, ui):
        panel = ui._render_stats()
        assert isinstance(panel, Panel)

    def test_stats_with_values(self, ui):
        ui._cpu = 45.0
        ui._ram_pct = 60.5
        ui._ram_used_gb = 9.6
        ui._ram_total_gb = 16.0
        ui._disk_pct = 40.0
        ui._disk_used_gb = 200.0
        ui._disk_total_gb = 500.0
        panel = ui._render_stats()
        assert isinstance(panel, Panel)

    def test_stats_with_vram(self, ui):
        ui._vram_available = True
        ui._vram_pct = 55.0
        ui._vram_used_mb = 4400.0
        ui._vram_total_mb = 8000.0
        panel = ui._render_stats()
        assert isinstance(panel, Panel)


class TestRenderLogs:
    def test_empty_logs(self, ui):
        panel = ui._render_logs()
        assert isinstance(panel, Panel)

    def test_with_lines(self, ui):
        for i in range(10):
            ui.add_log(f"log line {i}")
        panel = ui._render_logs()
        assert isinstance(panel, Panel)

    def test_with_scroll_offset(self, ui):
        for i in range(100):
            ui.add_log(f"line {i}")
        ui._scroll_offset = 20
        panel = ui._render_logs()
        assert isinstance(panel, Panel)


class TestRenderChat:
    def test_empty_chat(self, ui):
        panel = ui._render_chat()
        assert isinstance(panel, Panel)

    def test_all_roles(self, ui):
        ui.add_chat("user", "hello")
        ui.add_chat("bantz", "hi there")
        ui.add_chat("system", "connected")
        ui.add_chat("error", "oops")
        ui.add_chat("tool", "web_search")
        panel = ui._render_chat()
        assert isinstance(panel, Panel)

    def test_markdown_for_long_msg(self, ui):
        ui.add_chat("bantz", "a" * 400)
        panel = ui._render_chat()
        assert isinstance(panel, Panel)

    def test_markdown_for_code_block(self, ui):
        ui.add_chat("bantz", "Here's code:\n```python\nprint('hi')\n```")
        panel = ui._render_chat()
        assert isinstance(panel, Panel)

    def test_streaming_text(self, ui):
        ui._streaming_text = "partial response"
        panel = ui._render_chat()
        assert isinstance(panel, Panel)

    def test_thinking_indicator(self, ui):
        ui._busy = True
        ui._streaming_text = None
        panel = ui._render_chat()
        assert isinstance(panel, Panel)


class TestUpdatePanels:
    def test_updates_all_slots(self, ui):
        layout = ui._build_layout()
        ui._update_panels(layout)
        # All slots should now have Panel renderables
        for name in ("header", "chat"):
            assert layout[name] is not None
        assert layout["main"]["stats"] is not None
        assert layout["main"]["logs"] is not None


# ── Tests: Mouse reader ─────────────────────────────────────────────────────

class TestMouseReaderParse:
    def test_scroll_up(self):
        scrolls = []
        reader = _MouseReader(lambda d: scrolls.append(d))
        reader._parse(b"\x1b[<64;10;20M")
        assert scrolls == [1]

    def test_scroll_down(self):
        scrolls = []
        reader = _MouseReader(lambda d: scrolls.append(d))
        reader._parse(b"\x1b[<65;10;20M")
        assert scrolls == [-1]

    def test_left_click_ignored(self):
        scrolls = []
        reader = _MouseReader(lambda d: scrolls.append(d))
        reader._parse(b"\x1b[<0;10;20M")
        assert scrolls == []

    def test_multiple_events(self):
        scrolls = []
        reader = _MouseReader(lambda d: scrolls.append(d))
        reader._parse(b"\x1b[<64;5;5M\x1b[<65;5;5M\x1b[<64;5;5M")
        assert scrolls == [1, -1, 1]

    def test_release_event_ignored(self):
        scrolls = []
        reader = _MouseReader(lambda d: scrolls.append(d))
        # 'm' = button release, should be ignored by regex match but
        # btn 64/65 still triggers
        reader._parse(b"\x1b[<64;5;5m")
        assert scrolls == [1]  # SGR uses M for press, m for release

    def test_garbage_data(self):
        scrolls = []
        reader = _MouseReader(lambda d: scrolls.append(d))
        reader._parse(b"random garbage data\x00\xff")
        assert scrolls == []


# ── Tests: Log queue + handler ───────────────────────────────────────────────

class TestEmitLog:
    @pytest.mark.asyncio
    async def test_emit_log_pushes_to_queue(self):
        await emit_log("test message")
        msg = _log_queue.get_nowait()
        assert msg == "test message"


class TestEmitLogThreadsafe:
    @pytest.mark.asyncio
    async def test_threadsafe_pushes_to_queue(self):
        import bantz.interface.live_ui as mod
        old_loop = mod._event_loop
        mod._event_loop = asyncio.get_event_loop()
        try:
            emit_log_threadsafe("threadsafe msg")
            await asyncio.sleep(0.05)  # let call_soon_threadsafe fire
            msg = _log_queue.get_nowait()
            assert msg == "threadsafe msg"
        finally:
            mod._event_loop = old_loop


class TestQueueLogHandler:
    @pytest.mark.asyncio
    async def test_handler_pushes_to_queue(self):
        import bantz.interface.live_ui as mod
        old_loop = mod._event_loop
        mod._event_loop = asyncio.get_event_loop()
        try:
            handler = _QueueLogHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="",
                lineno=0, msg="handler test", args=(), exc_info=None,
            )
            handler.emit(record)
            await asyncio.sleep(0.05)
            msg = _log_queue.get_nowait()
            assert "handler test" in msg
        finally:
            mod._event_loop = old_loop


# ── Tests: Log consumer ─────────────────────────────────────────────────────

class TestLogConsumer:
    @pytest.mark.asyncio
    async def test_consumes_queue_into_log_lines(self):
        ui = LiveUI()
        ui._running = True
        await _log_queue.put("hello from queue")
        task = asyncio.create_task(ui._log_consumer())
        await asyncio.sleep(0.3)
        ui._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert any("hello from queue" in line for line in ui._log_lines)


# ── Tests: Stats collector ───────────────────────────────────────────────────

class TestStatsCollector:
    @pytest.mark.asyncio
    @patch("bantz.interface.live_ui.psutil")
    async def test_collects_stats(self, mock_psutil):
        mock_psutil.cpu_percent.return_value = 42.0
        mock_psutil.virtual_memory.return_value = SimpleNamespace(
            percent=55.0,
            used=8 * (1024 ** 3),
            total=16 * (1024 ** 3),
        )
        mock_psutil.disk_usage.return_value = SimpleNamespace(
            percent=30.0,
            used=200 * (1024 ** 3),
            total=500 * (1024 ** 3),
        )

        ui = LiveUI()
        ui._running = True

        task = asyncio.create_task(ui._stats_collector())
        await asyncio.sleep(0.3)
        ui._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert ui._cpu == 42.0
        assert ui._ram_pct == 55.0
        assert abs(ui._ram_used_gb - 8.0) < 0.1
        assert abs(ui._ram_total_gb - 16.0) < 0.1
        assert ui._disk_pct == 30.0


# ── Tests: VRAM collection ───────────────────────────────────────────────────

class TestVRAM:
    @patch("bantz.interface.live_ui.subprocess")
    def test_vram_nvidia_smi_success(self, mock_sub, ui):
        mock_sub.run.return_value = SimpleNamespace(
            returncode=0, stdout="4400, 8000\n",
        )
        ui._collect_vram()
        assert ui._vram_available is True
        assert ui._vram_used_mb == 4400.0
        assert ui._vram_total_mb == 8000.0
        assert abs(ui._vram_pct - 55.0) < 0.1

    @patch("bantz.interface.live_ui.subprocess")
    def test_vram_nvidia_smi_not_found(self, mock_sub, ui):
        mock_sub.run.side_effect = FileNotFoundError
        ui._collect_vram()
        assert ui._vram_available is False

    @patch("bantz.interface.live_ui.subprocess")
    def test_vram_nvidia_smi_error(self, mock_sub, ui):
        mock_sub.run.return_value = SimpleNamespace(
            returncode=1, stdout="",
        )
        ui._collect_vram()
        assert ui._vram_available is False


# ── Tests: Event bus handlers ────────────────────────────────────────────────

class TestBusHandlers:
    @pytest.mark.asyncio
    async def test_voice_input(self, ui):
        event = Event(name="voice_input", data={"text": "hello"})
        # Should not crash when not busy
        ui._busy = False
        with patch.object(ui, "_process_input", new_callable=AsyncMock):
            ui._on_bus_voice_input(event)
        assert any("🎤 hello" in msg for _, msg in ui._chat_lines)

    def test_voice_input_ignored_when_busy(self, ui):
        event = Event(name="voice_input", data={"text": "hello"})
        ui._busy = True
        ui._on_bus_voice_input(event)
        assert len(ui._chat_lines) == 0

    def test_health_alert(self, ui):
        event = Event(name="health_alert", data={"title": "CPU hot"})
        ui._on_bus_health_alert(event)
        assert any("CPU hot" in msg for _, msg in ui._chat_lines)

    def test_thinking_start(self, ui):
        ui._busy = False
        ui._on_bus_thinking_start(Event(name="thinking_start"))
        assert ui._busy is True

    def test_thinking_done(self, ui):
        ui._busy = True
        ui._on_bus_thinking_done(Event(name="thinking_done"))
        assert ui._busy is False

    def test_planner_step(self, ui):
        event = Event(
            name="planner_step",
            data={"step": 1, "total": 3, "description": "search", "status": "done"},
        )
        ui._on_bus_planner_step(event)
        assert any("Step 1/3" in line for line in ui._log_lines)

    def test_stt_ready(self, ui):
        ui._on_bus_stt_ready(Event(name="stt_model_ready"))
        assert any("Speech recognition ready" in msg for _, msg in ui._chat_lines)

    def test_stt_failed(self, ui):
        ui._on_bus_stt_failed(
            Event(name="stt_model_failed", data={"error": "no model"})
        )
        assert any("no model" in msg for _, msg in ui._chat_lines)


# ── Tests: No Textual imports ────────────────────────────────────────────────

class TestNoTextual:
    def test_no_textual_in_imports(self):
        """Ensure no 'import textual' or 'from textual' in the module."""
        src = inspect.getsource(
            __import__("bantz.interface.live_ui", fromlist=["live_ui"])
        )
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "textual" not in stripped.lower(), (
                    f"Textual import found: {stripped}"
                )

    def test_no_textual_in_module_dict(self):
        import bantz.interface.live_ui as mod
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if hasattr(obj, "__module__") and obj.__module__:
                assert "textual" not in obj.__module__.lower(), (
                    f"{name} imports from textual: {obj.__module__}"
                )


# ── Tests: SGR regex ─────────────────────────────────────────────────────────

class TestSGRRegex:
    def test_matches_press(self):
        m = _SGR_MOUSE_RE.search(b"\x1b[<64;10;20M")
        assert m is not None
        assert int(m.group(1)) == 64

    def test_matches_release(self):
        m = _SGR_MOUSE_RE.search(b"\x1b[<65;10;20m")
        assert m is not None

    def test_no_match_garbage(self):
        m = _SGR_MOUSE_RE.search(b"no mouse here")
        assert m is None


# ── Tests: LiveUI defaults ───────────────────────────────────────────────────

class TestDefaults:
    def test_refresh_fps(self, ui):
        assert ui.REFRESH_FPS == 4

    def test_stats_interval(self, ui):
        assert ui.STATS_INTERVAL == 2.0

    def test_log_max(self, ui):
        assert ui.LOG_MAX == 200

    def test_initial_services(self, ui):
        for name, status in ui._services.items():
            assert status == ServiceDot.UNCONFIGURED

    def test_initial_state(self, ui):
        assert ui._running is True
        assert ui._busy is False
        assert ui._streaming_text is None
        assert ui._scroll_offset == 0
        assert ui._pending is None
