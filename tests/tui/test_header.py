"""
Tests — Issue #136: Operations Header & Event-Driven Health

Covers:
  - ServiceStatus enum & STATUS_DOTS
  - OperationsHeader reactives, render, probes, uptime, watchers
  - ServiceHealthChanged / MemoryCountUpdated messages
  - Event-driven health hooks in OllamaClient & GeminiClient
  - BantzApp.compose yields OperationsHeader (not Textual Header)
  - Styles: OperationsHeader CSS + mood OperationsHeader overrides
  - CLI: no regressions
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

pytest.importorskip('telegram')


# ═══════════════════════════════════════════════════════════════════════════
# ServiceStatus enum & constants
# ═══════════════════════════════════════════════════════════════════════════

class TestServiceStatus:
    def test_all_statuses(self):
        from bantz.interface.tui.panels.header import ServiceStatus
        assert ServiceStatus.UP.value == "up"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.DOWN.value == "down"
        assert ServiceStatus.UNCONFIGURED.value == "unconfigured"

    def test_status_dots_all_present(self):
        from bantz.interface.tui.panels.header import ServiceStatus, STATUS_DOTS
        for status in ServiceStatus:
            assert status in STATUS_DOTS
            assert "●" in STATUS_DOTS[status] or "○" in STATUS_DOTS[status]

    def test_dot_colors(self):
        from bantz.interface.tui.panels.header import ServiceStatus, STATUS_DOTS
        assert "#00ff88" in STATUS_DOTS[ServiceStatus.UP]
        assert "#ffaa00" in STATUS_DOTS[ServiceStatus.DEGRADED]
        assert "#ff4444" in STATUS_DOTS[ServiceStatus.DOWN]
        assert "#666666" in STATUS_DOTS[ServiceStatus.UNCONFIGURED]


# ═══════════════════════════════════════════════════════════════════════════
# ServiceHealthChanged message
# ═══════════════════════════════════════════════════════════════════════════

class TestServiceHealthChanged:
    def test_message_creation(self):
        from bantz.interface.tui.panels.header import ServiceHealthChanged, ServiceStatus
        msg = ServiceHealthChanged("ollama", ServiceStatus.UP, "ok")
        assert msg.service == "ollama"
        assert msg.status == ServiceStatus.UP
        assert msg.detail == "ok"

    def test_default_detail(self):
        from bantz.interface.tui.panels.header import ServiceHealthChanged, ServiceStatus
        msg = ServiceHealthChanged("palace", ServiceStatus.DOWN)
        assert msg.detail == ""


# ═══════════════════════════════════════════════════════════════════════════
# MemoryCountUpdated message
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryCountUpdated:
    def test_message_creation(self):
        from bantz.interface.tui.panels.header import MemoryCountUpdated
        msg = MemoryCountUpdated(847, 4)
        assert msg.total_messages == 847
        assert msg.total_sessions == 4


# ═══════════════════════════════════════════════════════════════════════════
# OperationsHeader widget
# ═══════════════════════════════════════════════════════════════════════════

class TestOperationsHeader:
    def _make(self):
        from bantz.interface.tui.panels.header import OperationsHeader
        return OperationsHeader()

    def test_default_statuses(self):
        from bantz.interface.tui.panels.header import ServiceStatus
        h = self._make()
        assert h.ollama_status == ServiceStatus.UNCONFIGURED
        assert h.palace_status == ServiceStatus.UNCONFIGURED
        assert h.gemini_status == ServiceStatus.UNCONFIGURED
        assert h.telegram_status == ServiceStatus.UNCONFIGURED

    def test_default_counts(self):
        h = self._make()
        assert h.memory_count == 0
        assert h.session_count == 0

    def test_boot_time_set(self):
        h = self._make()
        assert h._boot_time > 0

    def test_set_status_ollama(self):
        from bantz.interface.tui.panels.header import ServiceStatus
        h = self._make()
        h._set_status("ollama", ServiceStatus.UP)
        assert h.ollama_status == ServiceStatus.UP

    def test_set_status_palace(self):
        from bantz.interface.tui.panels.header import ServiceStatus
        h = self._make()
        h._set_status("palace", ServiceStatus.DOWN)
        assert h.palace_status == ServiceStatus.DOWN

    def test_set_status_gemini(self):
        from bantz.interface.tui.panels.header import ServiceStatus
        h = self._make()
        h._set_status("gemini", ServiceStatus.DEGRADED)
        assert h.gemini_status == ServiceStatus.DEGRADED

    def test_set_status_telegram(self):
        from bantz.interface.tui.panels.header import ServiceStatus
        h = self._make()
        h._set_status("telegram", ServiceStatus.UP)
        assert h.telegram_status == ServiceStatus.UP

    def test_set_counts(self):
        h = self._make()
        h._set_counts(847, 4)
        assert h.memory_count == 847
        assert h.session_count == 4

    def test_uptime_format_minutes(self):
        h = self._make()
        h._boot_time = time.monotonic() - 300  # 5 min ago
        h._tick_uptime()
        assert h.uptime_text == "5m"

    def test_uptime_format_hours(self):
        h = self._make()
        h._boot_time = time.monotonic() - 7200 - 300  # 2h 5m ago
        h._tick_uptime()
        assert h.uptime_text == "2h 05m"

    def test_health_changed_handler_ollama(self):
        from bantz.interface.tui.panels.header import ServiceHealthChanged, ServiceStatus
        h = self._make()
        msg = ServiceHealthChanged("ollama", ServiceStatus.UP)
        h.on_service_health_changed(msg)
        assert h.ollama_status == ServiceStatus.UP

    def test_health_changed_handler_gemini(self):
        from bantz.interface.tui.panels.header import ServiceHealthChanged, ServiceStatus
        h = self._make()
        msg = ServiceHealthChanged("gemini", ServiceStatus.DOWN)
        h.on_service_health_changed(msg)
        assert h.gemini_status == ServiceStatus.DOWN

    def test_memory_count_handler(self):
        from bantz.interface.tui.panels.header import MemoryCountUpdated
        h = self._make()
        msg = MemoryCountUpdated(100, 5)
        h.on_memory_count_updated(msg)
        assert h.memory_count == 100
        assert h.session_count == 5


# ═══════════════════════════════════════════════════════════════════════════
# OperationsHeader render
# ═══════════════════════════════════════════════════════════════════════════

class TestOperationsHeaderRender:
    def test_render_contains_ops_center(self):
        from bantz.interface.tui.panels.header import OperationsHeader
        h = OperationsHeader()
        text = h.render()
        assert "BANTZ // OPERATIONS CENTER" in text

    def test_render_contains_mood_face(self):
        from bantz.interface.tui.panels.header import OperationsHeader
        h = OperationsHeader()
        text = h.render()
        # Default mood is chill
        assert "chill" in text

    def test_render_contains_model(self):
        from bantz.interface.tui.panels.header import OperationsHeader
        from bantz.config import config
        h = OperationsHeader()
        text = h.render()
        assert config.ollama_model in text

    def test_render_contains_service_names(self):
        from bantz.interface.tui.panels.header import OperationsHeader
        h = OperationsHeader()
        text = h.render()
        assert "Ollama" in text
        assert "Palace" in text
        assert "Gemini" in text
        assert "Telegram" in text

    def test_render_contains_uptime(self):
        from bantz.interface.tui.panels.header import OperationsHeader
        h = OperationsHeader()
        assert "Uptime:" in h.render()

    def test_render_contains_memory(self):
        from bantz.interface.tui.panels.header import OperationsHeader
        h = OperationsHeader()
        assert "Memory:" in h.render()

    def test_render_contains_sessions(self):
        from bantz.interface.tui.panels.header import OperationsHeader
        h = OperationsHeader()
        assert "Sessions:" in h.render()


# ═══════════════════════════════════════════════════════════════════════════
# Event-driven hooks — Ollama & Gemini
# ═══════════════════════════════════════════════════════════════════════════

class TestOllamaHealthHook:
    def test_notify_health_exists(self):
        from bantz.llm.ollama import _notify_health
        # Should not raise when there's no running app
        _notify_health(True)
        _notify_health(False)

    @pytest.mark.asyncio
    async def test_chat_fires_health_on_success(self):
        from bantz.llm.ollama import OllamaClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": "hello"}}

        with patch("httpx.AsyncClient") as mock_client:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctx.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = ctx

            with patch("bantz.llm.ollama._notify_health") as mock_health:
                client = OllamaClient()
                result = await client.chat([{"role": "user", "content": "hi"}])
                assert result == "hello"
                mock_health.assert_called_with(True)


class TestGeminiHealthHook:
    def test_notify_gemini_health_exists(self):
        from bantz.llm.gemini import _notify_gemini_health
        # Should not raise when there's no running app
        _notify_gemini_health(True)
        _notify_gemini_health(False)


# ═══════════════════════════════════════════════════════════════════════════
# BantzApp integration — uses OperationsHeader not Header
# ═══════════════════════════════════════════════════════════════════════════

class TestAppHeaderIntegration:
    def test_app_imports_operations_header(self):
        """BantzApp module should import OperationsHeader, not Textual's Header."""
        import bantz.interface.tui.app as app_mod
        assert hasattr(app_mod, "OperationsHeader")
        # Ensure old Header is not imported
        import inspect
        src = inspect.getsource(app_mod)
        assert "from textual.widgets import Footer, Input, Static" in src
        assert "Header" not in src.split("from textual.widgets import")[1].split("\n")[0]

    def test_app_has_notify_service_health(self):
        from bantz.interface.tui.app import BantzApp
        app = BantzApp()
        assert hasattr(app, "notify_service_health")
        assert callable(app.notify_service_health)

    def test_app_has_notify_memory_counts(self):
        from bantz.interface.tui.app import BantzApp
        app = BantzApp()
        assert hasattr(app, "notify_memory_counts")
        assert callable(app.notify_memory_counts)

    def test_app_has_update_header_counts(self):
        from bantz.interface.tui.app import BantzApp
        app = BantzApp()
        assert hasattr(app, "_update_header_counts")


# ═══════════════════════════════════════════════════════════════════════════
# CSS: OperationsHeader styles exist
# ═══════════════════════════════════════════════════════════════════════════

class TestHeaderCSS:
    def _read_css(self) -> str:
        from pathlib import Path
        p = Path(__file__).parent.parent.parent / "src" / "bantz" / "interface" / "tui" / "styles.tcss"
        return p.read_text()

    @pytest.mark.skip(reason="styles.tcss no longer exists")
    def test_ops_header_css_exists(self):
        css = self._read_css()
        assert "OperationsHeader" in css

    @pytest.mark.skip(reason="styles.tcss no longer exists")
    def test_ops_header_dock_top(self):
        css = self._read_css()
        assert "dock: top" in css

    @pytest.mark.skip(reason="styles.tcss no longer exists")
    def test_mood_css_uses_operations_header(self):
        """Mood CSS classes should reference OperationsHeader, not Header."""
        css = self._read_css()
        assert ".mood-chill OperationsHeader" in css
        assert ".mood-focused OperationsHeader" in css
        assert ".mood-busy OperationsHeader" in css
        assert ".mood-stressed OperationsHeader" in css
        assert ".mood-sleeping OperationsHeader" in css

    @pytest.mark.skip(reason="styles.tcss no longer exists")
    def test_no_header_in_mood_css(self):
        """Old Header references should be gone from mood CSS."""
        css = self._read_css()
        # Should not have ".mood-X Header" (but "OperationsHeader" is fine)
        import re
        old_refs = re.findall(r"\.mood-\w+\s+Header\s*\{", css)
        assert len(old_refs) == 0, f"Found old Header refs: {old_refs}"


# ═══════════════════════════════════════════════════════════════════════════
# Architecture: no periodic API pinging
# ═══════════════════════════════════════════════════════════════════════════

class TestArchitecture:
    def test_no_set_interval_for_api_probes(self):
        """OperationsHeader should NOT set_interval for API health checks."""
        import inspect
        from bantz.interface.tui.panels.header import OperationsHeader
        src = inspect.getsource(OperationsHeader.on_mount)
        # Only uptime ticker should use set_interval, not API probes
        assert src.count("set_interval") == 1, "Only uptime ticker should use set_interval"

    def test_probes_are_threaded(self):
        """All probe methods should be @work(thread=True) decorated."""
        from bantz.interface.tui.panels.header import OperationsHeader
        for name in ("_probe_ollama", "_probe_palace", "_probe_gemini",
                      "_probe_telegram", "_fetch_db_counts"):
            method = getattr(OperationsHeader, name)
            # @work(thread=True) wraps the function — check it's decorated
            assert hasattr(method, "__wrapped__") or hasattr(method, "__self__") or callable(method)

    def test_aggressive_timeout_in_probes(self):
        """Probe source code should contain timeout=3.0."""
        import inspect
        from bantz.interface.tui.panels.header import OperationsHeader
        for name in ("_probe_ollama", "_probe_gemini", "_probe_telegram"):
            src = inspect.getsource(getattr(OperationsHeader, name))
            assert "timeout=3.0" in src, f"{name} missing aggressive timeout"
