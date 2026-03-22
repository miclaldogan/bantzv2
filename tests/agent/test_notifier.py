"""Tests for bantz.agent.notifier (#153)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from unittest.mock import patch


from bantz.agent.notifier import (
    Notifier,
    _URGENCY_MAP,
    _DEFAULT_URGENCY,
    _TUI_IDENTIFIERS,
    notifier,
)
from bantz.agent.interventions import Priority as RealPriority


# ═══════════════════════════════════════════════════════════════════════════
# Fake Intervention / Priority for tests (avoid circular import)
# ═══════════════════════════════════════════════════════════════════════════


class FakePriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FakeType(str, Enum):
    ROUTINE = "routine"
    ERROR_ALERT = "error_alert"


@dataclass
class FakeIntervention:
    title: str = "Test"
    reason: str = "Testing"
    priority: FakePriority = FakePriority.MEDIUM
    type: FakeType = FakeType.ROUTINE
    source: str = "test"


# ═══════════════════════════════════════════════════════════════════════════
# Urgency mapping
# ═══════════════════════════════════════════════════════════════════════════


class TestUrgencyMap:
    def test_critical(self):
        urgency, ms = _URGENCY_MAP["critical"]
        assert urgency == "critical"
        assert ms == 0  # sticky

    def test_high(self):
        urgency, ms = _URGENCY_MAP["high"]
        assert urgency == "critical"
        assert ms == 10_000

    def test_medium(self):
        urgency, ms = _URGENCY_MAP["medium"]
        assert urgency == "normal"
        assert ms == 5_000

    def test_low(self):
        urgency, ms = _URGENCY_MAP["low"]
        assert urgency == "low"
        assert ms == 3_000

    def test_default_urgency(self):
        assert _DEFAULT_URGENCY == ("normal", 5_000)

    def test_all_priorities_mapped(self):
        for p in FakePriority:
            assert p.value in _URGENCY_MAP


# ═══════════════════════════════════════════════════════════════════════════
# Notifier init
# ═══════════════════════════════════════════════════════════════════════════


class TestNotifierInit:
    def _make(self) -> Notifier:
        return Notifier()

    def test_not_initialized_by_default(self):
        n = self._make()
        assert not n.initialized
        assert not n.available
        assert not n.enabled

    def test_init_with_notify_send(self):
        n = self._make()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init()
        assert n.initialized
        assert n.available
        assert n.enabled

    def test_init_without_notify_send(self):
        n = self._make()
        with patch("bantz.agent.notifier.shutil.which", return_value=None):
            n.init()
        assert n.initialized
        assert not n.available
        assert not n.enabled

    def test_init_disabled(self):
        n = self._make()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init(enabled=False)
        assert n.initialized
        assert n.available
        assert not n.enabled  # available but disabled

    def test_init_idempotent(self):
        n = self._make()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init()
        with patch("bantz.agent.notifier.shutil.which", return_value=None):
            n.init()
        # Should still be available — second init is no-op
        assert n.available

    def test_init_with_icon(self):
        n = self._make()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init(icon="/path/to/icon.png")
        assert n._icon == "/path/to/icon.png"

    def test_init_with_sound(self):
        n = self._make()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init(sound=True)
        assert n._sound is True


# ═══════════════════════════════════════════════════════════════════════════
# send() method
# ═══════════════════════════════════════════════════════════════════════════


class TestSend:
    def _make_ready(self) -> Notifier:
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init()
        return n

    def test_send_basic(self):
        n = self._make_ready()
        with patch("bantz.agent.notifier.subprocess.run") as mock_run:
            result = n.send("Hello", "World")
        assert result is True
        assert n._sent_count == 1
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/notify-send"
        assert "--app-name" in cmd
        assert "Bantz" in cmd
        assert "Hello" in cmd
        assert "World" in cmd

    def test_send_urgency(self):
        n = self._make_ready()
        with patch("bantz.agent.notifier.subprocess.run") as mock_run:
            n.send("T", "B", urgency="critical", expire_ms=0)
        cmd = mock_run.call_args[0][0]
        assert "--urgency" in cmd
        idx = cmd.index("--urgency")
        assert cmd[idx + 1] == "critical"
        # expire_ms=0 means no --expire-time
        assert "--expire-time" not in cmd

    def test_send_with_expire(self):
        n = self._make_ready()
        with patch("bantz.agent.notifier.subprocess.run") as mock_run:
            n.send("T", "B", expire_ms=5000)
        cmd = mock_run.call_args[0][0]
        assert "--expire-time" in cmd
        idx = cmd.index("--expire-time")
        assert cmd[idx + 1] == "5000"

    def test_send_with_icon(self):
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init(icon="/path/icon.png")
        with patch("bantz.agent.notifier.subprocess.run") as mock_run:
            n.send("T", "B")
        cmd = mock_run.call_args[0][0]
        assert "--icon" in cmd
        assert "/path/icon.png" in cmd

    def test_send_no_icon(self):
        n = self._make_ready()
        with patch("bantz.agent.notifier.subprocess.run") as mock_run:
            n.send("T", "B")
        cmd = mock_run.call_args[0][0]
        assert "--icon" not in cmd

    def test_send_disabled(self):
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init(enabled=False)
        with patch("bantz.agent.notifier.subprocess.run") as mock_run:
            result = n.send("T", "B")
        assert result is False
        mock_run.assert_not_called()

    def test_send_not_available(self):
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value=None):
            n.init()
        result = n.send("T", "B")
        assert result is False

    def test_send_timeout_error(self):
        n = self._make_ready()
        with patch("bantz.agent.notifier.subprocess.run", side_effect=subprocess.TimeoutExpired("x", 3)):
            result = n.send("T", "B")
        assert result is False

    def test_send_file_not_found(self):
        n = self._make_ready()
        with patch("bantz.agent.notifier.subprocess.run", side_effect=FileNotFoundError):
            result = n.send("T", "B")
        assert result is False

    def test_send_os_error(self):
        n = self._make_ready()
        with patch("bantz.agent.notifier.subprocess.run", side_effect=OSError):
            result = n.send("T", "B")
        assert result is False

    def test_sent_count_increments(self):
        n = self._make_ready()
        with patch("bantz.agent.notifier.subprocess.run"):
            n.send("A", "B")
            n.send("C", "D")
        assert n._sent_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# TUI active window detection
# ═══════════════════════════════════════════════════════════════════════════


class TestIsTuiActive:
    def _make_ready(self) -> Notifier:
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init()
        return n

    def test_tui_active_bantz_in_title(self):
        n = self._make_ready()
        from bantz.agent.app_detector import WindowInfo
        win = WindowInfo(name="Alacritty", title="python -m bantz", pid=100)
        with patch("bantz.agent.app_detector.app_detector") as mock_ad:
            mock_ad.initialized = True
            mock_ad.get_active_window.return_value = win
            result = n._is_tui_active()
        assert result is True

    def test_tui_not_active_browser(self):
        n = self._make_ready()
        from bantz.agent.app_detector import WindowInfo
        win = WindowInfo(name="firefox", title="GitHub - Firefox", pid=200)
        with patch("bantz.agent.app_detector.app_detector") as mock_ad:
            mock_ad.initialized = True
            mock_ad.get_active_window.return_value = win
            result = n._is_tui_active()
        assert result is False

    def test_tui_no_window(self):
        n = self._make_ready()
        with patch("bantz.agent.app_detector.app_detector") as mock_ad:
            mock_ad.initialized = True
            mock_ad.get_active_window.return_value = None
            result = n._is_tui_active()
        assert result is False

    def test_tui_app_detector_not_initialized(self):
        n = self._make_ready()
        with patch("bantz.agent.app_detector.app_detector") as mock_ad:
            mock_ad.initialized = False
            result = n._is_tui_active()
        assert result is False

    def test_tui_exception_returns_false(self):
        n = self._make_ready()
        with patch("bantz.agent.app_detector.app_detector", side_effect=Exception("boom")):
            result = n._is_tui_active()
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════
# dispatch() — smart delivery
# ═══════════════════════════════════════════════════════════════════════════


class TestDispatch:
    def _make_ready(self) -> Notifier:
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init()
        return n

    def test_dispatch_sends_notification(self):
        n = self._make_ready()
        iv = FakeIntervention(title="Break time", reason="You've been coding for 2h", priority=FakePriority.MEDIUM)
        with patch.object(n, "_is_tui_active", return_value=False), \
             patch.object(n, "send", return_value=True) as mock_send:
            result = n.dispatch(iv)
        assert result is True
        mock_send.assert_called_once_with(
            "Bantz: Break time",
            "You've been coding for 2h",
            urgency="normal",
            expire_ms=5000,
        )

    def test_dispatch_critical(self):
        n = self._make_ready()
        iv = FakeIntervention(priority=FakePriority.CRITICAL, title="Error", reason="Disk full")
        with patch.object(n, "_is_tui_active", return_value=False), \
             patch.object(n, "send", return_value=True) as mock_send:
            n.dispatch(iv)
        mock_send.assert_called_once_with(
            "Bantz: Error",
            "Disk full",
            urgency="critical",
            expire_ms=0,
        )

    def test_dispatch_low(self):
        n = self._make_ready()
        iv = FakeIntervention(priority=FakePriority.LOW, title="Tip", reason="Try this")
        with patch.object(n, "_is_tui_active", return_value=False), \
             patch.object(n, "send", return_value=True) as mock_send:
            n.dispatch(iv)
        mock_send.assert_called_once_with(
            "Bantz: Tip",
            "Try this",
            urgency="low",
            expire_ms=3000,
        )

    def test_dispatch_skipped_tui_active(self):
        n = self._make_ready()
        iv = FakeIntervention()
        with patch.object(n, "_is_tui_active", return_value=True), \
             patch.object(n, "send") as mock_send:
            result = n.dispatch(iv)
        assert result is False
        mock_send.assert_not_called()
        assert n._skipped_count == 1

    def test_dispatch_skipped_disabled(self):
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init(enabled=False)
        iv = FakeIntervention()
        result = n.dispatch(iv)
        assert result is False
        assert n._skipped_count == 1

    def test_dispatch_skipped_not_initialized(self):
        n = Notifier()
        iv = FakeIntervention()
        result = n.dispatch(iv)
        assert result is False
        assert n._skipped_count == 1

    def test_dispatch_focus_mode_blocks_medium(self):
        """When focus mode is active, only CRITICAL passes through."""
        n = self._make_ready()
        iv = FakeIntervention(priority=RealPriority.MEDIUM)

        from bantz.agent.interventions import intervention_queue as real_queue
        old_init = real_queue._initialized
        old_focus = real_queue._focus
        old_quiet = real_queue._quiet
        try:
            real_queue._initialized = True
            real_queue._focus = True
            real_queue._quiet = False
            with patch.object(n, "_is_tui_active", return_value=False), \
                 patch.object(n, "send") as mock_send:
                result = n.dispatch(iv)
            assert result is False
            mock_send.assert_not_called()
        finally:
            real_queue._initialized = old_init
            real_queue._focus = old_focus
            real_queue._quiet = old_quiet

    def test_dispatch_focus_mode_allows_critical(self):
        """CRITICAL notifications bypass focus mode."""
        n = self._make_ready()
        iv = FakeIntervention(priority=RealPriority.CRITICAL, title="Alert", reason="Critical!")

        from bantz.agent.interventions import intervention_queue as real_queue
        old_init = real_queue._initialized
        old_focus = real_queue._focus
        old_quiet = real_queue._quiet
        try:
            real_queue._initialized = True
            real_queue._focus = True
            real_queue._quiet = False
            with patch.object(n, "_is_tui_active", return_value=False), \
                 patch.object(n, "send", return_value=True) as mock_send:
                result = n.dispatch(iv)
            assert result is True
            mock_send.assert_called_once()
        finally:
            real_queue._initialized = old_init
            real_queue._focus = old_focus
            real_queue._quiet = old_quiet

    def test_dispatch_quiet_mode_blocks_medium(self):
        """Quiet mode only allows CRITICAL."""
        n = self._make_ready()
        iv = FakeIntervention(priority=RealPriority.HIGH)

        from bantz.agent.interventions import intervention_queue as real_queue
        old_init = real_queue._initialized
        old_focus = real_queue._focus
        old_quiet = real_queue._quiet
        try:
            real_queue._initialized = True
            real_queue._focus = False
            real_queue._quiet = True
            with patch.object(n, "_is_tui_active", return_value=False), \
                 patch.object(n, "send") as mock_send:
                result = n.dispatch(iv)
            assert result is False
            mock_send.assert_not_called()
        finally:
            real_queue._initialized = old_init
            real_queue._focus = old_focus
            real_queue._quiet = old_quiet

    def test_dispatch_empty_reason(self):
        n = self._make_ready()
        iv = FakeIntervention(title="Hello", reason="")
        with patch.object(n, "_is_tui_active", return_value=False), \
             patch.object(n, "send", return_value=True) as mock_send:
            n.dispatch(iv)
        mock_send.assert_called_once_with("Bantz: Hello", "", urgency="normal", expire_ms=5000)

    def test_skipped_count_increments(self):
        n = self._make_ready()
        iv = FakeIntervention()
        with patch.object(n, "_is_tui_active", return_value=True):
            n.dispatch(iv)
            n.dispatch(iv)
        assert n._skipped_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# Stats / Status
# ═══════════════════════════════════════════════════════════════════════════


class TestStats:
    def test_stats_initialized(self):
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init(icon="/icon.png")
        st = n.stats()
        assert st["initialized"] is True
        assert st["available"] is True
        assert st["enabled"] is True
        assert st["icon"] == "/icon.png"
        assert st["sent"] == 0
        assert st["skipped"] == 0

    def test_stats_not_initialized(self):
        n = Notifier()
        st = n.stats()
        assert st["initialized"] is False

    def test_status_line_active(self):
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init()
        line = n.status_line()
        assert "state=active" in line
        assert "sent=0" in line
        assert "skipped=0" in line

    def test_status_line_unavailable(self):
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value=None):
            n.init()
        line = n.status_line()
        assert "state=unavailable" in line

    def test_status_line_disabled(self):
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init(enabled=False)
        line = n.status_line()
        assert "state=disabled" in line

    def test_status_line_with_counts(self):
        n = Notifier()
        with patch("bantz.agent.notifier.shutil.which", return_value="/usr/bin/notify-send"):
            n.init()
        n._sent_count = 5
        n._skipped_count = 3
        line = n.status_line()
        assert "sent=5" in line
        assert "skipped=3" in line


# ═══════════════════════════════════════════════════════════════════════════
# Config fields
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigFields:
    def test_config_has_notification_fields(self):
        from bantz.config import config
        assert hasattr(config, "desktop_notifications")
        assert hasattr(config, "notification_icon")
        assert hasattr(config, "notification_sound")

    def test_config_defaults(self):
        from bantz.config import Config
        # Test the hardcoded Field defaults, independent of .env overrides
        fields = Config.model_fields
        assert fields["desktop_notifications"].default is True
        assert fields["notification_icon"].default == ""
        assert fields["notification_sound"].default is False


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════


class TestSingleton:
    def test_module_singleton_exists(self):
        assert isinstance(notifier, Notifier)

    def test_tui_identifiers(self):
        assert "bantz" in _TUI_IDENTIFIERS
        assert "python" in _TUI_IDENTIFIERS
        assert "python3" in _TUI_IDENTIFIERS
