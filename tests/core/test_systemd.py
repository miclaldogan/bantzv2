"""
Tests for Issue #173 — systemd integration hardening.

Covers:
  - _check_linger() via loginctl show-user API
  - _ensure_linger() interactive flow with polkit safety
  - _systemctl() helper replacing os.system()
  - _verify_service() post-install check
  - _systemd_check() full diagnostic
  - _format_uptime() timestamp parsing
  - _setup_systemd() service file writing + full flow
  - No os.system() calls remain in systemd code
"""
from __future__ import annotations

import io
import os
import subprocess
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest


# ── _check_linger ────────────────────────────────────────────────────────

class TestCheckLinger:
    """Uses loginctl show-user (official API) not hardcoded /var/lib path."""

    def test_linger_enabled(self):
        from bantz.__main__ import _check_linger
        mock_result = MagicMock(returncode=0, stdout="Linger=yes\n")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            assert _check_linger("testuser") is True
            mock_run.assert_called_once_with(
                ["loginctl", "show-user", "testuser", "--property=Linger"],
                capture_output=True, text=True, timeout=5,
            )

    def test_linger_disabled(self):
        from bantz.__main__ import _check_linger
        mock_result = MagicMock(returncode=0, stdout="Linger=no\n")
        with patch("subprocess.run", return_value=mock_result):
            assert _check_linger("testuser") is False

    def test_loginctl_not_found(self):
        from bantz.__main__ import _check_linger
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _check_linger("testuser") is False

    def test_loginctl_timeout(self):
        from bantz.__main__ import _check_linger
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="loginctl", timeout=5)):
            assert _check_linger("testuser") is False

    def test_loginctl_nonzero_exit(self):
        from bantz.__main__ import _check_linger
        mock_result = MagicMock(returncode=1, stdout="")
        with patch("subprocess.run", return_value=mock_result):
            assert _check_linger("testuser") is False

    def test_no_hardcoded_var_lib_path(self):
        """Ensure we never check /var/lib/systemd/linger directly."""
        import inspect
        from bantz.__main__ import _check_linger
        source = inspect.getsource(_check_linger)
        assert "/var/lib/systemd/linger" not in source


# ── _ensure_linger ───────────────────────────────────────────────────────

class TestEnsureLinger:
    def test_already_enabled_no_prompt(self):
        from bantz.__main__ import _ensure_linger
        with patch("bantz.__main__._check_linger", return_value=True):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _ensure_linger("testuser")
            assert result is True
            assert "already enabled" in buf.getvalue()

    def test_enable_success(self):
        from bantz.__main__ import _ensure_linger
        mock_result = MagicMock(returncode=0)
        with patch("bantz.__main__._check_linger", side_effect=[False, True]), \
             patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("builtins.input", return_value="y"):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _ensure_linger("testuser")
            assert result is True
            # Verify capture_output=False for polkit safety
            mock_run.assert_called_once_with(
                ["loginctl", "enable-linger", "testuser"],
                capture_output=False, text=True,
            )

    def test_enable_failure_shows_sudo_hint(self):
        from bantz.__main__ import _ensure_linger
        mock_result = MagicMock(returncode=1)
        with patch("bantz.__main__._check_linger", return_value=False), \
             patch("subprocess.run", return_value=mock_result), \
             patch("builtins.input", return_value="y"):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _ensure_linger("testuser")
            assert result is False
            assert "sudo" in buf.getvalue()

    def test_user_declines(self):
        from bantz.__main__ import _ensure_linger
        with patch("bantz.__main__._check_linger", return_value=False), \
             patch("builtins.input", return_value="n"):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _ensure_linger("testuser")
            assert result is False
            assert "Skipped" in buf.getvalue()

    def test_capture_output_false_for_polkit(self):
        """Critical: loginctl enable-linger must NOT use capture_output=True
        to avoid freezing when polkit asks for a password."""
        import inspect
        from bantz.__main__ import _ensure_linger
        source = inspect.getsource(_ensure_linger)
        assert "capture_output=False" in source


# ── _systemctl ───────────────────────────────────────────────────────────

class TestSystemctl:
    def test_success(self):
        from bantz.__main__ import _systemctl
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            assert _systemctl("daemon-reload") is True
            mock_run.assert_called_once_with(
                ["systemctl", "--user", "daemon-reload"],
                capture_output=True, text=True,
            )

    def test_failure_prints_stderr(self):
        from bantz.__main__ import _systemctl
        mock_result = MagicMock(returncode=1, stderr="Failed to reload daemon")
        with patch("subprocess.run", return_value=mock_result):
            buf = io.StringIO()
            with redirect_stdout(buf):
                assert _systemctl("daemon-reload") is False
            assert "Failed to reload daemon" in buf.getvalue()

    def test_multiple_args(self):
        from bantz.__main__ import _systemctl
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            assert _systemctl("enable", "bantz.service") is True
            mock_run.assert_called_once_with(
                ["systemctl", "--user", "enable", "bantz.service"],
                capture_output=True, text=True,
            )

    def test_no_os_system_in_systemd_code(self):
        """Verify os.system() is not used in any systemd-related function."""
        import inspect
        from bantz import __main__ as mod
        for name in ("_setup_systemd", "_systemctl", "_verify_service",
                      "_systemd_check", "_ensure_linger"):
            fn = getattr(mod, name)
            source = inspect.getsource(fn)
            assert "os.system(" not in source, f"{name} still uses os.system()"


# ── _verify_service ──────────────────────────────────────────────────────

class TestVerifyService:
    def test_active(self):
        from bantz.__main__ import _verify_service
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="active\n"),  # is-active
                MagicMock(returncode=0),  # status
            ]
            buf = io.StringIO()
            with redirect_stdout(buf):
                _verify_service()
            assert "running" in buf.getvalue()

    def test_failed(self):
        from bantz.__main__ import _verify_service
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=3, stdout="failed\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                _verify_service()
            assert "failed" in buf.getvalue()
            assert "journalctl" in buf.getvalue()


# ── _format_uptime ───────────────────────────────────────────────────────

class TestFormatUptime:
    def test_parse_with_weekday(self):
        from bantz.__main__ import _format_uptime
        from datetime import datetime, timedelta
        now = datetime.now()
        past = now - timedelta(days=3, hours=6, minutes=15)
        ts = past.strftime("%a %Y-%m-%d %H:%M:%S") + " TRT"
        result = _format_uptime(ts)
        assert "3d" in result
        assert "6h" in result

    def test_parse_without_weekday(self):
        from bantz.__main__ import _format_uptime
        from datetime import datetime, timedelta
        now = datetime.now()
        past = now - timedelta(hours=2, minutes=30)
        ts = past.strftime("%Y-%m-%d %H:%M:%S")
        result = _format_uptime(ts)
        assert "2h" in result
        assert "30m" in result

    def test_unparseable_returns_raw(self):
        from bantz.__main__ import _format_uptime
        raw = "some-weird-format"
        assert _format_uptime(raw) == raw

    def test_zero_days_omitted(self):
        from bantz.__main__ import _format_uptime
        from datetime import datetime, timedelta
        past = datetime.now() - timedelta(hours=1, minutes=5)
        ts = past.strftime("%a %Y-%m-%d %H:%M:%S") + " UTC"
        result = _format_uptime(ts)
        assert "d" not in result
        assert "1h" in result


# ── _systemd_check ───────────────────────────────────────────────────────

class TestSystemdCheck:
    def test_no_service_file(self, tmp_path):
        from bantz.__main__ import _systemd_check
        with patch("pathlib.Path.home", return_value=tmp_path):
            buf = io.StringIO()
            with redirect_stdout(buf):
                _systemd_check()
            output = buf.getvalue()
            assert "NOT FOUND" in output
            assert "bantz --setup systemd" in output

    def test_full_check_active(self, tmp_path):
        from bantz.__main__ import _systemd_check
        # Create fake service file
        svc_dir = tmp_path / ".config" / "systemd" / "user"
        svc_dir.mkdir(parents=True)
        (svc_dir / "bantz.service").write_text("[Unit]\n")

        from datetime import datetime, timedelta
        past = datetime.now() - timedelta(days=1, hours=5)
        ts = past.strftime("%a %Y-%m-%d %H:%M:%S") + " TRT"

        side_effects = [
            MagicMock(returncode=0, stdout="active\n"),           # is-active
            MagicMock(returncode=0, stdout=(                      # show properties
                f"MainPID=12345\n"
                f"MemoryCurrent=47448064\n"
                f"ActiveEnterTimestamp={ts}\n"
            )),
            MagicMock(returncode=0, stdout=""),                   # journalctl
            MagicMock(returncode=0, stdout="enabled\n"),          # is-enabled
        ]

        with patch("pathlib.Path.home", return_value=tmp_path), \
             patch("bantz.__main__._check_linger", return_value=True), \
             patch("subprocess.run", side_effect=side_effects), \
             patch.dict(os.environ, {"USER": "testuser"}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                _systemd_check()
            output = buf.getvalue()
            assert "active (running)" in output
            assert "PID: 12345" in output
            assert "45.2M" in output
            assert "1d" in output
            assert "Linger: enabled" in output
            assert "Boot: enabled" in output


# ── _setup_systemd ───────────────────────────────────────────────────────

class TestSetupSystemd:
    def test_writes_service_file(self, tmp_path):
        from bantz.__main__ import _setup_systemd
        systemd_dir = tmp_path / ".config" / "systemd" / "user"

        with patch("pathlib.Path.home", return_value=tmp_path), \
             patch.dict(os.environ, {"USER": "testuser"}), \
             patch("bantz.__main__._ensure_linger", return_value=True), \
             patch("builtins.input", return_value="n"):
            buf = io.StringIO()
            with redirect_stdout(buf):
                _setup_systemd()

        svc = systemd_dir / "bantz.service"
        assert svc.exists()
        content = svc.read_text()
        assert "ExecStart=" in content
        assert "--daemon" in content
        assert "WantedBy=default.target" in content

    def test_enable_and_start_flow(self, tmp_path):
        from bantz.__main__ import _setup_systemd
        systemd_dir = tmp_path / ".config" / "systemd" / "user"

        with patch("pathlib.Path.home", return_value=tmp_path), \
             patch.dict(os.environ, {"USER": "testuser"}), \
             patch("bantz.__main__._ensure_linger", return_value=True), \
             patch("bantz.__main__._systemctl", return_value=True) as mock_sctl, \
             patch("bantz.__main__._verify_service") as mock_verify, \
             patch("builtins.input", return_value="y"):
            buf = io.StringIO()
            with redirect_stdout(buf):
                _setup_systemd()

        assert mock_sctl.call_count == 3
        mock_sctl.assert_any_call("daemon-reload")
        mock_sctl.assert_any_call("enable", "bantz.service")
        mock_sctl.assert_any_call("start", "bantz.service")
        mock_verify.assert_called_once()

    def test_no_user_env(self):
        from bantz.__main__ import _setup_systemd
        with patch.dict(os.environ, {}, clear=True):
            buf = io.StringIO()
            with redirect_stdout(buf):
                _setup_systemd()
            assert "Cannot determine" in buf.getvalue()


# ── argparse routing ─────────────────────────────────────────────────────

class TestArgparseRouting:
    def test_setup_systemd_routes_to_setup(self):
        from bantz.__main__ import _handle_setup
        with patch("bantz.__main__._setup_systemd") as mock:
            _handle_setup(["systemd"])
            mock.assert_called_once()

    def test_setup_systemd_check_routes_to_check(self):
        from bantz.__main__ import _handle_setup
        with patch("bantz.__main__._systemd_check") as mock:
            _handle_setup(["systemd", "--check"])
            mock.assert_called_once()

    def test_check_does_not_leak_globally(self):
        """--check is scoped to systemd setup, not a global arg."""
        from bantz.__main__ import main
        import sys
        # --check alone should NOT be a recognized argument
        with patch.object(sys, "argv", ["bantz", "--check"]):
            with pytest.raises(SystemExit):
                main()
