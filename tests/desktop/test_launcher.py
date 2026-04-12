"""Tests for bantz.desktop.launcher — Desktop session launcher (#365)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bantz.desktop.launcher import DesktopLauncher


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def launcher(tmp_path: Path) -> DesktopLauncher:
    return DesktopLauncher(
        config_dir=tmp_path,
        wallpaper="",
        left_ratio=0.6,
    )


# ── Dependency check ──────────────────────────────────────────────────────────

class TestDependencyCheck:
    def test_checks_all_binaries(self, launcher: DesktopLauncher) -> None:
        result = launcher.check_dependencies()
        assert "Hyprland" in result
        assert "kitty" in result
        assert "waybar" in result
        assert "eww" in result
        assert "mako" in result

    def test_missing_required(self, launcher: DesktopLauncher) -> None:
        with patch("shutil.which", return_value=None):
            missing = launcher.missing_required()
            assert "Hyprland" in missing
            assert "kitty" in missing
            assert "waybar" in missing

    def test_no_missing_when_all_present(self, launcher: DesktopLauncher) -> None:
        with patch("shutil.which", return_value="/usr/bin/something"):
            assert launcher.missing_required() == []
            assert launcher.missing_optional() == []


# ── Config generation ─────────────────────────────────────────────────────────

class TestConfigGeneration:
    def test_ensure_configs_creates_files(self, launcher: DesktopLauncher) -> None:
        written = launcher.ensure_configs()
        assert len(written) == 7
        for p in written:
            assert p.exists()

    def test_ensure_configs_dry_run(self, launcher: DesktopLauncher) -> None:
        written = launcher.ensure_configs(dry_run=True)
        assert len(written) == 7
        assert not any(p.exists() for p in written)


# ── Process management ────────────────────────────────────────────────────────

class TestProcessManagement:
    def test_spawn_success(self, launcher: DesktopLauncher) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        with patch("subprocess.Popen", return_value=mock_proc):
            ok = launcher._spawn("test", ["echo", "hi"])
            assert ok is True
            assert "test" in launcher._processes

    def test_spawn_failure(self, launcher: DesktopLauncher) -> None:
        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            ok = launcher._spawn("test", ["nonexistent"])
            assert ok is False

    def test_is_alive(self, launcher: DesktopLauncher) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        launcher._processes["test"] = mock_proc
        assert launcher._is_alive("test") is True

    def test_is_not_alive(self, launcher: DesktopLauncher) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # exited
        launcher._processes["test"] = mock_proc
        assert launcher._is_alive("test") is False

    def test_is_alive_unknown(self, launcher: DesktopLauncher) -> None:
        assert launcher._is_alive("nonexistent") is False


# ── Launch components ─────────────────────────────────────────────────────────

class TestLaunchComponents:
    def test_launches_waybar(self, launcher: DesktopLauncher) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_proc.poll.return_value = None
        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("shutil.which", return_value="/usr/bin/something"):
                # Need to patch time.sleep to avoid delay
                with patch("time.sleep"):
                    results = launcher.launch_components()
                    assert "waybar" in results

    def test_skips_missing_optionals(self, launcher: DesktopLauncher) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 100
        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("shutil.which", return_value=None):
                results = launcher.launch_components()
                assert "swww" not in results
                assert "mako" not in results
                assert "eww-daemon" not in results


# ── Shutdown ──────────────────────────────────────────────────────────────────

class TestShutdown:
    def test_terminates_processes(self, launcher: DesktopLauncher) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # alive
        mock_proc.wait.return_value = None
        launcher._processes["test"] = mock_proc
        with patch("shutil.which", return_value=None):  # no eww
            launcher.shutdown()
        mock_proc.terminate.assert_called_once()
        assert len(launcher._processes) == 0

    def test_force_kills_on_timeout(self, launcher: DesktopLauncher) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("test", 3)
        launcher._processes["test"] = mock_proc
        with patch("shutil.which", return_value=None):
            launcher.shutdown()
        mock_proc.kill.assert_called_once()

    def test_stops_monitoring_flag(self, launcher: DesktopLauncher) -> None:
        launcher._running = True
        with patch("shutil.which", return_value=None):
            launcher.shutdown()
        assert launcher._running is False


# ── Status ────────────────────────────────────────────────────────────────────

class TestStatus:
    def test_running_status(self, launcher: DesktopLauncher) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 42
        launcher._processes["waybar"] = mock_proc
        status = launcher.status()
        assert "running" in status["waybar"]
        assert "42" in status["waybar"]

    def test_stopped_status(self, launcher: DesktopLauncher) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        launcher._processes["waybar"] = mock_proc
        status = launcher.status()
        assert "stopped" in status["waybar"]

    def test_empty_status(self, launcher: DesktopLauncher) -> None:
        assert launcher.status() == {}


# ── Launch Hyprland ───────────────────────────────────────────────────────────

class TestLaunchHyprland:
    def test_fails_with_missing_deps(self, launcher: DesktopLauncher) -> None:
        with patch("shutil.which", return_value=None):
            code = launcher.launch_hyprland()
            assert code == 1

    def test_exits_with_hyprland_code(self, launcher: DesktopLauncher) -> None:
        with patch("shutil.which", return_value="/usr/bin/something"):
            mock_result = MagicMock()
            mock_result.returncode = 0
            with patch("subprocess.run", return_value=mock_result):
                code = launcher.launch_hyprland()
                assert code == 0

    def test_hyprland_not_found(self, launcher: DesktopLauncher) -> None:
        with patch("shutil.which", return_value="/usr/bin/something"):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                code = launcher.launch_hyprland()
                assert code == 127
