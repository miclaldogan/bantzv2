"""Tests for bantz.desktop.installer — Dependency checker (#365)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from bantz.desktop.installer import (
    DependencyChecker,
    DependencyReport,
    DependencyStatus,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def checker() -> DependencyChecker:
    return DependencyChecker()


# ── Package manager detection ─────────────────────────────────────────────────

class TestPackageManagerDetection:
    def test_detects_pacman(self, checker: DependencyChecker) -> None:
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/pacman" if x == "pacman" else None):
            assert checker.detect_package_manager() == "pacman"

    def test_detects_apt(self, checker: DependencyChecker) -> None:
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/apt" if x == "apt" else None):
            assert checker.detect_package_manager() == "apt"

    def test_detects_dnf(self, checker: DependencyChecker) -> None:
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/dnf" if x == "dnf" else None):
            assert checker.detect_package_manager() == "dnf"

    def test_unknown_when_none(self, checker: DependencyChecker) -> None:
        with patch("shutil.which", return_value=None):
            assert checker.detect_package_manager() == "unknown"


# ── Dependency check ──────────────────────────────────────────────────────────

class TestDependencyCheck:
    def test_all_installed(self, checker: DependencyChecker) -> None:
        with patch("shutil.which", return_value="/usr/bin/something"):
            report = checker.check()
            assert report.all_required_met
            assert len(report.missing_required) == 0

    def test_missing_hyprland(self, checker: DependencyChecker) -> None:
        def mock_which(binary: str) -> str | None:
            if binary == "Hyprland":
                return None
            return f"/usr/bin/{binary}"
        with patch("shutil.which", side_effect=mock_which):
            report = checker.check()
            assert not report.all_required_met
            missing = [d.name for d in report.missing_required]
            assert "Hyprland" in missing

    def test_missing_optional(self, checker: DependencyChecker) -> None:
        def mock_which(binary: str) -> str | None:
            # Required are installed, optionals are not
            required_bins = {"Hyprland", "kitty", "waybar", "pacman"}
            return f"/usr/bin/{binary}" if binary in required_bins else None
        with patch("shutil.which", side_effect=mock_which):
            report = checker.check()
            assert report.all_required_met
            assert len(report.missing_optional) > 0

    def test_install_command_generated(self, checker: DependencyChecker) -> None:
        def mock_which(binary: str) -> str | None:
            if binary == "pacman":
                return "/usr/bin/pacman"
            if binary in ("Hyprland", "kitty", "waybar"):
                return f"/usr/bin/{binary}"
            return None
        with patch("shutil.which", side_effect=mock_which):
            report = checker.check()
            assert "pacman" in report.install_command
            assert report.install_command.startswith("sudo pacman")

    def test_apt_install_command(self, checker: DependencyChecker) -> None:
        def mock_which(binary: str) -> str | None:
            if binary == "apt":
                return "/usr/bin/apt"
            if binary in ("Hyprland", "kitty", "waybar"):
                return f"/usr/bin/{binary}"
            return None
        with patch("shutil.which", side_effect=mock_which):
            report = checker.check()
            assert "apt install" in report.install_command


# ── DependencyReport ──────────────────────────────────────────────────────────

class TestDependencyReport:
    def test_all_required_met_property(self) -> None:
        report = DependencyReport(
            package_manager="pacman",
            dependencies=[
                DependencyStatus("A", "a", True, True),
                DependencyStatus("B", "b", True, True),
            ],
        )
        assert report.all_required_met

    def test_not_all_required_met(self) -> None:
        report = DependencyReport(
            package_manager="pacman",
            dependencies=[
                DependencyStatus("A", "a", True, True),
                DependencyStatus("B", "b", False, True),
            ],
        )
        assert not report.all_required_met

    def test_missing_required_list(self) -> None:
        report = DependencyReport(
            package_manager="pacman",
            dependencies=[
                DependencyStatus("A", "a", True, True),
                DependencyStatus("B", "b", False, True),
                DependencyStatus("C", "c", False, False),
            ],
        )
        assert len(report.missing_required) == 1
        assert report.missing_required[0].name == "B"

    def test_missing_optional_list(self) -> None:
        report = DependencyReport(
            package_manager="pacman",
            dependencies=[
                DependencyStatus("A", "a", True, True),
                DependencyStatus("C", "c", False, False),
            ],
        )
        assert len(report.missing_optional) == 1
        assert report.missing_optional[0].name == "C"

    def test_installed_count(self) -> None:
        report = DependencyReport(
            package_manager="pacman",
            dependencies=[
                DependencyStatus("A", "a", True, True),
                DependencyStatus("B", "b", False, True),
                DependencyStatus("C", "c", True, False),
            ],
        )
        assert report.installed_count == 2

    def test_summary_output(self) -> None:
        report = DependencyReport(
            package_manager="pacman",
            dependencies=[
                DependencyStatus("Hyprland", "Hyprland", True, True),
                DependencyStatus("Eww", "eww", False, False),
            ],
            install_command="sudo pacman -S eww",
        )
        summary = report.summary()
        assert "Bantz Hyprland Desktop" in summary
        assert "✅" in summary
        assert "⚠️" in summary
        assert "sudo pacman" in summary

    def test_summary_all_satisfied(self) -> None:
        report = DependencyReport(
            package_manager="pacman",
            dependencies=[
                DependencyStatus("A", "a", True, True),
                DependencyStatus("B", "b", True, False),
            ],
        )
        summary = report.summary()
        assert "All dependencies satisfied" in summary


# ── Wayland check ─────────────────────────────────────────────────────────────

class TestWaylandCheck:
    def test_wayland_session(self, checker: DependencyChecker) -> None:
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}):
            assert checker.check_wayland() is True

    def test_wayland_display(self, checker: DependencyChecker) -> None:
        with patch.dict("os.environ", {"WAYLAND_DISPLAY": "wayland-0"}, clear=False):
            assert checker.check_wayland() is True

    def test_x11_session(self, checker: DependencyChecker) -> None:
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=True):
            assert checker.check_wayland() is False


# ── GPU check ─────────────────────────────────────────────────────────────────

class TestGPUCheck:
    def test_returns_dict(self, checker: DependencyChecker) -> None:
        result = checker.check_gpu()
        assert "driver" in result
        assert "renderer" in result

    def test_no_lspci(self, checker: DependencyChecker) -> None:
        with patch("shutil.which", return_value=None):
            result = checker.check_gpu()
            assert result["renderer"] == "unknown"
