"""
Bantz — Dependency Checker & Installer (#365)

Detects missing Hyprland desktop dependencies and provides install
commands for common package managers (pacman, apt, dnf, zypper).

Does NOT auto-install without user confirmation — presents a summary
and the appropriate install command for the detected distro.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

log = logging.getLogger("bantz.desktop.installer")

# ── Package mapping per distro ────────────────────────────────────────────────

# Maps binary name → package name per package manager
_PACKAGE_MAP: dict[str, dict[str, str]] = {
    "Hyprland": {
        "pacman": "hyprland",
        "apt": "hyprland",
        "dnf": "hyprland",
        "zypper": "hyprland",
    },
    "kitty": {
        "pacman": "kitty",
        "apt": "kitty",
        "dnf": "kitty",
        "zypper": "kitty",
    },
    "waybar": {
        "pacman": "waybar",
        "apt": "waybar",
        "dnf": "waybar",
        "zypper": "waybar",
    },
    "eww": {
        "pacman": "eww",
        "apt": "eww",
        "dnf": "eww",
        "zypper": "eww",
    },
    "swww-daemon": {
        "pacman": "swww",
        "apt": "swww",
        "dnf": "swww",
        "zypper": "swww",
    },
    "mako": {
        "pacman": "mako",
        "apt": "mako-notifier",
        "dnf": "mako",
        "zypper": "mako",
    },
    "grim": {
        "pacman": "grim",
        "apt": "grim",
        "dnf": "grim",
        "zypper": "grim",
    },
    "slurp": {
        "pacman": "slurp",
        "apt": "slurp",
        "dnf": "slurp",
        "zypper": "slurp",
    },
    "wl-copy": {
        "pacman": "wl-clipboard",
        "apt": "wl-clipboard",
        "dnf": "wl-clipboard",
        "zypper": "wl-clipboard",
    },
}

PackageManager = Literal["pacman", "apt", "dnf", "zypper", "unknown"]


@dataclass
class DependencyStatus:
    """Status of a single dependency."""
    name: str
    binary: str
    installed: bool
    required: bool
    package_hint: str = ""


@dataclass
class DependencyReport:
    """Full dependency check report."""
    package_manager: PackageManager
    dependencies: list[DependencyStatus] = field(default_factory=list)
    install_command: str = ""

    @property
    def all_required_met(self) -> bool:
        return all(d.installed for d in self.dependencies if d.required)

    @property
    def missing_required(self) -> list[DependencyStatus]:
        return [d for d in self.dependencies if d.required and not d.installed]

    @property
    def missing_optional(self) -> list[DependencyStatus]:
        return [d for d in self.dependencies if not d.required and not d.installed]

    @property
    def installed_count(self) -> int:
        return sum(1 for d in self.dependencies if d.installed)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["🖥️  Bantz Hyprland Desktop — Dependency Check", "─" * 50]
        for dep in self.dependencies:
            icon = "✅" if dep.installed else ("❌" if dep.required else "⚠️")
            tag = "required" if dep.required else "optional"
            lines.append(f"  {icon} {dep.name:20s} [{tag}]")
        lines.append("─" * 50)
        lines.append(
            f"  {self.installed_count}/{len(self.dependencies)} installed"
        )
        if self.missing_required:
            lines.append(f"\n❌ Missing required: {', '.join(d.name for d in self.missing_required)}")
        if self.missing_optional:
            lines.append(f"⚠️  Missing optional: {', '.join(d.name for d in self.missing_optional)}")
        if self.install_command:
            lines.append(f"\n📦 Install command:\n   {self.install_command}")
        elif self.all_required_met and not self.missing_optional:
            lines.append("\n✅ All dependencies satisfied!")
        return "\n".join(lines)


class DependencyChecker:
    """Check and report on Hyprland desktop dependencies."""

    REQUIRED = [
        ("Hyprland", "Hyprland"),
        ("Kitty Terminal", "kitty"),
        ("Waybar", "waybar"),
    ]

    OPTIONAL = [
        ("Eww Widgets", "eww"),
        ("SWWW Wallpaper", "swww-daemon"),
        ("Mako Notifications", "mako"),
        ("Grim Screenshot", "grim"),
        ("Slurp Selection", "slurp"),
        ("Clipboard (wl-copy)", "wl-copy"),
    ]

    def detect_package_manager(self) -> PackageManager:
        """Detect the system's package manager."""
        for pm in ("pacman", "apt", "dnf", "zypper"):
            if shutil.which(pm):
                return pm  # type: ignore[return-value]
        return "unknown"

    def check(self) -> DependencyReport:
        """Run a full dependency check."""
        pm = self.detect_package_manager()
        deps: list[DependencyStatus] = []

        for name, binary in self.REQUIRED:
            pkg = _PACKAGE_MAP.get(binary, {}).get(pm, binary)
            deps.append(DependencyStatus(
                name=name,
                binary=binary,
                installed=shutil.which(binary) is not None,
                required=True,
                package_hint=pkg,
            ))

        for name, binary in self.OPTIONAL:
            pkg = _PACKAGE_MAP.get(binary, {}).get(pm, binary)
            deps.append(DependencyStatus(
                name=name,
                binary=binary,
                installed=shutil.which(binary) is not None,
                required=False,
                package_hint=pkg,
            ))

        # Build install command for missing packages
        missing_pkgs = [d.package_hint for d in deps if not d.installed]
        install_cmd = ""
        if missing_pkgs:
            prefix = {
                "pacman": "sudo pacman -S --noconfirm",
                "apt": "sudo apt install -y",
                "dnf": "sudo dnf install -y",
                "zypper": "sudo zypper install -y",
            }.get(pm, "# Install manually:")
            install_cmd = f"{prefix} {' '.join(missing_pkgs)}"

        report = DependencyReport(
            package_manager=pm,
            dependencies=deps,
            install_command=install_cmd,
        )
        return report

    def check_wayland(self) -> bool:
        """Check if we're running under Wayland."""
        return (
            os.environ.get("XDG_SESSION_TYPE") == "wayland"
            or os.environ.get("WAYLAND_DISPLAY") is not None
        )

    def check_gpu(self) -> dict[str, str]:
        """Check GPU availability for Hyprland."""
        result: dict[str, str] = {"driver": "unknown", "renderer": "unknown"}
        # Try lspci
        if shutil.which("lspci"):
            try:
                out = subprocess.check_output(
                    ["lspci", "-v"],
                    timeout=5,
                ).decode()
                for line in out.splitlines():
                    low = line.lower()
                    if "vga" in low or "3d" in low:
                        result["renderer"] = line.strip().split(": ", 1)[-1] if ": " in line else line.strip()
                        break
            except (subprocess.SubprocessError, OSError):
                pass
        return result
