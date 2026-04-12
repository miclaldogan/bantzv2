"""
Bantz v2 — Hyprland Desktop Environment (#365)

Provides a fully configured Hyprland tiling desktop where Bantz is the
OS-level AI assistant:

- **generator**: Produce hyprland.conf, kitty.conf, waybar, eww, mako configs
- **hyprctl**: Async IPC wrapper around ``hyprctl`` JSON mode
- **widgets**: JSON data providers for eww widget panels
- **launcher**: Session orchestrator (start/stop all components)
- **installer**: Dependency checker + pacman/apt install helper
"""
from __future__ import annotations

from bantz.desktop.hyprctl import HyprctlClient
from bantz.desktop.generator import ConfigGenerator
from bantz.desktop.widgets import WidgetDataProvider
from bantz.desktop.launcher import DesktopLauncher
from bantz.desktop.installer import DependencyChecker

__all__ = [
    "HyprctlClient",
    "ConfigGenerator",
    "WidgetDataProvider",
    "DesktopLauncher",
    "DependencyChecker",
]
