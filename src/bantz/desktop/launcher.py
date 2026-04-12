"""
Bantz — Desktop Launcher (#365)

Orchestrates the Bantz Hyprland desktop session:

1. Checks dependencies (Hyprland, kitty, waybar, eww, swww, mako)
2. Generates / validates config files
3. Launches Hyprland with the generated config
4. Manages auto-restart of crashed components
5. Graceful shutdown (stop eww, waybar, mako → exit Hyprland)
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from bantz.desktop.generator import ConfigGenerator
from bantz.desktop.installer import DependencyChecker

log = logging.getLogger("bantz.desktop.launcher")


class DesktopLauncher:
    """Launch and manage the Bantz Hyprland desktop session."""

    REQUIRED_BINARIES = ["Hyprland", "kitty", "waybar"]
    OPTIONAL_BINARIES = ["eww", "swww-daemon", "mako", "grim", "slurp"]

    def __init__(
        self,
        *,
        config_dir: str | Path | None = None,
        wallpaper: str = "",
        left_ratio: float = 0.6,
        monitor_config: str = ",preferred,auto,1",
    ) -> None:
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".config"
        self.wallpaper = wallpaper
        self.left_ratio = left_ratio
        self.monitor_config = monitor_config
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._running = False

    # ── Dependency check ──────────────────────────────────────────────────

    def check_dependencies(self) -> dict[str, bool]:
        """Check which binaries are available."""
        result: dict[str, bool] = {}
        for binary in self.REQUIRED_BINARIES + self.OPTIONAL_BINARIES:
            result[binary] = shutil.which(binary) is not None
        return result

    def missing_required(self) -> list[str]:
        """Return list of missing required binaries."""
        deps = self.check_dependencies()
        return [b for b in self.REQUIRED_BINARIES if not deps.get(b)]

    def missing_optional(self) -> list[str]:
        """Return list of missing optional binaries."""
        deps = self.check_dependencies()
        return [b for b in self.OPTIONAL_BINARIES if not deps.get(b)]

    # ── Config generation ─────────────────────────────────────────────────

    def ensure_configs(self, *, dry_run: bool = False) -> list[Path]:
        """Generate Bantz desktop configs if not present."""
        gen = ConfigGenerator(
            wallpaper=self.wallpaper,
            left_ratio=self.left_ratio,
            monitor_config=self.monitor_config,
        )
        return gen.deploy(self.config_dir, dry_run=dry_run)

    # ── Launch components ─────────────────────────────────────────────────

    def _spawn(self, name: str, cmd: list[str], **kwargs: Any) -> bool:
        """Spawn a background process and track it."""
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs,
            )
            self._processes[name] = proc
            log.info("Started %s (PID %d): %s", name, proc.pid, " ".join(cmd))
            return True
        except (FileNotFoundError, OSError) as exc:
            log.warning("Failed to start %s: %s", name, exc)
            return False

    def _is_alive(self, name: str) -> bool:
        proc = self._processes.get(name)
        return proc is not None and proc.poll() is None

    def _restart_if_dead(self, name: str, cmd: list[str]) -> None:
        """Auto-restart a component if it has crashed."""
        if name in self._processes and not self._is_alive(name):
            log.warning("%s died — restarting", name)
            self._spawn(name, cmd)

    # ── Session lifecycle ─────────────────────────────────────────────────

    def launch_components(self) -> dict[str, bool]:
        """Launch all desktop components (call before or inside Hyprland)."""
        results: dict[str, bool] = {}

        # swww (wallpaper daemon)
        if shutil.which("swww-daemon"):
            results["swww"] = self._spawn("swww", ["swww-daemon"])
            if self.wallpaper and results["swww"]:
                subprocess.Popen(
                    ["swww", "img", self.wallpaper,
                     "--transition-type", "grow",
                     "--transition-pos", "0.5,0.5"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        # waybar (top bar)
        waybar_conf = self.config_dir / "waybar" / "config"
        waybar_css = self.config_dir / "waybar" / "style.css"
        waybar_cmd = ["waybar"]
        if waybar_conf.exists():
            waybar_cmd += ["-c", str(waybar_conf)]
        if waybar_css.exists():
            waybar_cmd += ["-s", str(waybar_css)]
        results["waybar"] = self._spawn("waybar", waybar_cmd)

        # mako (notifications)
        if shutil.which("mako"):
            mako_conf = self.config_dir / "mako" / "config"
            mako_cmd = ["mako"]
            if mako_conf.exists():
                mako_cmd += ["-c", str(mako_conf)]
            results["mako"] = self._spawn("mako", mako_cmd)

        # eww (widgets)
        if shutil.which("eww"):
            results["eww-daemon"] = self._spawn("eww-daemon", ["eww", "daemon"])
            # Give eww daemon time to start before opening windows
            import time
            time.sleep(1)
            results["eww-widgets"] = self._spawn(
                "eww-widgets",
                ["eww", "open-many", "bantz-news", "bantz-calendar", "bantz-stats"],
            )

        return results

    def launch_hyprland(self) -> int:
        """Launch Hyprland as the main compositor (blocking).

        This replaces the current process with Hyprland. Returns
        the exit code (only if Hyprland exits on its own).
        """
        missing = self.missing_required()
        if missing:
            log.error("Missing required dependencies: %s", ", ".join(missing))
            print(f"❌ Missing required: {', '.join(missing)}")
            print("   Run: bantz --setup hyprland")
            return 1

        # Ensure configs exist
        self.ensure_configs()

        hypr_conf = self.config_dir / "hypr" / "hyprland.conf"
        env = os.environ.copy()
        if hypr_conf.exists():
            env["HYPRLAND_CONFIG"] = str(hypr_conf)

        log.info("Launching Hyprland with config: %s", hypr_conf)
        try:
            result = subprocess.run(
                ["Hyprland"],
                env=env,
            )
            return result.returncode
        except FileNotFoundError:
            log.error("Hyprland not found in PATH")
            return 127

    # ── Health monitor (async) ────────────────────────────────────────────

    async def monitor_components(self, check_interval: float = 10.0) -> None:
        """Periodically check and restart crashed components.

        Call this from an async context running inside Hyprland.
        """
        self._running = True
        restart_cmds: dict[str, list[str]] = {
            "waybar": ["waybar"],
            "mako": ["mako"],
        }
        while self._running:
            for name, cmd in restart_cmds.items():
                self._restart_if_dead(name, cmd)
            await asyncio.sleep(check_interval)

    def stop_monitoring(self) -> None:
        """Stop the health monitor loop."""
        self._running = False

    # ── Shutdown ──────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Gracefully stop all managed components."""
        log.info("Shutting down Bantz desktop components")
        self._running = False
        # Close eww windows first
        if shutil.which("eww"):
            subprocess.run(
                ["eww", "close-all"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        # Terminate all managed processes
        for name, proc in self._processes.items():
            if proc.poll() is None:
                log.info("Stopping %s (PID %d)", name, proc.pid)
                proc.terminate()
        # Wait briefly then force-kill stragglers
        for name, proc in self._processes.items():
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                log.warning("Force-killing %s", name)
                proc.kill()
        self._processes.clear()

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict[str, str]:
        """Return status of each managed component."""
        result: dict[str, str] = {}
        for name, proc in self._processes.items():
            if proc.poll() is None:
                result[name] = f"running (PID {proc.pid})"
            else:
                result[name] = f"stopped (exit {proc.returncode})"
        return result
