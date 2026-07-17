"""
Bantz — LanternTool: desktop ambiance control for Misa's Hyprland setup.

Bridges Bantz to the "Lantern" desktop customizations so the butler can run
the room: focus mode, the music edge-glow, album-art theming, desktop
sounds, and a composite night mode. Everything is driven through the same
scripts/state files the keybindings use, so Bantz and the hotkeys never
fight over state.

Actions
-------
  action=focus        state=on|off|toggle     — DND + dim + glow off
  action=glow         state=on|off|toggle     — music edge-glow
  action=glow_mode    mode=waves|pulse|hybrid — glow visual mode
  action=album_theme  state=on|off|toggle     — album-art→desktop colors
  action=sounds       state=on|off            — workspace/notification sounds
  action=night        state=on|off            — composite evening preset
  action=status                                — one-shot ambiance report

The daemon runs under a systemd user unit, where HYPRLAND_INSTANCE_SIGNATURE
is absent; `_hypr_env()` rediscovers it from $XDG_RUNTIME_DIR/hypr so
`hyprctl`/scripts keep working.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tools.lantern")

HOME = Path.home()
BIN = HOME / ".local" / "bin"
STATE = HOME / ".local" / "state"

FOCUS_FLAG = STATE / "focus-mode"
SOUNDS_OFF_FLAG = STATE / "desktop-sounds-off"
GLOW_MODE_FILE = STATE / "musicglow-mode"
BATTERY_CSV = STATE / "battery-health.csv"
GPU_STATS = STATE / "gpu-stats.txt"

_GLOW_MODES = {"waves": "0", "dalga": "0", "pulse": "1", "nabız": "1",
               "nabiz": "1", "hybrid": "2", "hibrit": "2"}
_GLOW_MODE_NAMES = {"0": "waves", "1": "pulse", "2": "hybrid"}


def _hypr_env() -> dict[str, str]:
    """Env with HYPRLAND_INSTANCE_SIGNATURE discovered (systemd unit lacks it)."""
    env = dict(os.environ)
    if "HYPRLAND_INSTANCE_SIGNATURE" not in env:
        runtime = Path(env.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        hypr = runtime / "hypr"
        if hypr.is_dir():
            sigs = sorted(hypr.iterdir(), key=lambda p: p.stat().st_mtime)
            if sigs:
                env["HYPRLAND_INSTANCE_SIGNATURE"] = sigs[-1].name
    return env


def _run(cmd: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=_hypr_env(), capture_output=True,
                          text=True, timeout=timeout)


def _norm_state(value: Any) -> str:
    s = str(value or "toggle").strip().lower()
    if s in ("on", "açık", "acik", "aç", "ac", "true", "1", "start", "enable"):
        return "on"
    if s in ("off", "kapalı", "kapali", "kapat", "false", "0", "stop", "disable"):
        return "off"
    return "toggle"


# ── state probes ──────────────────────────────────────────────────────────────

def _glow_running() -> bool:
    proc = Path("/proc")
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmdline = (p / "cmdline").read_bytes()
        except OSError:
            continue
        if b"musicglow" in cmdline and b"qs" in cmdline:
            return True
    return False


def _album_theme_active() -> bool:
    r = _run(["systemctl", "--user", "is-active", "album-theme.service"])
    return r.stdout.strip() == "active"


def _glow_mode() -> str:
    try:
        return _GLOW_MODE_NAMES.get(GLOW_MODE_FILE.read_text().strip(), "waves")
    except OSError:
        return "waves"


# ── the tool ──────────────────────────────────────────────────────────────────

class LanternTool(BaseTool):
    name = "lantern"
    description = (
        "Control the desktop ambiance (Lantern setup): focus mode (DND+dim), "
        "music edge-glow on/off and its visual mode, album-art theming, "
        "desktop sounds, a composite night mode, and an ambiance status report."
    )
    risk_level = "safe"

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action") or "status").strip().lower()
        try:
            if action == "focus":
                return self._set_focus(_norm_state(kwargs.get("state")))
            if action == "glow":
                return self._set_glow(_norm_state(kwargs.get("state")))
            if action == "glow_mode":
                return self._set_glow_mode(str(kwargs.get("mode") or ""))
            if action == "album_theme":
                return self._set_album(_norm_state(kwargs.get("state")))
            if action == "sounds":
                return self._set_sounds(_norm_state(kwargs.get("state")))
            if action == "night":
                return self._set_night(_norm_state(kwargs.get("state")))
            if action == "status":
                return self._status()
            return ToolResult(success=False,
                              output=f"unknown action '{action}'; see tool description")
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the brain
            log.exception("lantern action failed")
            return ToolResult(success=False, output=f"lantern error: {exc}")

    # -- individual actions ---------------------------------------------------

    def _set_focus(self, state: str) -> ToolResult:
        current = FOCUS_FLAG.exists()
        want = {"on": True, "off": False}.get(state, not current)
        if want != current:
            _run([str(BIN / "focus-mode.sh")])
        return ToolResult(success=True,
                          output=f"focus mode {'on' if want else 'off'}")

    def _set_glow(self, state: str) -> ToolResult:
        current = _glow_running()
        want = {"on": True, "off": False}.get(state, not current)
        if want != current:
            _run([str(BIN / "musicglow-toggle.sh")])
        return ToolResult(success=True,
                          output=f"music glow {'on' if want else 'off'}")

    def _set_glow_mode(self, mode: str) -> ToolResult:
        key = mode.strip().lower()
        if key not in _GLOW_MODES:
            return ToolResult(success=False,
                              output=f"unknown glow mode '{mode}' (waves|pulse|hybrid)")
        GLOW_MODE_FILE.write_text(_GLOW_MODES[key] + "\n")
        return ToolResult(success=True, output=f"glow mode set to {key}")

    def _set_album(self, state: str) -> ToolResult:
        current = _album_theme_active()
        want = {"on": True, "off": False}.get(state, not current)
        if want != current:
            _run([str(BIN / "album-theme-toggle.sh")])
        return ToolResult(success=True,
                          output=f"album theming {'on' if want else 'off'}")

    def _set_sounds(self, state: str) -> ToolResult:
        if state == "toggle":
            state = "off" if not SOUNDS_OFF_FLAG.exists() else "on"
        if state == "on":
            SOUNDS_OFF_FLAG.unlink(missing_ok=True)
        else:
            SOUNDS_OFF_FLAG.touch()
        return ToolResult(success=True, output=f"desktop sounds {state}")

    def _set_night(self, state: str) -> ToolResult:
        """Composite evening preset. on: soft pulse glow, DND, sounds off.
        off: hybrid glow, notifications back, sounds on."""
        if state not in ("on", "off"):
            state = "on"
        if state == "on":
            GLOW_MODE_FILE.write_text("1\n")
            SOUNDS_OFF_FLAG.touch()
            _run(["qs", "-c", "ii", "ipc", "call", "notifications",
                  "setSilent", "true"])
            out = "night mode on: glow softened to pulse, sounds muted, notifications silenced"
        else:
            GLOW_MODE_FILE.write_text("2\n")
            SOUNDS_OFF_FLAG.unlink(missing_ok=True)
            _run(["qs", "-c", "ii", "ipc", "call", "notifications",
                  "setSilent", "false"])
            out = "night mode off: hybrid glow, sounds and notifications restored"
        return ToolResult(success=True, output=out)

    def _status(self) -> ToolResult:
        lines = [
            f"focus mode: {'on' if FOCUS_FLAG.exists() else 'off'}",
            f"music glow: {'on' if _glow_running() else 'off'} (mode: {_glow_mode()})",
            f"album theming: {'on' if _album_theme_active() else 'off'}",
            f"desktop sounds: {'off' if SOUNDS_OFF_FLAG.exists() else 'on'}",
        ]
        try:  # battery: date,full_wh,design_wh,health_pct,...
            last = BATTERY_CSV.read_text().strip().splitlines()[-1].split(",")
            lines.append(f"battery health: {last[3]}% ({last[1]}/{last[2]} Wh, logged {last[0]})")
        except (OSError, IndexError):
            pass
        try:
            gpu = GPU_STATS.read_text().strip()
            if gpu == "suspended":
                lines.append("gpu: asleep (power saving)")
            else:
                t, util, mem_used, mem_total, power = [s.strip() for s in gpu.split(",")][:5]
                lines.append(f"gpu: {t}°C, {util}% load, {mem_used}/{mem_total} MiB, {power} W")
        except (OSError, ValueError):
            pass
        return ToolResult(success=True, output="\n".join(lines))


registry.register(LanternTool())
