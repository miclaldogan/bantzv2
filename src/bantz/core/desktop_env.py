"""
Bantz v2 — Wayland/Hyprland environment self-healing.

The daemon runs as a systemd **user** service. Whether it inherits
``WAYLAND_DISPLAY`` / ``HYPRLAND_INSTANCE_SIGNATURE`` depends on timing:
if the unit starts before the compositor runs ``systemctl --user
import-environment`` (e.g. at boot, or after a crash-restart), those vars
are missing — and then every ``grim`` screenshot ("failed to create
display") and every ``hyprctl`` call ("HYPRLAND_INSTANCE_SIGNATURE not
set!") fails even though Hyprland is running fine.

Rather than depend on systemd propagation, discover the live values from
``$XDG_RUNTIME_DIR`` (the compositor creates a ``wayland-*`` socket and a
``hypr/<signature>/`` directory there) and patch ``os.environ`` in place.
Idempotent and cheap (a few stats); safe to call on every screenshot /
workspace action as well as once at daemon startup. Also re-discovers when
a var is present but **stale** (points at a path that no longer exists,
i.e. the compositor was restarted).
"""
from __future__ import annotations

import glob
import logging
import os

log = logging.getLogger("bantz.desktop_env")


def _runtime_dir() -> str:
    return os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"


def ensure_wayland_env() -> dict[str, str]:
    """Repair WAYLAND_DISPLAY / HYPRLAND_INSTANCE_SIGNATURE in os.environ.

    Returns a dict of whatever was (re)set, for logging/tests.
    """
    fixed: dict[str, str] = {}
    xdg = _runtime_dir()
    os.environ.setdefault("XDG_RUNTIME_DIR", xdg)
    if not os.path.isdir(xdg):
        return fixed

    # ── WAYLAND_DISPLAY ──
    wd = os.environ.get("WAYLAND_DISPLAY")
    wd_path = os.path.join(xdg, wd) if wd else ""
    if not wd or not os.path.exists(wd_path):
        socks = [
            s for s in glob.glob(os.path.join(xdg, "wayland-*"))
            if not s.endswith(".lock")
        ]
        if socks:
            # Newest socket = the live compositor's (survives stale leftovers).
            newest = max(socks, key=os.path.getmtime)
            os.environ["WAYLAND_DISPLAY"] = os.path.basename(newest)
            fixed["WAYLAND_DISPLAY"] = os.environ["WAYLAND_DISPLAY"]

    # ── HYPRLAND_INSTANCE_SIGNATURE ──
    sig = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
    hypr_dir = os.path.join(xdg, "hypr")
    sig_valid = bool(sig) and os.path.isdir(os.path.join(hypr_dir, sig or ""))
    if not sig_valid and os.path.isdir(hypr_dir):
        cands = [d for d in glob.glob(os.path.join(hypr_dir, "*"))
                 if os.path.isdir(d)]
        if cands:
            newest = max(cands, key=os.path.getmtime)
            os.environ["HYPRLAND_INSTANCE_SIGNATURE"] = os.path.basename(newest)
            fixed["HYPRLAND_INSTANCE_SIGNATURE"] = \
                os.environ["HYPRLAND_INSTANCE_SIGNATURE"]

    if fixed:
        log.info("Repaired desktop env from %s: %s", xdg, list(fixed))
    return fixed
