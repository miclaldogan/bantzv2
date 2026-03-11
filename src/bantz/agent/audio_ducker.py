"""
Bantz — Audio Ducking (#171)

Lowers other applications' audio volume while Bantz speaks via TTS,
then restores it afterwards — like car navigation ducking the music.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  AudioDucker                                             │
    │                                                          │
    │   duck()                                                 │
    │   ├─ pactl list sink-inputs (parse JSON)                 │
    │   ├─ skip entries with application.name = "BantzTTS"     │
    │   ├─ save original volumes  {sink_input_id → volume}     │
    │   └─ pactl set-sink-input-volume <id> <duck_pct>%        │
    │                                                          │
    │   restore()                                              │
    │   ├─ for each saved {id → volume}:                       │
    │   │   pactl set-sink-input-volume <id> <original>        │
    │   └─ clear saved state                                   │
    └──────────────────────────────────────────────────────────┘

Key decisions:
  - pactl ONLY — works on both PulseAudio and PipeWire (via
    pipewire-pulse bridge).  No wpctl parsing nightmare.
  - PULSE_PROP tagging: tts.py sets PULSE_PROP=application.name='BantzTTS'
    on aplay so the ducker can filter Bantz's own audio by name.
    No unreliable PID matching.
  - Sync-only: instant volume set (no async fade in MVP).  Fast 3-step
    fade with time.sleep(15ms) optional enhancement.
  - Thread-safe: duck/restore are idempotent and lock-protected.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
import time
from typing import Optional

log = logging.getLogger("bantz.audio_ducker")

# ── Bantz TTS label — must match PULSE_PROP in tts.py ──────────────────
_BANTZ_APP_NAME = "BantzTTS"

# Number of fade steps when ducking/restoring (MVP: 3 steps × 15ms = 45ms)
_FADE_STEPS = 3
_FADE_STEP_MS = 15


class AudioDucker:
    """Ducks other applications' audio while Bantz's TTS is active.

    Usage (called internally by TTSEngine):
        ducker.duck()     # lower other apps' volume
        ... play TTS ...
        ducker.restore()  # restore original volumes
    """

    def __init__(self) -> None:
        self._pactl: str | None = None          # path to pactl binary
        self._duck_pct: int = 30                # ducked volume percentage
        self._saved: dict[int, int] = {}        # sink_input_id → original volume %
        self._ducked: bool = False
        self._lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────

    def available(self) -> bool:
        """Check if audio ducking is possible (pactl present)."""
        return self._ensure_init()

    @property
    def is_ducked(self) -> bool:
        return self._ducked

    def duck(self) -> bool:
        """Lower volume of all non-Bantz audio streams.

        Returns True if ducking was applied (at least one stream ducked).
        Safe to call multiple times — second call is a no-op.
        """
        with self._lock:
            if self._ducked:
                return True
            if not self._ensure_init():
                return False

            inputs = self._get_sink_inputs()
            if not inputs:
                return False

            targets = [
                si for si in inputs
                if si["app_name"] != _BANTZ_APP_NAME
            ]
            if not targets:
                return False

            # Save originals and duck
            for si in targets:
                sid = si["id"]
                self._saved[sid] = si["volume"]
                self._set_volume(sid, self._duck_pct)

            self._ducked = True
            log.info("Audio ducked: %d stream(s) → %d%%",
                     len(targets), self._duck_pct)
            return True

    def restore(self) -> None:
        """Restore all ducked streams to their original volumes.

        Safe to call multiple times — second call is a no-op.
        """
        with self._lock:
            if not self._ducked or not self._saved:
                self._ducked = False
                return
            if not self._pactl:
                self._ducked = False
                self._saved.clear()
                return

            for sid, original_vol in self._saved.items():
                self._set_volume(sid, original_vol)

            log.info("Audio restored: %d stream(s)", len(self._saved))
            self._saved.clear()
            self._ducked = False

    def duck_with_fade(self) -> bool:
        """Duck with a quick 3-step fade (~45ms) instead of instant cut.

        Falls back to instant duck() on error.
        """
        with self._lock:
            if self._ducked:
                return True
            if not self._ensure_init():
                return False

            inputs = self._get_sink_inputs()
            if not inputs:
                return False

            targets = [
                si for si in inputs
                if si["app_name"] != _BANTZ_APP_NAME
            ]
            if not targets:
                return False

            # Save originals
            for si in targets:
                self._saved[si["id"]] = si["volume"]

            # Fade down in _FADE_STEPS steps
            for step in range(1, _FADE_STEPS + 1):
                for si in targets:
                    original = self._saved[si["id"]]
                    target = self._duck_pct
                    current = original - int(
                        (original - target) * step / _FADE_STEPS
                    )
                    self._set_volume(si["id"], max(target, current))
                if step < _FADE_STEPS:
                    time.sleep(_FADE_STEP_MS / 1000.0)

            self._ducked = True
            log.info("Audio ducked (fade): %d stream(s) → %d%%",
                     len(targets), self._duck_pct)
            return True

    def restore_with_fade(self) -> None:
        """Restore with a quick 3-step fade (~45ms)."""
        with self._lock:
            if not self._ducked or not self._saved:
                self._ducked = False
                return
            if not self._pactl:
                self._ducked = False
                self._saved.clear()
                return

            # Fade up in _FADE_STEPS steps
            for step in range(1, _FADE_STEPS + 1):
                for sid, original_vol in self._saved.items():
                    current = self._duck_pct + int(
                        (original_vol - self._duck_pct) * step / _FADE_STEPS
                    )
                    self._set_volume(sid, min(original_vol, current))
                if step < _FADE_STEPS:
                    time.sleep(_FADE_STEP_MS / 1000.0)

            log.info("Audio restored (fade): %d stream(s)", len(self._saved))
            self._saved.clear()
            self._ducked = False

    # ── Diagnostics ─────────────────────────────────────────────────────

    def diagnose(self) -> dict:
        """Return diagnostic info for --doctor."""
        return {
            "pactl_available": shutil.which("pactl") is not None,
            "ducked": self._ducked,
            "saved_streams": len(self._saved),
            "duck_pct": self._duck_pct,
        }

    def stats(self) -> dict:
        return {
            "ducked": self._ducked,
            "saved_streams": len(self._saved),
            "duck_pct": self._duck_pct,
        }

    def status_line(self) -> str:
        if not self._ensure_init():
            return "audio_ducking=unavailable (no pactl)"
        state = "active" if self._ducked else "idle"
        return f"audio_ducking={state} duck_pct={self._duck_pct}%"

    # ── Internal ────────────────────────────────────────────────────────

    def _ensure_init(self) -> bool:
        """Find pactl binary (lazy). Returns True if ready."""
        if self._pactl is not None:
            return bool(self._pactl)
        pactl = shutil.which("pactl")
        if pactl:
            self._pactl = pactl
            self._load_config()
            log.debug("Audio ducker: pactl found at %s", pactl)
            return True
        self._pactl = ""
        log.warning("Audio ducker: pactl not found — ducking disabled")
        return False

    def _load_config(self) -> None:
        """Load ducking config from bantz config."""
        try:
            from bantz.config import config
            self._duck_pct = config.audio_duck_pct
        except Exception:
            pass

    def _get_sink_inputs(self) -> list[dict]:
        """List active audio sink inputs via pactl.

        Returns list of dicts: {"id": int, "app_name": str, "volume": int}
        """
        if not self._pactl:
            return []
        try:
            result = subprocess.run(
                [self._pactl, "list", "sink-inputs"],
                capture_output=True, text=True, timeout=5.0,
            )
            if result.returncode != 0:
                return []
            return self._parse_sink_inputs(result.stdout)
        except Exception as exc:
            log.warning("Audio ducker: pactl list failed — %s", exc)
            return []

    @staticmethod
    def _parse_sink_inputs(output: str) -> list[dict]:
        """Parse 'pactl list sink-inputs' output into structured data.

        Extracts sink input index, volume percentage, and application.name.
        """
        inputs: list[dict] = []
        current: dict | None = None

        for line in output.splitlines():
            line_stripped = line.strip()

            # New sink input block
            m = re.match(r"Sink Input #(\d+)", line)
            if m:
                if current:
                    inputs.append(current)
                current = {
                    "id": int(m.group(1)),
                    "app_name": "",
                    "volume": 100,
                }
                continue

            if not current:
                continue

            # Volume line: "Volume: front-left: 42000 /  64% / ..."
            if line_stripped.startswith("Volume:"):
                vol_match = re.search(r"(\d+)%", line_stripped)
                if vol_match:
                    current["volume"] = int(vol_match.group(1))
                continue

            # Application name property
            if "application.name" in line_stripped:
                name_match = re.search(
                    r'application\.name\s*=\s*["\']?([^"\']+)["\']?',
                    line_stripped,
                )
                if name_match:
                    current["app_name"] = name_match.group(1).strip()
                continue

        # Don't forget last entry
        if current:
            inputs.append(current)

        return inputs

    def _set_volume(self, sink_input_id: int, volume_pct: int) -> bool:
        """Set volume of a specific sink input via pactl."""
        if not self._pactl:
            return False
        volume_pct = max(0, min(150, volume_pct))
        try:
            subprocess.run(
                [self._pactl, "set-sink-input-volume",
                 str(sink_input_id), f"{volume_pct}%"],
                capture_output=True, timeout=3.0,
            )
            return True
        except Exception as exc:
            log.warning("Audio ducker: set-volume failed for #%d — %s",
                        sink_input_id, exc)
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

audio_ducker = AudioDucker()
