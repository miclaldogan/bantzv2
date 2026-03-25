"""
Bantz — Ambient Sound Analyser (#166)

Passive environmental audio classifier that piggybacks on the
WakeWordListener's existing microphone stream.  **No new pyaudio
stream is ever opened** — the wake word loop feeds raw PCM frames
into a rolling buffer, and every ``sample_interval`` seconds the
analyser computes lightweight features (RMS + ZCR) to classify the
environment.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    WakeWordListener                         │
    │   _listen_loop()                                            │
    │   ┌─ pcm = stream.read(frame_length)                       │
    │   │  porcupine.process(pcm)                                │
    │   │  ambient_analyzer.feed_frames(pcm)  ← piggybacking     │
    │   └─────────────────────────────────────────────────────────│
    └──────────────────┬──────────────────────────────────────────┘
                       │ every sample_interval_s
                       ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   AmbientAnalyzer                           │
    │                                                             │
    │   feed_frames(pcm)  → accumulates into _ring_buf            │
    │                     → when ≥ sample_window_s of data:       │
    │                        compute RMS + ZCR                    │
    │                        classify → AmbientLabel              │
    │                        store in _history deque              │
    │                        notify RL engine (passive signal)    │
    └─────────────────────────────────────────────────────────────┘

Labels (MVP — no FFT):
    SILENCE  — very low RMS
    SPEECH   — moderate RMS + moderate/steady ZCR
    NOISY    — high RMS + high/erratic ZCR
    UNKNOWN  — fallback if classification is ambiguous

Used by:
    • RL engine → ambient_bucket enriches State for smarter suggestions
    • Nightly Reflection → day-level ambient summary in the journal
    • **NOT** used by TTS — see Issue #171 audio ducking instead

Key decisions:
    1. Zero extra hardware cost — reuses wake word mic stream
    2. RMS + ZCR only (no FFT / spectral centroid in MVP)
    3. Pure numpy-free — stdlib math only for portability
    4. Thread-safe — called from wake word's audio thread
    5. 10-minute default sample interval, 3-second analysis window
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from bantz.core.event_bus import bus

log = logging.getLogger("bantz.ambient")

# ── Defaults ─────────────────────────────────────────────────────────────

_DEFAULT_SAMPLE_RATE = 16000       # Hz (Porcupine default)
_DEFAULT_SAMPLE_WINDOW = 3.0       # seconds of audio to analyse
_DEFAULT_SAMPLE_INTERVAL = 600.0   # seconds between analyses (10 min)

# ── Classification thresholds (tuned for 16-bit PCM @ 16 kHz) ────────────
#
# RMS of 16-bit silence ≈ 0–200
# RMS of normal speech ≈ 1500–5000
# RMS of noisy env     ≈ 4000–15000+
#
# ZCR (zero-crossing rate) = fraction of adjacent samples that cross zero
#   Clean speech: 0.02–0.10
#   Noisy / music: 0.10–0.30+
#   Silence: variable but irrelevant (masked by low RMS)

_SILENCE_RMS_CEIL = 500         # below this → SILENCE
_SPEECH_ZCR_CEIL = 0.12         # speech has steady, moderate ZCR
_NOISY_ZCR_FLOOR = 0.12         # above this with high RMS → NOISY


# ── Data types ───────────────────────────────────────────────────────────

class AmbientLabel(str, Enum):
    """Coarse environmental sound classification."""
    SILENCE = "silence"
    SPEECH = "speech"
    NOISY = "noisy"
    UNKNOWN = "unknown"


@dataclass
class AmbientSnapshot:
    """One ambient measurement."""
    timestamp: float
    rms: float
    zcr: float
    label: AmbientLabel
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "rms": round(self.rms, 1),
            "zcr": round(self.zcr, 4),
            "label": self.label.value,
            "duration_s": round(self.duration_s, 1),
        }


# ── Pure-Python DSP helpers ──────────────────────────────────────────────

def compute_rms(samples: list[int]) -> float:
    """Root Mean Square of 16-bit PCM samples."""
    if not samples:
        return 0.0
    sq_sum = sum(s * s for s in samples)
    return math.sqrt(sq_sum / len(samples))


def compute_zcr(samples: list[int]) -> float:
    """Zero-Crossing Rate: fraction of adjacent pairs that cross zero."""
    if len(samples) < 2:
        return 0.0
    crossings = sum(
        1 for i in range(1, len(samples))
        if (samples[i - 1] >= 0) != (samples[i] >= 0)
    )
    return crossings / (len(samples) - 1)


def classify(rms: float, zcr: float) -> AmbientLabel:
    """Classify ambient environment from RMS and ZCR.

    Decision tree (MVP — no spectral centroid):
        rms < SILENCE_CEIL           → SILENCE
        rms ≥ SILENCE_CEIL:
            zcr < SPEECH_ZCR_CEIL    → SPEECH  (steady, moderate freq)
            zcr ≥ NOISY_ZCR_FLOOR   → NOISY   (chaotic, high freq)
    """
    if rms < _SILENCE_RMS_CEIL:
        return AmbientLabel.SILENCE
    if zcr < _SPEECH_ZCR_CEIL:
        return AmbientLabel.SPEECH
    return AmbientLabel.NOISY


# ── Analyser ─────────────────────────────────────────────────────────────

class AmbientAnalyzer:
    """Passive ambient sound analyser fed by the wake word audio stream.

    Thread-safe: ``feed_frames()`` is called from the audio thread,
    ``latest()``, ``history()`` etc. from any thread.
    """

    def __init__(
        self,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        sample_window_s: float = _DEFAULT_SAMPLE_WINDOW,
        sample_interval_s: float = _DEFAULT_SAMPLE_INTERVAL,
    ) -> None:
        self._sample_rate = sample_rate
        self._sample_window_s = sample_window_s
        self._sample_interval_s = sample_interval_s

        # Number of 16-bit mono samples needed for one analysis window
        self._window_frames = int(sample_rate * sample_window_s)

        # Rolling buffer for incoming PCM (int16 samples)
        self._buf: list[int] = []
        self._buf_lock = threading.Lock()

        # Timing: when was the last analysis?
        self._last_analysis: float = 0.0

        # Whether we are currently accumulating a sample
        self._accumulating = False

        # Result history (newest first)
        self._history: deque[AmbientSnapshot] = deque(maxlen=144)
        # 144 snapshots × 10 min = 24 h rolling window
        self._history_lock = threading.Lock()

        # Latest snapshot (quick access)
        self._latest: Optional[AmbientSnapshot] = None

        # Cumulative stats
        self._total_analyses = 0
        self._enabled = True

    # ── Configuration ────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val

    @property
    def sample_interval_s(self) -> float:
        return self._sample_interval_s

    @sample_interval_s.setter
    def sample_interval_s(self, val: float) -> None:
        self._sample_interval_s = max(0.0, val)  # allow 0 for tests, but practically bound

    # ── Feed from wake word ──────────────────────────────────────────

    def feed_frames(self, pcm: tuple | list) -> Optional[AmbientSnapshot]:
        """Called from the wake word audio thread with each Porcupine frame.

        Accumulates frames.  When enough data has been collected for one
        window AND the sample interval has elapsed, runs analysis and
        returns the snapshot.  Otherwise returns None.
        """
        if not self._enabled:
            return None

        now = time.monotonic()

        # Should we start accumulating?
        if not self._accumulating:
            if now - self._last_analysis < self._sample_interval_s:
                return None  # not time yet
            self._accumulating = True
            with self._buf_lock:
                self._buf.clear()

        # Accumulate frames
        with self._buf_lock:
            self._buf.extend(pcm)

            if len(self._buf) < self._window_frames:
                return None  # need more frames

            # We have enough — take a copy and analyse
            samples = self._buf[:self._window_frames]
            self._buf.clear()

        self._accumulating = False
        self._last_analysis = now

        # Compute features
        rms = compute_rms(samples)
        zcr = compute_zcr(samples)
        label = classify(rms, zcr)

        snap = AmbientSnapshot(
            timestamp=time.time(),
            rms=rms,
            zcr=zcr,
            label=label,
            duration_s=self._sample_window_s,
        )

        with self._history_lock:
            self._history.appendleft(snap)
        self._latest = snap
        self._total_analyses += 1

        log.info(
            "Ambient: %s (RMS=%.0f, ZCR=%.4f) [#%d]",
            label.value.upper(), rms, zcr, self._total_analyses,
        )

        # ── Publish to EventBus (Sprint 3 Part 2) ─────────────
        try:
            bus.emit_threadsafe("ambient_change", **snap.to_dict())
        except Exception:
            pass  # never crash the audio thread

        return snap

    # ── Query API ─────────────────────────────────────────────────────

    def latest(self) -> Optional[AmbientSnapshot]:
        """Return the most recent snapshot, or None."""
        return self._latest

    def history(self, n: int = 10) -> list[AmbientSnapshot]:
        """Return the last *n* snapshots (newest first)."""
        with self._history_lock:
            return list(self._history)[:n]

    def label_distribution(self, hours: float = 24.0) -> dict[str, int]:
        """Count labels in the last *hours* window.

        Returns e.g. {"silence": 80, "speech": 30, "noisy": 10}
        """
        cutoff = time.time() - hours * 3600
        dist: dict[str, int] = {l.value: 0 for l in AmbientLabel}
        with self._history_lock:
            for snap in self._history:
                if snap.timestamp < cutoff:
                    break
                dist[snap.label.value] += 1
        return dist

    def day_summary(self) -> str:
        """Human-readable day summary for nightly reflection."""
        dist = self.label_distribution(24.0)
        total = sum(dist.values())
        if total == 0:
            return "No ambient data collected today."

        parts: list[str] = []
        for label_name in ("silence", "speech", "noisy"):
            count = dist.get(label_name, 0)
            if count > 0:
                pct = count * 100 // total
                minutes = count * self._sample_interval_s / 60
                parts.append(f"{label_name}: {pct}% ({minutes:.0f} min)")

        return f"Ambient today ({total} samples): " + ", ".join(parts)

    def ambient_bucket(self) -> str:
        """Return current ambient label as a string for RL state encoding.

        Returns 'unknown' if no recent data (stale > 2× sample interval).
        """
        snap = self._latest
        if snap is None:
            return "unknown"
        age = time.time() - snap.timestamp
        if age > self._sample_interval_s * 2:
            return "unknown"
        return snap.label.value

    # ── Stats / Diagnostics ───────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        dist = self.label_distribution(24.0)
        return {
            "enabled": self._enabled,
            "total_analyses": self._total_analyses,
            "sample_interval_s": self._sample_interval_s,
            "sample_window_s": self._sample_window_s,
            "latest": self._latest.to_dict() if self._latest else None,
            "distribution_24h": dist,
        }

    def status_line(self) -> str:
        if not self._enabled:
            return "ambient=disabled"
        snap = self._latest
        if snap is None:
            return "ambient=waiting (no data yet)"
        age = int(time.time() - snap.timestamp)
        return (
            f"ambient={snap.label.value} "
            f"RMS={snap.rms:.0f} ZCR={snap.zcr:.3f} "
            f"({age}s ago, #{self._total_analyses})"
        )

    def diagnose(self) -> dict[str, Any]:
        """Return diagnostic info for --doctor."""
        return {
            "enabled": self._enabled,
            "total_analyses": self._total_analyses,
            "sample_rate": self._sample_rate,
            "window_frames": self._window_frames,
            "sample_interval_s": self._sample_interval_s,
            "has_data": self._latest is not None,
        }

    def reset(self) -> None:
        """Clear all state (for testing)."""
        with self._buf_lock:
            self._buf.clear()
        with self._history_lock:
            self._history.clear()
        self._latest = None
        self._total_analyses = 0
        self._last_analysis = 0.0
        self._accumulating = False


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

ambient_analyzer = AmbientAnalyzer()
