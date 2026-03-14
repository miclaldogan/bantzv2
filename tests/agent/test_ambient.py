"""Tests for bantz.agent.ambient — Issue #166 Ambient Sound Analyser."""
from __future__ import annotations

import math
import time
from unittest.mock import patch

import pytest

from bantz.agent.ambient import (
    AmbientAnalyzer,
    AmbientLabel,
    AmbientSnapshot,
    classify,
    compute_rms,
    compute_zcr,
)


# ═══════════════════════════════════════════════════════════════════════════
# DSP helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeRMS:
    """Tests for compute_rms()."""

    def test_empty_returns_zero(self):
        assert compute_rms([]) == 0.0

    def test_silence_very_low(self):
        """All-zero samples → RMS = 0."""
        assert compute_rms([0] * 1000) == 0.0

    def test_constant_signal(self):
        """Constant signal [k, k, k] → RMS = |k|."""
        rms = compute_rms([1000] * 100)
        assert rms == pytest.approx(1000.0, rel=0.01)

    def test_sine_wave(self):
        """Sine wave RMS ≈ amplitude / sqrt(2)."""
        amplitude = 10000
        samples = [int(amplitude * math.sin(2 * math.pi * i / 100)) for i in range(1000)]
        rms = compute_rms(samples)
        expected = amplitude / math.sqrt(2)
        assert rms == pytest.approx(expected, rel=0.05)

    def test_single_sample(self):
        assert compute_rms([5000]) == 5000.0

    def test_mixed_positive_negative(self):
        """RMS of [A, -A] = A."""
        rms = compute_rms([3000, -3000])
        assert rms == pytest.approx(3000.0, rel=0.01)


class TestComputeZCR:
    """Tests for compute_zcr()."""

    def test_empty_returns_zero(self):
        assert compute_zcr([]) == 0.0

    def test_single_sample_returns_zero(self):
        assert compute_zcr([1000]) == 0.0

    def test_all_positive_no_crossings(self):
        """All positive → ZCR = 0."""
        assert compute_zcr([100, 200, 300, 400]) == 0.0

    def test_alternating_sign_max_zcr(self):
        """Alternating +/- → ZCR = 1.0."""
        samples = [1000, -1000, 1000, -1000, 1000]
        assert compute_zcr(samples) == 1.0

    def test_half_crossings(self):
        """Two positives then two negatives → 1 crossing out of 3 pairs."""
        samples = [100, 200, -300, -400]
        assert compute_zcr(samples) == pytest.approx(1 / 3, rel=0.01)

    def test_zero_as_non_negative(self):
        """0 is treated as non-negative (>= 0)."""
        samples = [0, -1, 0, -1]
        # 0→-1 cross, -1→0 cross, 0→-1 cross → 3 crossings / 3 pairs
        assert compute_zcr(samples) == 1.0

    def test_speech_like_zcr(self):
        """Speech-like signal: moderate ZCR (0.02 – 0.10)."""
        # Simulate speech: slow sine (200 Hz at 16 kHz → ~25 crossings per 800 samples)
        samples = [int(5000 * math.sin(2 * math.pi * 200 * i / 16000)) for i in range(800)]
        zcr = compute_zcr(samples)
        assert 0.01 < zcr < 0.15


class TestClassify:
    """Tests for classify() decision tree."""

    def test_low_rms_is_silence(self):
        assert classify(rms=100, zcr=0.5) == AmbientLabel.SILENCE

    def test_boundary_rms_silence(self):
        assert classify(rms=499, zcr=0.9) == AmbientLabel.SILENCE

    def test_moderate_rms_low_zcr_is_speech(self):
        assert classify(rms=2000, zcr=0.05) == AmbientLabel.SPEECH

    def test_high_rms_high_zcr_is_noisy(self):
        assert classify(rms=5000, zcr=0.25) == AmbientLabel.NOISY

    def test_boundary_zcr_speech_vs_noisy(self):
        """ZCR exactly at 0.12 → NOISY (>= floor)."""
        assert classify(rms=3000, zcr=0.12) == AmbientLabel.NOISY

    def test_just_below_zcr_boundary_is_speech(self):
        assert classify(rms=3000, zcr=0.119) == AmbientLabel.SPEECH


# ═══════════════════════════════════════════════════════════════════════════
# AmbientSnapshot
# ═══════════════════════════════════════════════════════════════════════════

class TestAmbientSnapshot:
    def test_to_dict(self):
        snap = AmbientSnapshot(
            timestamp=1700000000.0,
            rms=2500.7,
            zcr=0.06789,
            label=AmbientLabel.SPEECH,
            duration_s=3.0,
        )
        d = snap.to_dict()
        assert d["label"] == "speech"
        assert d["rms"] == 2500.7
        assert d["zcr"] == 0.0679  # rounded to 4 decimals
        assert d["duration_s"] == 3.0


# ═══════════════════════════════════════════════════════════════════════════
# AmbientAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

class TestAmbientAnalyzerFeed:
    """Tests for AmbientAnalyzer.feed_frames()."""

    def _make_analyzer(self, interval=0.0, window=0.1, rate=16000):
        a = AmbientAnalyzer(
            sample_rate=rate,
            sample_window_s=window,
            sample_interval_s=interval,
        )
        return a

    def test_disabled_returns_none(self):
        a = self._make_analyzer()
        a.enabled = False
        assert a.feed_frames([0] * 1600) is None

    def test_not_enough_frames_returns_none(self):
        """First feed with fewer frames than window → None."""
        a = self._make_analyzer(window=1.0)  # need 16000 frames
        result = a.feed_frames([0] * 100)
        assert result is None

    def test_single_window_silence(self):
        """Feed exactly one window of silence → snapshot with SILENCE label."""
        a = self._make_analyzer(interval=0.0, window=0.1)
        # 0.1s @ 16kHz = 1600 frames
        result = a.feed_frames([0] * 1600)
        assert result is not None
        assert result.label == AmbientLabel.SILENCE
        assert result.rms < 500

    def test_single_window_speech(self):
        """Feed a speech-like sine wave → SPEECH label."""
        a = self._make_analyzer(interval=0.0, window=0.1)
        # 200 Hz sine at amplitude 3000 → moderate RMS, low ZCR
        samples = [int(3000 * math.sin(2 * math.pi * 200 * i / 16000)) for i in range(1600)]
        result = a.feed_frames(samples)
        assert result is not None
        assert result.label == AmbientLabel.SPEECH

    def test_single_window_noisy(self):
        """Feed alternating high-amplitude signal → NOISY."""
        a = self._make_analyzer(interval=0.0, window=0.1)
        # Alternating +10000 / -10000 → high RMS, ZCR = 1.0
        samples = [10000 * (1 if i % 2 == 0 else -1) for i in range(1600)]
        result = a.feed_frames(samples)
        assert result is not None
        assert result.label == AmbientLabel.NOISY

    def test_accumulation_across_multiple_feeds(self):
        """Multiple small feeds accumulate until window is full."""
        a = self._make_analyzer(interval=0.0, window=0.1)
        chunk_size = 512  # Porcupine typical frame length
        total_needed = 1600
        fed = 0
        result = None
        while fed < total_needed:
            n = min(chunk_size, total_needed - fed)
            result = a.feed_frames([0] * n)
            fed += n
            if result is not None:
                break
        assert result is not None
        assert result.label == AmbientLabel.SILENCE

    def test_interval_gating(self):
        """Second analysis is blocked until sample_interval elapses."""
        a = self._make_analyzer(interval=60.0, window=0.1)
        # First analysis: should work
        r1 = a.feed_frames([0] * 1600)
        assert r1 is not None

        # Immediately after: should be gated
        r2 = a.feed_frames([0] * 1600)
        assert r2 is None

    def test_interval_gating_expired(self):
        """After interval elapses, next analysis proceeds."""
        a = self._make_analyzer(interval=1.0, window=0.1)
        r1 = a.feed_frames([0] * 1600)
        assert r1 is not None

        # Fast-forward time
        a._last_analysis = time.monotonic() - 2.0
        r2 = a.feed_frames([0] * 1600)
        assert r2 is not None

    def test_history_populated(self):
        a = self._make_analyzer(interval=0.0, window=0.1)
        for _ in range(5):
            a.feed_frames([0] * 1600)
            a._last_analysis = 0  # reset to allow immediate re-analysis
        hist = a.history(10)
        assert len(hist) == 5

    def test_total_analyses_incremented(self):
        a = self._make_analyzer(interval=0.0, window=0.1)
        a.feed_frames([0] * 1600)
        assert a._total_analyses == 1
        a._last_analysis = 0
        a.feed_frames([0] * 1600)
        assert a._total_analyses == 2


class TestAmbientAnalyzerQuery:
    """Tests for query methods."""

    def _populated_analyzer(self, n_silence=3, n_speech=2, n_noisy=1):
        a = AmbientAnalyzer(sample_rate=16000, sample_window_s=0.1, sample_interval_s=0)
        now = time.time()
        for i in range(n_silence):
            snap = AmbientSnapshot(now - i * 600, 100.0, 0.01, AmbientLabel.SILENCE, 3.0)
            a._history.append(snap)
        for i in range(n_speech):
            snap = AmbientSnapshot(now - (n_silence + i) * 600, 2500.0, 0.06, AmbientLabel.SPEECH, 3.0)
            a._history.append(snap)
        for i in range(n_noisy):
            snap = AmbientSnapshot(now - (n_silence + n_speech + i) * 600, 8000.0, 0.25, AmbientLabel.NOISY, 3.0)
            a._history.append(snap)
        a._latest = a._history[0] if a._history else None
        a._total_analyses = n_silence + n_speech + n_noisy
        return a

    def test_latest(self):
        a = self._populated_analyzer()
        assert a.latest() is not None
        assert a.latest().label == AmbientLabel.SILENCE

    def test_history_limit(self):
        a = self._populated_analyzer(n_silence=10, n_speech=0, n_noisy=0)
        assert len(a.history(5)) == 5
        assert len(a.history(20)) == 10

    def test_label_distribution(self):
        a = self._populated_analyzer(n_silence=3, n_speech=2, n_noisy=1)
        dist = a.label_distribution(24.0)
        assert dist["silence"] == 3
        assert dist["speech"] == 2
        assert dist["noisy"] == 1

    def test_day_summary_no_data(self):
        a = AmbientAnalyzer()
        assert "No ambient" in a.day_summary()

    def test_day_summary_with_data(self):
        a = self._populated_analyzer()
        summary = a.day_summary()
        assert "silence" in summary
        assert "samples" in summary

    def test_ambient_bucket_no_data(self):
        a = AmbientAnalyzer()
        assert a.ambient_bucket() == "unknown"

    def test_ambient_bucket_fresh(self):
        a = AmbientAnalyzer(sample_interval_s=600)
        a._latest = AmbientSnapshot(time.time(), 2000, 0.05, AmbientLabel.SPEECH, 3.0)
        assert a.ambient_bucket() == "speech"

    def test_ambient_bucket_stale(self):
        a = AmbientAnalyzer(sample_interval_s=600)
        a._latest = AmbientSnapshot(time.time() - 2000, 2000, 0.05, AmbientLabel.SPEECH, 3.0)
        assert a.ambient_bucket() == "unknown"


class TestAmbientAnalyzerDiagnostics:
    def test_stats(self):
        a = AmbientAnalyzer()
        s = a.stats()
        assert s["enabled"] is True
        assert s["total_analyses"] == 0
        assert s["latest"] is None

    def test_status_line_no_data(self):
        a = AmbientAnalyzer()
        assert "waiting" in a.status_line()

    def test_status_line_disabled(self):
        a = AmbientAnalyzer()
        a.enabled = False
        assert "disabled" in a.status_line()

    def test_diagnose(self):
        a = AmbientAnalyzer()
        d = a.diagnose()
        assert "enabled" in d
        assert "window_frames" in d
        assert d["has_data"] is False

    def test_reset(self):
        a = AmbientAnalyzer(sample_rate=16000, sample_window_s=0.1, sample_interval_s=0)
        a.feed_frames([0] * 1600)
        assert a._total_analyses == 1
        a.reset()
        assert a._total_analyses == 0
        assert a.latest() is None
        assert len(a.history()) == 0


# ═══════════════════════════════════════════════════════════════════════════
# EventBus integration (Sprint 3 Part 2)
# ═══════════════════════════════════════════════════════════════════════════

class TestAmbientEventBus:
    """Verify ambient_change events are emitted via the EventBus."""

    def test_feed_frames_emits_event(self):
        """A successful analysis should emit 'ambient_change'."""
        from bantz.core.event_bus import EventBus
        with patch("bantz.agent.ambient.bus") as mock_bus:
            a = AmbientAnalyzer(sample_rate=16000, sample_window_s=0.1, sample_interval_s=0)
            result = a.feed_frames([0] * 1600)
            assert result is not None
            mock_bus.emit_threadsafe.assert_called_once()
            call_args = mock_bus.emit_threadsafe.call_args
            assert call_args[0][0] == "ambient_change"
            assert "label" in call_args[1]
            assert call_args[1]["label"] == "silence"

    def test_no_event_when_disabled(self):
        """Disabled analyser emits nothing."""
        with patch("bantz.agent.ambient.bus") as mock_bus:
            a = AmbientAnalyzer(sample_rate=16000, sample_window_s=0.1, sample_interval_s=0)
            a.enabled = False
            a.feed_frames([0] * 1600)
            mock_bus.emit_threadsafe.assert_not_called()

    def test_no_event_when_not_enough_frames(self):
        """Partial feed (not enough for window) emits nothing."""
        with patch("bantz.agent.ambient.bus") as mock_bus:
            a = AmbientAnalyzer(sample_rate=16000, sample_window_s=1.0, sample_interval_s=0)
            a.feed_frames([0] * 100)
            mock_bus.emit_threadsafe.assert_not_called()

    def test_bus_exception_does_not_crash(self):
        """If bus.emit_threadsafe raises, feed_frames still returns the snapshot."""
        with patch("bantz.agent.ambient.bus") as mock_bus:
            mock_bus.emit_threadsafe.side_effect = RuntimeError("bus down")
            a = AmbientAnalyzer(sample_rate=16000, sample_window_s=0.1, sample_interval_s=0)
            result = a.feed_frames([0] * 1600)
            assert result is not None  # snapshot still returned

    def test_no_brain_or_tui_imports(self):
        """ambient.py must NOT import from brain or TUI."""
        import ast, inspect
        from bantz.agent import ambient
        tree = ast.parse(inspect.getsource(ambient))
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        for imp in imports:
            if imp.module:
                assert "bantz.core.brain" not in imp.module, \
                    f"Forbidden import: {imp.module}"
                assert "bantz.interface.tui" not in imp.module, \
                    f"Forbidden import: {imp.module}"
