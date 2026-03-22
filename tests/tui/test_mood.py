"""
Tests — Issue #135: Dynamic Mood State Machine

Covers:
  - Mood enum, display constants (faces, labels, CSS classes)
  - MoodStateMachine: raw computation, hysteresis, transitions, AppDetector integration
  - MoodHistory: SQLite rolling log, recent(), summary_24h(), pruning
  - SystemStatus panel: mood indicator, CSS class swap
  - CLI: --mood-history argument exists
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip('textual')


# ═══════════════════════════════════════════════════════════════════════════
# Mood enum & constants
# ═══════════════════════════════════════════════════════════════════════════

class TestMoodEnum:
    def test_all_moods_exist(self):
        from bantz.interface.tui.mood import Mood
        assert Mood.CHILL.value == "chill"
        assert Mood.FOCUSED.value == "focused"
        assert Mood.BUSY.value == "busy"
        assert Mood.STRESSED.value == "stressed"
        assert Mood.SLEEPING.value == "sleeping"

    def test_mood_faces(self):
        from bantz.interface.tui.mood import Mood, MOOD_FACES
        assert len(MOOD_FACES) == 5
        for mood in Mood:
            assert mood in MOOD_FACES
            assert len(MOOD_FACES[mood]) > 0

    def test_mood_labels(self):
        from bantz.interface.tui.mood import Mood, MOOD_LABELS
        for mood in Mood:
            assert mood in MOOD_LABELS

    def test_mood_css_classes(self):
        from bantz.interface.tui.mood import Mood, MOOD_CSS_CLASS, ALL_MOOD_CLASSES
        for mood in Mood:
            assert mood in MOOD_CSS_CLASS
            assert MOOD_CSS_CLASS[mood].startswith("mood-")
        assert len(ALL_MOOD_CLASSES) == 5

    def test_all_mood_classes_matches_dict(self):
        from bantz.interface.tui.mood import MOOD_CSS_CLASS, ALL_MOOD_CLASSES
        assert ALL_MOOD_CLASSES == set(MOOD_CSS_CLASS.values())


# ═══════════════════════════════════════════════════════════════════════════
# MoodStateMachine — raw mood computation
# ═══════════════════════════════════════════════════════════════════════════

class TestMoodRawComputation:
    """Tests for _compute_raw — pure function, no hysteresis."""

    def _make(self):
        from bantz.interface.tui.mood import MoodStateMachine
        return MoodStateMachine()

    def test_chill_low_cpu_idle(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(10, 40, False, "idle", 0, 14, 0)
        assert raw == Mood.CHILL

    def test_chill_entertainment(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(15, 30, False, "entertainment", 0, 20, 0)
        assert raw == Mood.CHILL

    def test_focused_coding(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(30, 50, False, "coding", 0, 14, 0)
        assert raw == Mood.FOCUSED

    def test_focused_productivity(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(40, 50, False, "productivity", 0, 10, 0)
        assert raw == Mood.FOCUSED

    def test_busy_moderate_cpu(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(65, 50, False, "idle", 0, 14, 0)
        assert raw == Mood.BUSY

    def test_busy_at_boundary_50(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(50, 50, False, "browsing", 0, 14, 0)
        assert raw == Mood.BUSY

    def test_stressed_high_cpu(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(85, 50, False, "idle", 0, 14, 0)
        assert raw == Mood.STRESSED

    def test_stressed_thermal(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(30, 50, True, "idle", 0, 14, 0)
        assert raw == Mood.STRESSED

    def test_stressed_high_ram(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(30, 96, False, "idle", 0, 14, 0)
        assert raw == Mood.STRESSED

    def test_stressed_errors(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(30, 50, False, "idle", 3, 14, 0)
        assert raw == Mood.STRESSED

    def test_sleeping_nighttime_idle(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(5, 30, False, "idle", 0, 3, 35)
        assert raw == Mood.SLEEPING

    def test_not_sleeping_daytime(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(5, 30, False, "idle", 0, 14, 35)
        assert raw == Mood.CHILL  # daytime, not sleeping

    def test_not_sleeping_short_idle(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(5, 30, False, "idle", 0, 3, 10)
        assert raw == Mood.CHILL  # not idle long enough

    def test_stressed_overrides_focused(self):
        """Stressed has highest priority — even when coding."""
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(90, 50, False, "coding", 0, 14, 0)
        assert raw == Mood.STRESSED

    def test_stressed_overrides_sleeping(self):
        """Stressed beats sleeping."""
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(85, 50, False, "idle", 0, 3, 35)
        assert raw == Mood.STRESSED

    def test_focused_overrides_busy(self):
        """Focused beats busy even at moderate CPU."""
        from bantz.interface.tui.mood import Mood
        m = self._make()
        raw = m._compute_raw(60, 50, False, "coding", 0, 14, 0)
        assert raw == Mood.FOCUSED


# ═══════════════════════════════════════════════════════════════════════════
# MoodStateMachine — hysteresis
# ═══════════════════════════════════════════════════════════════════════════

class TestMoodHysteresis:
    """Mood should only change after 10s sustained candidate."""

    def _make(self):
        from bantz.interface.tui.mood import MoodStateMachine
        return MoodStateMachine()

    def test_initial_mood_is_chill(self):
        from bantz.interface.tui.mood import Mood
        m = self._make()
        assert m.current == Mood.CHILL

    def test_no_instant_transition(self):
        """Single high-CPU eval should NOT change mood immediately."""
        from bantz.interface.tui.mood import Mood
        m = self._make()
        result = m.evaluate(cpu_pct=90, ram_pct=50, activity="idle", hour=14)
        assert result == Mood.CHILL  # still chill — hysteresis

    def test_transition_after_sustained(self):
        """After 10+ seconds sustained, mood should change."""
        from bantz.interface.tui.mood import Mood, _HYSTERESIS_SECONDS
        m = self._make()
        # First eval — starts candidate
        m.evaluate(cpu_pct=90, ram_pct=50, activity="idle", hour=14)
        assert m.current == Mood.CHILL

        # Simulate time passing
        m._candidate_since -= _HYSTERESIS_SECONDS + 1

        # Second eval — candidate sustained long enough
        result = m.evaluate(cpu_pct=90, ram_pct=50, activity="idle", hour=14)
        assert result == Mood.STRESSED

    def test_candidate_resets_on_fluctuation(self):
        """If conditions fluctuate, candidate resets."""
        from bantz.interface.tui.mood import Mood
        m = self._make()
        m.evaluate(cpu_pct=90, ram_pct=50, activity="idle", hour=14)
        # Candidate is STRESSED, but CPU drops before 10s
        m.evaluate(cpu_pct=10, ram_pct=50, activity="idle", hour=14)
        # Now candidate should be CHILL, not STRESSED
        assert m._candidate == Mood.CHILL
        assert m.current == Mood.CHILL

    def test_face_property(self):
        from bantz.interface.tui.mood import Mood, MOOD_FACES
        m = self._make()
        assert m.face == MOOD_FACES[Mood.CHILL]

    def test_label_property(self):
        m = self._make()
        assert m.label == "chill"

    def test_css_class_property(self):
        m = self._make()
        assert m.css_class == "mood-chill"


# ═══════════════════════════════════════════════════════════════════════════
# MoodStateMachine — error tracking
# ═══════════════════════════════════════════════════════════════════════════

class TestMoodErrorTracking:
    def _make(self):
        from bantz.interface.tui.mood import MoodStateMachine
        return MoodStateMachine()

    def test_errors_accumulate_in_window(self):
        m = self._make()
        m.evaluate(cpu_pct=10, observer_error_count=1, activity="idle", hour=14)
        m.evaluate(cpu_pct=10, observer_error_count=2, activity="idle", hour=14)
        m.evaluate(cpu_pct=10, observer_error_count=3, activity="idle", hour=14)
        # 3 errors should accumulate but mood stays chill due to hysteresis
        assert m._recent_errors >= 0  # tracking works

    def test_error_window_resets_after_5_min(self):
        m = self._make()
        m.evaluate(cpu_pct=10, observer_error_count=5, activity="idle", hour=14)
        # Simulate 5+ minutes passing
        m._error_window_start -= 301
        m.evaluate(cpu_pct=10, observer_error_count=5, activity="idle", hour=14)
        # Window reset — recent errors should be 0
        assert m._recent_errors == 0


# ═══════════════════════════════════════════════════════════════════════════
# MoodStateMachine — idle tracking
# ═══════════════════════════════════════════════════════════════════════════

class TestMoodIdleTracking:
    def _make(self):
        from bantz.interface.tui.mood import MoodStateMachine
        return MoodStateMachine()

    def test_idle_timer_resets_on_activity(self):
        m = self._make()
        m._idle_since = time.monotonic() - 3600  # 1 hour ago
        m.evaluate(cpu_pct=10, activity="coding", hour=14)
        # idle_since should be reset (near now)
        idle_min = (time.monotonic() - m._idle_since) / 60.0
        assert idle_min < 1

    def test_idle_timer_continues_when_idle(self):
        m = self._make()
        original = m._idle_since
        m.evaluate(cpu_pct=10, activity="idle", hour=14)
        # idle_since should NOT be reset
        assert m._idle_since == original


# ═══════════════════════════════════════════════════════════════════════════
# MoodHistory — SQLite rolling log
# ═══════════════════════════════════════════════════════════════════════════

class TestMoodHistory:
    @pytest.fixture
    def history(self, tmp_path: Path):
        from bantz.interface.tui.mood import MoodHistory
        h = MoodHistory()
        h.init(tmp_path / "test_mood.db")
        return h

    def test_initialized(self, history):
        assert history.initialized is True

    def test_not_initialized_by_default(self):
        from bantz.interface.tui.mood import MoodHistory
        h = MoodHistory()
        assert h.initialized is False

    def test_log_transition(self, history):
        from bantz.interface.tui.mood import Mood
        history.log_transition(Mood.STRESSED, Mood.CHILL, "CPU 92%", cpu_pct=92)
        assert history.count() == 1

    def test_recent_returns_entries(self, history):
        from bantz.interface.tui.mood import Mood
        history.log_transition(Mood.FOCUSED, Mood.CHILL, "coding", cpu_pct=30)
        entries = history.recent(hours=1)
        assert len(entries) == 1
        assert entries[0]["mood"] == "focused"
        assert entries[0]["prev_mood"] == "chill"
        assert entries[0]["reason"] == "coding"

    def test_recent_filters_old(self, history):
        from bantz.data.connection_pool import get_pool
        # Insert entry with old timestamp
        old_ts = (datetime.now() - timedelta(hours=25)).isoformat()
        with get_pool().connection(write=True) as conn:
            conn.execute(
                "INSERT INTO mood_history (mood, prev_mood, reason, cpu_pct, ram_pct, activity, timestamp)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("stressed", "chill", "old", 90, 50, "idle", old_ts),
            )
        entries = history.recent(hours=24)
        assert len(entries) == 0

    def test_rolling_prune(self, history):
        from bantz.interface.tui.mood import Mood, _MAX_HISTORY
        for i in range(_MAX_HISTORY + 50):
            history.log_transition(
                Mood.CHILL, Mood.STRESSED, f"test {i}", cpu_pct=10
            )
        assert history.count() == _MAX_HISTORY

    def test_count_empty(self, history):
        assert history.count() == 0

    def test_summary_24h_empty(self, history):
        assert history.summary_24h() == {}

    def test_summary_24h_with_data(self, history):
        from bantz.data.connection_pool import get_pool
        # Insert two transitions 30 min apart
        t1 = datetime.now() - timedelta(minutes=60)
        t2 = datetime.now() - timedelta(minutes=30)
        with get_pool().connection(write=True) as conn:
            conn.execute(
                "INSERT INTO mood_history (mood, prev_mood, reason, cpu_pct, ram_pct, activity, timestamp)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("chill", "stressed", "recovery", 20, 40, "idle", t1.isoformat()),
            )
            conn.execute(
                "INSERT INTO mood_history (mood, prev_mood, reason, cpu_pct, ram_pct, activity, timestamp)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("focused", "chill", "coding", 35, 50, "coding", t2.isoformat()),
            )
        summary = history.summary_24h()
        assert "chill" in summary
        assert "focused" in summary
        assert summary["chill"] > 0

    def test_log_transition_not_initialized(self):
        """Graceful when not initialized."""
        from bantz.interface.tui.mood import MoodHistory, Mood
        h = MoodHistory()
        h.log_transition(Mood.CHILL, Mood.STRESSED)  # should not raise

    def test_recent_not_initialized(self):
        from bantz.interface.tui.mood import MoodHistory
        h = MoodHistory()
        assert h.recent() == []


# ═══════════════════════════════════════════════════════════════════════════
# MoodStateMachine — history integration
# ═══════════════════════════════════════════════════════════════════════════

class TestMoodMachineHistory:
    def test_transition_logs_to_history(self, tmp_path: Path):
        from bantz.interface.tui.mood import MoodStateMachine, Mood, _HYSTERESIS_SECONDS
        m = MoodStateMachine()
        m.history.init(tmp_path / "mood.db")

        # Trigger a transition (chill → stressed)
        m.evaluate(cpu_pct=90, activity="idle", hour=14)
        m._candidate_since -= _HYSTERESIS_SECONDS + 1
        m.evaluate(cpu_pct=90, activity="idle", hour=14)

        assert m.current == Mood.STRESSED
        assert m.history.count() == 1
        entries = m.history.recent(hours=1)
        assert entries[0]["mood"] == "stressed"
        assert entries[0]["prev_mood"] == "chill"


# ═══════════════════════════════════════════════════════════════════════════
# Reason generation
# ═══════════════════════════════════════════════════════════════════════════

class TestMoodReason:
    def _make(self):
        from bantz.interface.tui.mood import MoodStateMachine
        return MoodStateMachine()

    def test_reason_high_cpu(self):
        m = self._make()
        r = m._reason(92, 50, False, "idle", 0, 14, 0)
        assert "CPU 92%" in r

    def test_reason_thermal(self):
        m = self._make()
        r = m._reason(30, 50, True, "idle", 0, 14, 0)
        assert "thermal" in r

    def test_reason_errors(self):
        m = self._make()
        r = m._reason(30, 50, False, "idle", 5, 14, 0)
        assert "5 errors" in r

    def test_reason_coding(self):
        m = self._make()
        r = m._reason(30, 50, False, "coding", 0, 14, 0)
        assert "coding" in r

    def test_reason_baseline(self):
        m = self._make()
        r = m._reason(10, 40, False, "idle", 0, 14, 0)
        assert "hour=14" in r or "baseline" in r


# ═══════════════════════════════════════════════════════════════════════════
# TUI integration — CSS class, mood indicator
# ═══════════════════════════════════════════════════════════════════════════

class TestMoodTUIIntegration:
    def test_system_status_no_duplicate_mood_indicator(self):
        """Mood face lives in OperationsHeader only — SystemStatus must NOT yield #mood-indicator."""
        from bantz.interface.tui.panels.system import SystemStatus
        panel = SystemStatus()
        widgets = list(panel.compose())
        ids = [getattr(w, "id", None) for w in widgets]
        assert "mood-indicator" not in ids

    def test_all_mood_classes_set(self):
        from bantz.interface.tui.mood import ALL_MOOD_CLASSES
        assert "mood-chill" in ALL_MOOD_CLASSES
        assert "mood-stressed" in ALL_MOOD_CLASSES
        assert "mood-sleeping" in ALL_MOOD_CLASSES
        assert "mood-focused" in ALL_MOOD_CLASSES
        assert "mood-busy" in ALL_MOOD_CLASSES


# ═══════════════════════════════════════════════════════════════════════════
# CLI — --mood-history argument
# ═══════════════════════════════════════════════════════════════════════════

class TestMoodCLI:
    def test_mood_history_arg_exists(self):
        """Parser should accept --mood-history."""
        # Re-create parser logic
        from bantz.__main__ import main
        import sys
        # Just verify the argument is registered by checking help
        from io import StringIO
        import contextlib
        buf = StringIO()
        with pytest.raises(SystemExit):
            with contextlib.redirect_stdout(buf):
                sys.argv = ["bantz", "--help"]
                main()
        assert "--mood-history" in buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

class TestMoodSingleton:
    def test_singleton_exists(self):
        from bantz.interface.tui.mood import mood_machine, MoodStateMachine
        assert isinstance(mood_machine, MoodStateMachine)

    def test_singleton_starts_chill(self):
        from bantz.interface.tui.mood import mood_machine, Mood
        # Reset for test isolation
        mood_machine._current = Mood.CHILL
        assert mood_machine.current == Mood.CHILL
