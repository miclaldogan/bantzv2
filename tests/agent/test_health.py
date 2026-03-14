"""
Tests for bantz.agent.health — Proactive Health & Break Interventions (#168).

Covers:
  ✓ SessionTracker: active time accumulation, idle reset (Ghost Session fix)
  ✓ ThermalHistory: sustained-temperature check (Thermal Panic fix)
  ✓ get_idle_ms / is_screen_locked stubs
  ✓ HealthRuleEvaluator: all 5 rules
  ✓ Cooldown management
  ✓ RL break reward (False Positive fix)
  ✓ Config-disabled bypass
  ✓ Intervention push
  ✓ Brain route matching
  ✓ Integration: Action enum, InterventionType enum, config fields
"""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# SessionTracker
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionTracker:
    """Ghost Session Trap fix: session_duration from OS idle, not uptime."""

    def test_initial_state(self):
        from bantz.agent.health import SessionTracker
        st = SessionTracker(idle_threshold_s=900)
        assert st.active_hours == 0.0
        assert not st.had_recent_break

    def test_active_accumulation(self):
        """Active time should accumulate when user is present."""
        from bantz.agent.health import SessionTracker
        st = SessionTracker(idle_threshold_s=900)

        # Simulate get_idle_ms returning 0 (user active)
        with patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            # Move monotonic forward by 1h
            t0 = time.monotonic()
            st._last_tick = t0 - 3600
            st.tick()
            # Should have accumulated ~1h
            assert st.active_hours >= 0.9

    def test_idle_pauses_session(self):
        """Session clock should NOT run during idle periods (Ghost Session fix)."""
        from bantz.agent.health import SessionTracker
        st = SessionTracker(idle_threshold_s=900)

        with patch("bantz.agent.health.get_idle_ms", return_value=1_000_000), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            st._last_tick = time.monotonic() - 3600
            st.tick()
            # Should NOT have accumulated time — user was idle
            assert st.active_hours < 0.1

    def test_screen_lock_resets_session(self):
        """Screen lock → genuine break → reset session."""
        from bantz.agent.health import SessionTracker
        st = SessionTracker(idle_threshold_s=900)
        st._active_seconds = 7200  # 2h

        with patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=True):
            st.tick()
            # Session should be reset
            assert st.active_hours == 0.0
            assert st.had_recent_break

    def test_consume_break_flag_one_shot(self):
        """Break flag is consumed once — single RL reward."""
        from bantz.agent.health import SessionTracker
        st = SessionTracker()
        st._break_detected = True
        assert st.consume_break_flag() is True
        assert st.consume_break_flag() is False

    def test_15min_idle_resets(self):
        """Coming back from 15min+ idle counts as break."""
        from bantz.agent.health import SessionTracker
        st = SessionTracker(idle_threshold_s=900)
        st._active_seconds = 7200

        # First: go idle
        with patch("bantz.agent.health.get_idle_ms", return_value=1_000_000), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            st.tick()
            assert st._was_idle is True

        # Then: come back from idle
        with patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            st.tick()
            # Should have registered a break
            assert st.had_recent_break
            assert st.active_hours == 0.0

    def test_reset(self):
        from bantz.agent.health import SessionTracker
        st = SessionTracker()
        st._active_seconds = 7200
        st._break_detected = True
        st.reset()
        assert st.active_hours == 0.0
        assert not st.had_recent_break


# ═══════════════════════════════════════════════════════════════════════════
# ThermalHistory
# ═══════════════════════════════════════════════════════════════════════════

class TestThermalHistory:
    """Thermal Panic fix: require 3 consecutive readings above threshold."""

    def test_single_spike_no_fire(self):
        """One spike should NOT trigger thermal stress."""
        from bantz.agent.health import ThermalHistory
        th = ThermalHistory(required_count=3)
        assert th.record(90, 60, 85, 80) is False  # 1st above
        assert th.cpu_streak == 1

    def test_two_spikes_no_fire(self):
        from bantz.agent.health import ThermalHistory
        th = ThermalHistory(required_count=3)
        th.record(90, 60, 85, 80)
        assert th.record(88, 60, 85, 80) is False  # 2nd
        assert th.cpu_streak == 2

    def test_three_consecutive_fires(self):
        """3 consecutive above-threshold readings → fire."""
        from bantz.agent.health import ThermalHistory
        th = ThermalHistory(required_count=3)
        th.record(90, 60, 85, 80)
        th.record(88, 60, 85, 80)
        assert th.record(92, 60, 85, 80) is True
        assert th.cpu_streak == 3

    def test_spike_then_drop_resets(self):
        """Drop below threshold resets streak."""
        from bantz.agent.health import ThermalHistory
        th = ThermalHistory(required_count=3)
        th.record(90, 60, 85, 80)
        th.record(88, 60, 85, 80)
        th.record(70, 60, 85, 80)  # drops below → reset
        assert th.cpu_streak == 0
        assert th.record(90, 60, 85, 80) is False  # starts over

    def test_gpu_sustained(self):
        """GPU streak also works independently."""
        from bantz.agent.health import ThermalHistory
        th = ThermalHistory(required_count=3)
        th.record(60, 85, 85, 80)
        th.record(60, 82, 85, 80)
        assert th.record(60, 90, 85, 80) is True
        assert th.gpu_streak == 3
        assert th.cpu_streak == 0

    def test_reset(self):
        from bantz.agent.health import ThermalHistory
        th = ThermalHistory(required_count=3)
        th.record(90, 85, 85, 80)
        th.reset()
        assert th.cpu_streak == 0
        assert th.gpu_streak == 0


# ═══════════════════════════════════════════════════════════════════════════
# get_idle_ms / is_screen_locked
# ═══════════════════════════════════════════════════════════════════════════

class TestIdleDetection:
    def test_get_idle_ms_xprintidle(self):
        from bantz.agent.health import get_idle_ms
        with patch("shutil.which", side_effect=lambda cmd: "/usr/bin/xprintidle" if cmd == "xprintidle" else None), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="42000\n")
            assert get_idle_ms() == 42000

    def test_get_idle_ms_fallback_returns_zero(self):
        from bantz.agent.health import get_idle_ms
        with patch("shutil.which", return_value=None):
            assert get_idle_ms() == 0

    def test_is_screen_locked_true(self):
        from bantz.agent.health import is_screen_locked
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="LockedHint=yes\n",
            )
            assert is_screen_locked() is True

    def test_is_screen_locked_false(self):
        from bantz.agent.health import is_screen_locked
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="LockedHint=no\n",
            )
            assert is_screen_locked() is False

    def test_is_screen_locked_error(self):
        from bantz.agent.health import is_screen_locked
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert is_screen_locked() is False


# ═══════════════════════════════════════════════════════════════════════════
# HealthRuleEvaluator
# ═══════════════════════════════════════════════════════════════════════════

def _make_engine():
    """Create a fresh evaluator with mocked dependencies."""
    from bantz.agent.health import HealthRuleEvaluator
    engine = HealthRuleEvaluator()
    engine.init()
    return engine


def _mock_gather(hour=10, cpu_pct=50, cpu_temp=60, gpu_temp=50,
                 activity="coding", ambient="silence",
                 active_hours=0.5, had_break=False):
    """Return a dict matching HealthRuleEvaluator._gather() schema."""
    return {
        "hour": hour,
        "segment": "late_night" if hour < 5 else "morning" if hour < 12 else "afternoon",
        "cpu_pct": cpu_pct,
        "cpu_temp": cpu_temp,
        "gpu_temp": gpu_temp,
        "activity": activity,
        "ambient_label": ambient,
        "active_hours": active_hours,
        "had_break": had_break,
    }


class TestLateNightLoad:
    """Rule 1: hour >= 2 (or < 5), CPU > 80%, activity == coding."""

    def test_triggers_at_3am_high_cpu(self):
        engine = _make_engine()
        ctx = _mock_gather(hour=3, cpu_pct=90, activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "late_night_load"]
            assert len(r) == 1
            assert r[0].fired is True

    def test_no_trigger_during_day(self):
        engine = _make_engine()
        ctx = _mock_gather(hour=14, cpu_pct=90, activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "late_night_load"]
            assert r[0].fired is False

    def test_no_trigger_low_cpu(self):
        engine = _make_engine()
        ctx = _mock_gather(hour=3, cpu_pct=40, activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "late_night_load"]
            assert r[0].fired is False

    def test_no_trigger_browsing(self):
        engine = _make_engine()
        ctx = _mock_gather(hour=3, cpu_pct=90, activity="browsing")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "late_night_load"]
            assert r[0].fired is False


class TestMarathonSession:
    """Rule 2: active_hours >= 4h, no break taken."""

    def test_triggers_after_4h_no_break(self):
        engine = _make_engine()
        ctx = _mock_gather(active_hours=4.5, had_break=False, activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "marathon_session"]
            assert r[0].fired is True

    def test_no_trigger_with_break(self):
        engine = _make_engine()
        ctx = _mock_gather(active_hours=5.0, had_break=True, activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "marathon_session"]
            assert r[0].fired is False

    def test_no_trigger_short_session(self):
        engine = _make_engine()
        ctx = _mock_gather(active_hours=2.0, had_break=False, activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "marathon_session"]
            assert r[0].fired is False

    def test_no_trigger_when_idle(self):
        """IDLE activity should NOT count — Ghost Session fix."""
        engine = _make_engine()
        ctx = _mock_gather(active_hours=5.0, had_break=False, activity="idle")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "marathon_session"]
            assert r[0].fired is False


class TestEyeStrain:
    """Rule 3: screen time > 2h, user active."""

    def test_triggers_after_2h_screen(self):
        engine = _make_engine()
        # Simulate 2.5h of screen time
        engine._screen_time_start = time.monotonic() - (2.5 * 3600)
        ctx = _mock_gather(activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "eye_strain"]
            assert r[0].fired is True

    def test_no_trigger_short_screen(self):
        engine = _make_engine()
        engine._screen_time_start = time.monotonic() - (1.0 * 3600)
        ctx = _mock_gather(activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "eye_strain"]
            assert r[0].fired is False


class TestThermalStress:
    """Rule 4: sustained thermal > threshold (3 consecutive readings)."""

    def test_triggers_after_3_consecutive(self):
        engine = _make_engine()
        # Pre-load 2 readings
        engine._thermal.record(90, 60, 85, 80)
        engine._thermal.record(88, 60, 85, 80)

        ctx = _mock_gather(cpu_temp=92, gpu_temp=60)
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "thermal_stress"]
            assert r[0].fired is True

    def test_spike_then_drop_no_fire(self):
        """Thermal Panic fix: spike→drop→spike should NOT fire."""
        engine = _make_engine()
        engine._thermal.record(90, 60, 85, 80)
        engine._thermal.record(70, 60, 85, 80)  # drop → reset

        ctx = _mock_gather(cpu_temp=90, gpu_temp=60)
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "thermal_stress"]
            assert r[0].fired is False

    def test_single_reading_no_fire(self):
        engine = _make_engine()
        ctx = _mock_gather(cpu_temp=90, gpu_temp=60)
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "thermal_stress"]
            assert r[0].fired is False


class TestLateNightMusic:
    """Rule 5: late night + ambient noisy/speech + coding."""

    def test_triggers(self):
        engine = _make_engine()
        ctx = _mock_gather(hour=3, ambient="noisy", activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "late_night_music"]
            assert r[0].fired is True

    def test_no_trigger_silence(self):
        engine = _make_engine()
        ctx = _mock_gather(hour=3, ambient="silence", activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "late_night_music"]
            assert r[0].fired is False

    def test_no_trigger_daytime(self):
        engine = _make_engine()
        ctx = _mock_gather(hour=14, ambient="noisy", activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r = [x for x in results if x.rule_id.value == "late_night_music"]
            assert r[0].fired is False


# ═══════════════════════════════════════════════════════════════════════════
# Cooldowns
# ═══════════════════════════════════════════════════════════════════════════

class TestCooldowns:
    """Per-rule cooldowns prevent spam."""

    def test_cooldown_blocks_repeat(self):
        engine = _make_engine()
        # Fire late_night_load once
        ctx = _mock_gather(hour=3, cpu_pct=90, activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            results = engine.evaluate_all()
            r1 = [x for x in results if x.rule_id.value == "late_night_load"]
            assert r1[0].fired is True

            # Immediately run again → cooldown should block
            results2 = engine.evaluate_all()
            r2 = [x for x in results2 if x.rule_id.value == "late_night_load"]
            assert r2[0].fired is False

    def test_cooldown_remaining(self):
        from bantz.agent.health import RuleID
        engine = _make_engine()
        # Not yet fired → remaining should be 0
        assert engine.cooldown_remaining(RuleID.LATE_NIGHT_LOAD) == 0.0

        # Mark as fired
        engine._mark_fired(RuleID.LATE_NIGHT_LOAD)
        remaining = engine.cooldown_remaining(RuleID.LATE_NIGHT_LOAD)
        assert remaining > 7000  # ~7200s cooldown


# ═══════════════════════════════════════════════════════════════════════════
# RL Break Reward — Senior Fix #2
# ═══════════════════════════════════════════════════════════════════════════

class TestBreakReward:
    """Only reward on screen lock — not just mouse idle."""

    def test_no_reward_just_idle(self):
        """Mouse idle (YouTube watching) should NOT get RL reward."""
        engine = _make_engine()
        with patch("bantz.agent.health.is_screen_locked", return_value=False):
            assert engine.check_break_reward() is False

    def test_reward_on_screen_lock(self):
        """Screen lock = genuine break → RL reward."""
        engine = _make_engine()
        with patch("bantz.agent.health.is_screen_locked", return_value=True):
            assert engine.check_break_reward() is True
            # One-shot: second call should be False
            with patch("bantz.agent.health.is_screen_locked", return_value=False):
                assert engine.check_break_reward() is False


# ═══════════════════════════════════════════════════════════════════════════
# Config disabled
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigDisabled:
    def test_not_initialized_returns_empty(self):
        from bantz.agent.health import HealthRuleEvaluator
        engine = HealthRuleEvaluator()
        # Not initialized → should return empty
        assert engine.evaluate_all() == []


# ═══════════════════════════════════════════════════════════════════════════
# Integration: enums and config
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_intervention_type_health(self):
        from bantz.agent.interventions import InterventionType
        assert InterventionType.HEALTH == "health"

    def test_action_health_break(self):
        from bantz.agent.interventions import ACTION_LABELS
        assert "health_break" in ACTION_LABELS

    def test_action_labels_has_health_break(self):
        from bantz.agent.interventions import ACTION_LABELS
        assert "health_break" in ACTION_LABELS
        emoji, label = ACTION_LABELS["health_break"]
        assert "🏥" in emoji

    def test_config_fields(self):
        from bantz.config import config
        assert hasattr(config, "health_enabled")
        assert hasattr(config, "health_check_interval")
        assert hasattr(config, "health_late_hour")
        assert hasattr(config, "health_session_max_hours")
        assert hasattr(config, "health_thermal_cpu")
        assert hasattr(config, "health_thermal_gpu")
        assert hasattr(config, "health_eye_strain_hours")

    def test_config_defaults(self):
        from bantz.config import config
        assert isinstance(config.health_enabled, bool)
        assert config.health_check_interval == 300
        assert config.health_late_hour == 2
        assert config.health_session_max_hours == 4.0
        assert config.health_thermal_cpu == 85.0
        assert config.health_thermal_gpu == 80.0


# ═══════════════════════════════════════════════════════════════════════════
# Brain route
# ═══════════════════════════════════════════════════════════════════════════

class TestBrainRoute:
    """Regex in _quick_route should match health status queries."""

    def _match(self, text: str) -> bool:
        import re
        pattern = (
            r"health\s+(?:status|info|stats|check)|"
            r"break\s+(?:status|timer|count)|"
            r"sa[ğg]l[ıi]k\s+durum|session\s+(?:time|timer|hours)"
        )
        return bool(re.search(pattern, text.lower()))

    def test_health_status(self):
        assert self._match("health status")

    def test_health_check(self):
        assert self._match("health check")

    def test_break_status(self):
        assert self._match("break status")

    def test_session_time(self):
        assert self._match("session time")

    def test_turkish(self):
        assert self._match("sağlık durum")

    def test_no_match_random(self):
        assert not self._match("open spotify")
        assert not self._match("what's the weather")


# ═══════════════════════════════════════════════════════════════════════════
# Status line
# ═══════════════════════════════════════════════════════════════════════════

class TestStatus:
    def test_status_dict(self):
        engine = _make_engine()
        s = engine.status()
        assert "initialized" in s
        assert s["initialized"] is True
        assert "active_hours" in s
        assert "cooldowns" in s

    def test_status_line(self):
        engine = _make_engine()
        line = engine.status_line()
        assert "active=" in line
        assert "break=" in line


# ═══════════════════════════════════════════════════════════════════════════
# RuleResult data
# ═══════════════════════════════════════════════════════════════════════════

class TestRuleResult:
    def test_fields(self):
        from bantz.agent.health import RuleResult, RuleID
        r = RuleResult(
            rule_id=RuleID.LATE_NIGHT_LOAD,
            fired=True,
            title="Test",
            reason="Test reason",
            data={"key": "value"},
        )
        assert r.fired is True
        assert r.data["key"] == "value"

    def test_rule_id_enum(self):
        from bantz.agent.health import RuleID
        assert len(RuleID) == 5
        assert RuleID.LATE_NIGHT_LOAD.value == "late_night_load"
        assert RuleID.MARATHON_SESSION.value == "marathon_session"
        assert RuleID.EYE_STRAIN.value == "eye_strain"
        assert RuleID.THERMAL_STRESS.value == "thermal_stress"
        assert RuleID.LATE_NIGHT_MUSIC.value == "late_night_music"


# ═══════════════════════════════════════════════════════════════════════════
# Push intervention with mock
# ═══════════════════════════════════════════════════════════════════════════

class TestPushIntervention:
    def test_push_succeeds(self):
        engine = _make_engine()
        from bantz.agent.health import RuleResult, RuleID
        r = RuleResult(
            rule_id=RuleID.LATE_NIGHT_LOAD,
            fired=True,
            title="Test",
            reason="Test reason",
        )
        with patch("bantz.agent.interventions.intervention_queue") as mock_q:
            mock_q.push.return_value = True
            assert engine.push_intervention(r) is True

    def test_push_handles_error(self):
        engine = _make_engine()
        from bantz.agent.health import RuleResult, RuleID
        r = RuleResult(
            rule_id=RuleID.LATE_NIGHT_LOAD,
            fired=True,
            title="Test",
            reason="Test reason",
        )
        with patch("bantz.agent.interventions.intervention_queue") as mock_q:
            mock_q.push.side_effect = RuntimeError("boom")
            assert engine.push_intervention(r) is False


# ═══════════════════════════════════════════════════════════════════════════
# Job health check integration
# ═══════════════════════════════════════════════════════════════════════════

class TestJobHealthCheck:
    @pytest.mark.asyncio
    async def test_job_runs_evaluation(self):
        from bantz.agent.job_scheduler import _job_health_check
        from bantz.agent.health import health_engine, RuleResult, RuleID

        health_engine.init()
        ctx = _mock_gather(hour=10, cpu_pct=30)
        with patch.object(health_engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False):
            await _job_health_check()
            # Should complete without error; no rules should fire

    @pytest.mark.asyncio
    async def test_job_skips_uninitialized(self):
        from bantz.agent.job_scheduler import _job_health_check
        from bantz.agent.health import HealthRuleEvaluator

        with patch("bantz.agent.health.health_engine") as mock_engine:
            mock_engine.initialized = False
            await _job_health_check()
            mock_engine.evaluate_all.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# EventBus integration (Sprint 3 Part 2)
# ═══════════════════════════════════════════════════════════════════════════

class TestHealthEventBus:
    """Verify health_alert events are emitted via the EventBus."""

    def test_fired_rule_emits_health_alert(self):
        """When a rule fires, bus.emit_threadsafe('health_alert') is called."""
        engine = _make_engine()
        ctx = _mock_gather(hour=3, cpu_pct=90, activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False), \
             patch("bantz.agent.health.bus") as mock_bus:
            results = engine.evaluate_all()
            fired = [r for r in results if r.fired]
            assert len(fired) >= 1
            mock_bus.emit_threadsafe.assert_called()
            call_args = mock_bus.emit_threadsafe.call_args
            assert call_args[0][0] == "health_alert"
            assert "rule_id" in call_args[1]

    def test_no_event_when_no_rule_fires(self):
        """No bus emission when all rules pass clean."""
        engine = _make_engine()
        ctx = _mock_gather(hour=10, cpu_pct=30, activity="idle")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False), \
             patch("bantz.agent.health.bus") as mock_bus:
            results = engine.evaluate_all()
            mock_bus.emit_threadsafe.assert_not_called()

    def test_telemetry_cache_via_bus(self):
        """_on_telemetry updates the internal cache from bus events."""
        from bantz.core.event_bus import Event
        engine = _make_engine()
        event = Event(name="telemetry_update", data={
            "cpu_pct": 87.5, "cpu_temp": 72.0, "gpu_temp": 65.0,
        })
        engine._on_telemetry(event)
        assert engine._telemetry["cpu_pct"] == 87.5
        assert engine._telemetry["cpu_temp"] == 72.0
        assert engine._telemetry["gpu_temp"] == 65.0

    def test_no_tui_telemetry_import(self):
        """health.py must NOT import from bantz.interface.tui."""
        import ast, inspect
        from bantz.agent import health
        tree = ast.parse(inspect.getsource(health))
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        for imp in imports:
            if imp.module:
                assert "bantz.interface.tui" not in imp.module, \
                    f"Forbidden TUI import: {imp.module}"

    def test_no_brain_import(self):
        """health.py must NOT import from bantz.core.brain."""
        import ast, inspect
        from bantz.agent import health
        tree = ast.parse(inspect.getsource(health))
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        for imp in imports:
            if imp.module:
                assert "bantz.core.brain" not in imp.module, \
                    f"Forbidden brain import: {imp.module}"

    def test_bus_exception_does_not_crash_evaluate(self):
        """If bus.emit_threadsafe raises, evaluate_all still returns results."""
        engine = _make_engine()
        ctx = _mock_gather(hour=3, cpu_pct=90, activity="coding")
        with patch.object(engine, "_gather", return_value=ctx), \
             patch("bantz.agent.health.get_idle_ms", return_value=0), \
             patch("bantz.agent.health.is_screen_locked", return_value=False), \
             patch("bantz.agent.health.bus") as mock_bus:
            mock_bus.emit_threadsafe.side_effect = RuntimeError("bus down")
            results = engine.evaluate_all()
            assert len(results) == 5  # all 5 rules evaluated
            fired = [r for r in results if r.fired]
            assert len(fired) >= 1  # rule still fired despite bus error

    def test_init_subscribes_to_telemetry(self):
        """init() should register _on_telemetry on the bus."""
        with patch("bantz.agent.health.bus") as mock_bus:
            from bantz.agent.health import HealthRuleEvaluator
            engine = HealthRuleEvaluator()
            engine.init()
            mock_bus.on.assert_called_once_with(
                "telemetry_update", engine._on_telemetry,
            )
