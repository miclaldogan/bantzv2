"""
Tests for bantz.personality.persona — Dynamic Persona Adaptation (#169).

Covers:
  ✓ PersonaStateBuilder.build() for all 6 states + NEUTRAL fallback
  ✓ Priority: STRAINED > FOCUSED > BONDED > time-based
  ✓ Config disabled → empty string fallback
  ✓ Build speed < 5ms
  ✓ 1920s butler character in all persona prompts
  ✓ system_prompt.py templates include {persona_state}
  ✓ brain.py CHAT_SYSTEM includes {persona_state}
  ✓ finalizer.py FINALIZER_SYSTEM includes {persona_state}
  ✓ Config field existence
  ✓ AI glitch rule character reference in prompts
"""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_builder():
    from bantz.personality.persona import PersonaStateBuilder
    return PersonaStateBuilder()


def _mock_gather(hour=12, cpu_pct=30, ram_pct=40, thermal_alert=False,
                 activity="idle", rl_avg_reward=0.0):
    return {
        "hour": hour,
        "cpu_pct": cpu_pct,
        "ram_pct": ram_pct,
        "thermal_alert": thermal_alert,
        "activity": activity,
        "rl_avg_reward": rl_avg_reward,
    }


# ═══════════════════════════════════════════════════════════════════════════
# State resolution
# ═══════════════════════════════════════════════════════════════════════════

class TestPersonaEnergeticMorning:
    """8 AM + low CPU → energetic."""

    def test_morning_low_cpu(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=8, cpu_pct=20, activity="idle")
        assert b._resolve(ctx) == PersonaState.ENERGETIC

    def test_morning_high_cpu_not_energetic(self):
        """Morning but CPU > 30% → not energetic (might be focused or neutral)."""
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=8, cpu_pct=50, activity="idle")
        # CPU > 30, not coding, not strained → NEUTRAL
        assert b._resolve(ctx) != PersonaState.ENERGETIC

    def test_afternoon_not_energetic(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=14, cpu_pct=20, activity="idle")
        assert b._resolve(ctx) != PersonaState.ENERGETIC


class TestPersonaFocusedDuringCoding:
    """AppDetector=CODING → concise/technical prompt."""

    def test_coding_focused(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(activity="coding")
        assert b._resolve(ctx) == PersonaState.FOCUSED

    def test_productivity_focused(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(activity="productivity")
        assert b._resolve(ctx) == PersonaState.FOCUSED


class TestPersonaStrainedHighCpu:
    """CPU > 85% → "running hot" acknowledgment."""

    def test_high_cpu(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(cpu_pct=92)
        assert b._resolve(ctx) == PersonaState.STRAINED

    def test_high_ram(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(ram_pct=95)
        assert b._resolve(ctx) == PersonaState.STRAINED

    def test_thermal_alert(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(thermal_alert=True)
        assert b._resolve(ctx) == PersonaState.STRAINED


class TestPersonaSleepyLateNight:
    """2 AM + low activity → "late, suggest wrapping up"."""

    def test_late_night_idle(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=2, activity="idle")
        assert b._resolve(ctx) == PersonaState.SLEEPY

    def test_late_night_browsing(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=23, activity="browsing")
        assert b._resolve(ctx) == PersonaState.SLEEPY

    def test_midnight(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=0, activity="idle")
        assert b._resolve(ctx) == PersonaState.SLEEPY


class TestPersonaRelaxedEvening:
    """Evening (18-22) + idle/browsing → relaxed."""

    def test_evening_idle(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=20, activity="idle")
        assert b._resolve(ctx) == PersonaState.RELAXED

    def test_evening_entertainment(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=19, activity="entertainment")
        assert b._resolve(ctx) == PersonaState.RELAXED

    def test_evening_coding_not_relaxed(self):
        """Evening but coding → FOCUSED wins over RELAXED."""
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=20, activity="coding")
        assert b._resolve(ctx) == PersonaState.FOCUSED


class TestPersonaBonded:
    """High RL reward → bonded tone."""

    def test_high_rl_reward(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(rl_avg_reward=2.0, activity="browsing")
        assert b._resolve(ctx) == PersonaState.BONDED

    def test_low_rl_reward_not_bonded(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(rl_avg_reward=0.5, activity="browsing")
        assert b._resolve(ctx) != PersonaState.BONDED


class TestPersonaNeutral:
    """Default: no state triggers → neutral."""

    def test_neutral_fallback(self):
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=14, cpu_pct=50, activity="communication")
        assert b._resolve(ctx) == PersonaState.NEUTRAL


# ═══════════════════════════════════════════════════════════════════════════
# Priority: STRAINED > FOCUSED > BONDED > time-based
# ═══════════════════════════════════════════════════════════════════════════

class TestPersonaPriority:
    def test_strained_over_energetic(self):
        """CPU=90% at 8 AM → STRAINED wins over ENERGETIC."""
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=8, cpu_pct=90, activity="idle")
        assert b._resolve(ctx) == PersonaState.STRAINED

    def test_strained_over_focused(self):
        """CPU=92% while coding → STRAINED wins over FOCUSED."""
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(cpu_pct=92, activity="coding")
        assert b._resolve(ctx) == PersonaState.STRAINED

    def test_focused_over_bonded(self):
        """Coding + high reward → FOCUSED wins over BONDED."""
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(activity="coding", rl_avg_reward=3.0)
        assert b._resolve(ctx) == PersonaState.FOCUSED

    def test_bonded_over_relaxed(self):
        """Evening + high reward + idle → BONDED wins over RELAXED."""
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=20, activity="idle", rl_avg_reward=2.0)
        assert b._resolve(ctx) == PersonaState.BONDED

    def test_focused_over_sleepy(self):
        """2 AM coding → FOCUSED wins over SLEEPY."""
        from bantz.personality.persona import PersonaState
        b = _make_builder()
        ctx = _mock_gather(hour=2, activity="coding")
        assert b._resolve(ctx) == PersonaState.FOCUSED


# ═══════════════════════════════════════════════════════════════════════════
# Config disabled → fallback to empty string
# ═══════════════════════════════════════════════════════════════════════════

class TestPersonaDisabledFallback:
    def test_disabled_returns_empty(self):
        b = _make_builder()
        with patch("bantz.config.config") as mock_cfg:
            mock_cfg.persona_enabled = False
            result = b.build()
            assert result == ""

    def test_enabled_returns_nonempty_for_strained(self):
        b = _make_builder()
        ctx = _mock_gather(cpu_pct=92)
        with patch.object(b, "_gather", return_value=ctx):
            result = b.build()
            assert len(result) > 0
            assert "machine" in result.lower() or "heat" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Performance: build() < 5ms
# ═══════════════════════════════════════════════════════════════════════════

class TestPersonaPerformance:
    def test_build_under_5ms(self):
        b = _make_builder()
        ctx = _mock_gather(hour=8, cpu_pct=20, activity="idle")
        with patch.object(b, "_gather", return_value=ctx):
            start = time.monotonic()
            for _ in range(100):
                b.build()
            elapsed_ms = (time.monotonic() - start) * 1000 / 100
            assert elapsed_ms < 5, f"build() took {elapsed_ms:.2f}ms (limit: 5ms)"


# ═══════════════════════════════════════════════════════════════════════════
# 1920s butler character in all persona prompts
# ═══════════════════════════════════════════════════════════════════════════

class TestButlerCharacterPrompts:
    """All persona prompts should match the 1920s butler narrative."""

    def test_strained_mentions_machines(self):
        from bantz.personality.persona import PERSONA_PROMPTS, PersonaState
        text = PERSONA_PROMPTS[PersonaState.STRAINED]
        assert "machine" in text.lower()
        assert "heat" in text.lower()

    def test_energetic_mentions_morning(self):
        from bantz.personality.persona import PERSONA_PROMPTS, PersonaState
        text = PERSONA_PROMPTS[PersonaState.ENERGETIC]
        assert "morning" in text.lower()
        assert "contraption" in text.lower() or "noise" in text.lower()

    def test_sleepy_mentions_maam(self):
        from bantz.personality.persona import PERSONA_PROMPTS, PersonaState
        text = PERSONA_PROMPTS[PersonaState.SLEEPY]
        assert "ma'am" in text.lower() or "maam" in text.lower()

    def test_focused_mentions_composure(self):
        from bantz.personality.persona import PERSONA_PROMPTS, PersonaState
        text = PERSONA_PROMPTS[PersonaState.FOCUSED]
        assert "composure" in text.lower() or "concise" in text.lower()

    def test_relaxed_mentions_aristocratic(self):
        from bantz.personality.persona import PERSONA_PROMPTS, PersonaState
        text = PERSONA_PROMPTS[PersonaState.RELAXED]
        assert "aristocratic" in text.lower()

    def test_bonded_no_flattery(self):
        from bantz.personality.persona import PERSONA_PROMPTS, PersonaState
        text = PERSONA_PROMPTS[PersonaState.BONDED]
        assert "flattery" in text.lower()

    def test_neutral_is_empty(self):
        from bantz.personality.persona import PERSONA_PROMPTS, PersonaState
        assert PERSONA_PROMPTS[PersonaState.NEUTRAL] == ""


# ═══════════════════════════════════════════════════════════════════════════
# Template integration — {persona_state} placeholder exists
# ═══════════════════════════════════════════════════════════════════════════

class TestTemplateIntegration:
    def test_brain_chat_system_has_persona_state(self):
        from bantz.core.brain import CHAT_SYSTEM
        assert "{persona_state}" in CHAT_SYSTEM

    def test_finalizer_system_has_persona_state(self):
        from bantz.core.finalizer import FINALIZER_SYSTEM
        assert "{persona_state}" in FINALIZER_SYSTEM

    def test_system_prompt_chat_has_persona_state(self):
        from bantz.personality.system_prompt import BANTZ_CHAT
        assert "{persona_state}" in BANTZ_CHAT

    def test_system_prompt_finalizer_has_persona_state(self):
        from bantz.personality.system_prompt import BANTZ_FINALIZER
        assert "{persona_state}" in BANTZ_FINALIZER

    def test_chat_system_format_accepts_persona_state(self):
        from bantz.core.brain import CHAT_SYSTEM
        result = CHAT_SYSTEM.format(
            persona_state="TEST_PERSONA",
            style_hint="", time_hint="", profile_hint="",
            graph_hint="", vector_hint="", desktop_hint="",
            deep_memory="",
        )
        assert "TEST_PERSONA" in result

    def test_get_chat_system_accepts_persona_state(self):
        from bantz.personality.system_prompt import get_chat_system
        result = get_chat_system(persona_state="TEST_STATE")
        assert "TEST_STATE" in result

    def test_get_finalizer_system_accepts_persona_state(self):
        from bantz.personality.system_prompt import get_finalizer_system
        result = get_finalizer_system(persona_state="TEST_STATE")
        assert "TEST_STATE" in result


# ═══════════════════════════════════════════════════════════════════════════
# AI Glitch Rule — prompts mention the "not AI" instruction
# ═══════════════════════════════════════════════════════════════════════════

class TestAIGlitchRule:
    def test_chat_system_has_ai_glitch_instruction(self):
        from bantz.core.brain import CHAT_SYSTEM
        assert "not an ai" in CHAT_SYSTEM.lower()
        assert "..." in CHAT_SYSTEM

    def test_identity_has_ai_glitch(self):
        from bantz.personality.system_prompt import BANTZ_IDENTITY
        assert "not an ai" in BANTZ_IDENTITY.lower()

    def test_bantz_chat_has_ai_glitch(self):
        from bantz.personality.system_prompt import BANTZ_CHAT
        assert "not an ai" in BANTZ_CHAT.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 1920s butler identity in runtime prompts
# ═══════════════════════════════════════════════════════════════════════════

class TestButlerIdentityInPrompts:
    def test_brain_chat_mentions_1920s(self):
        from bantz.core.brain import CHAT_SYSTEM
        assert "1920s" in CHAT_SYSTEM

    def test_brain_chat_mentions_maam(self):
        from bantz.core.brain import CHAT_SYSTEM
        assert "ma'am" in CHAT_SYSTEM

    def test_brain_chat_mentions_sarcasm(self):
        from bantz.core.brain import CHAT_SYSTEM
        assert "sarcasm" in CHAT_SYSTEM.lower()

    def test_finalizer_mentions_butler(self):
        from bantz.core.finalizer import FINALIZER_SYSTEM
        assert "1920s" in FINALIZER_SYSTEM

    def test_finalizer_mentions_maam(self):
        from bantz.core.finalizer import FINALIZER_SYSTEM
        assert "ma'am" in FINALIZER_SYSTEM


# ═══════════════════════════════════════════════════════════════════════════
# Config field
# ═══════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_persona_enabled_exists(self):
        from bantz.config import config
        assert hasattr(config, "persona_enabled")

    def test_persona_enabled_default_true(self):
        from bantz.config import config
        assert config.persona_enabled is True


# ═══════════════════════════════════════════════════════════════════════════
# Status
# ═══════════════════════════════════════════════════════════════════════════

class TestPersonaStatus:
    def test_status_dict(self):
        b = _make_builder()
        s = b.status()
        assert "state" in s
        assert s["state"] == "neutral"

    def test_status_after_build(self):
        b = _make_builder()
        ctx = _mock_gather(cpu_pct=92)
        with patch.object(b, "_gather", return_value=ctx):
            b.build()
        s = b.status()
        assert s["state"] == "strained"


# ═══════════════════════════════════════════════════════════════════════════
# PersonaState enum
# ═══════════════════════════════════════════════════════════════════════════

class TestPersonaStateEnum:
    def test_state_count(self):
        from bantz.personality.persona import PersonaState
        assert len(PersonaState) == 7

    def test_all_states_have_prompts(self):
        from bantz.personality.persona import PersonaState, PERSONA_PROMPTS
        for state in PersonaState:
            assert state in PERSONA_PROMPTS


# ═══════════════════════════════════════════════════════════════════════════
# Grand Telegraph Archives lore
# ═══════════════════════════════════════════════════════════════════════════

class TestGrandTelegraphArchives:
    """Butler must believe he can research via 'Grand Telegraph Archives'."""

    def test_chat_system_has_telegraph_lore(self):
        """brain.py CHAT_SYSTEM must mention Grand Telegraph Archives."""
        from bantz.core.brain import CHAT_SYSTEM
        assert "grand telegraph archives" in CHAT_SYSTEM.lower()

    def test_chat_system_no_refusal_rule(self):
        """CHAT_SYSTEM must NOT say 'NO access to ... external data'."""
        from bantz.core.brain import CHAT_SYSTEM
        assert "no access to" not in CHAT_SYSTEM.lower()

    def test_chat_system_never_say_lack_access(self):
        """CHAT_SYSTEM must tell LLM to NEVER say it lacks external info."""
        from bantz.core.brain import CHAT_SYSTEM
        assert "never say you lack access" in CHAT_SYSTEM.lower()

    def test_system_prompt_has_telegraph_lore(self):
        """system_prompt.py BANTZ_CHAT must mention Grand Telegraph Archives."""
        from bantz.personality.system_prompt import BANTZ_CHAT
        assert "grand telegraph archives" in BANTZ_CHAT.lower()

    def test_system_prompt_never_say_lack_access(self):
        from bantz.personality.system_prompt import BANTZ_CHAT
        assert "never say you lack access" in BANTZ_CHAT.lower()

    def test_telegram_metaphor(self):
        """Butler acts as if sending a telegram to the archives."""
        from bantz.core.brain import CHAT_SYSTEM
        assert "telegram" in CHAT_SYSTEM.lower()

    def test_cot_router_web_search_entity_lookup(self):
        """intent.py CoT prompt must map entity lookups to web_search."""
        from bantz.core.intent import COT_SYSTEM
        lower = COT_SYSTEM.lower()
        assert "who is x" in lower
        assert "entity lookup" in lower
