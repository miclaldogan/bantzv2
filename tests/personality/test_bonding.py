"""
Tests for bantz.personality.bonding — RL-Based Bonding Meter (#172).

Covers:
  ✓ Sigmoid math: edge values (0, midpoint, large positive, large negative)
  ✓ Formality index: monotonic with reward (higher reward → higher index)
  ✓ Highwater mark: index never drops more than 10 % below peak
  ✓ Highwater persistence: stored and reloaded from SQLite
  ✓ Tier selection: correct tier for each index range
  ✓ Tier labels: 5 butler-specific tiers (Ultra Formal → Bonded)
  ✓ formality_hint(): returns prompt text matching tier
  ✓ Config disabled: returns default "Balanced" tier when bonding_enabled=False
  ✓ get_formality_hint(): one-call convenience with RL engine integration
  ✓ Config fields: bonding_enabled, bonding_sigmoid_rate, bonding_sigmoid_midpoint
  ✓ Template integration: {formality_hint} in CHAT_SYSTEM and FINALIZER_SYSTEM
  ✓ brain.py _formality_hint() helper exists
  ✓ 1920s butler character: all tiers use "ma'am", never modern slang
"""
from __future__ import annotations

import math
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bantz.personality.bonding import (
    TIERS,
    DEFAULT_TIER,
    BondingMeter,
    bonding_meter,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_meter(db_path=None, rate=0.04, midpoint=25.0, enabled=True):
    """Create a fresh BondingMeter with a temp DB."""
    m = BondingMeter()
    with patch("bantz.personality.bonding.config") as cfg:
        cfg.bonding_sigmoid_rate = rate
        cfg.bonding_sigmoid_midpoint = midpoint
        cfg.bonding_enabled = enabled
        if db_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            db_path = tmp.name
            tmp.close()
        m.init(db_path)
    return m, db_path


def _sigmoid(reward, rate=0.04, midpoint=25.0):
    return 1.0 / (1.0 + math.exp(-rate * (reward - midpoint)))


# ═══════════════════════════════════════════════════════════════════════════
# Sigmoid math
# ═══════════════════════════════════════════════════════════════════════════


class TestSigmoidMath:
    def test_at_midpoint_returns_half(self):
        assert BondingMeter._sigmoid(25.0, 0.04, 25.0) == pytest.approx(0.5, abs=1e-6)

    def test_at_zero_reward(self):
        val = BondingMeter._sigmoid(0.0, 0.04, 25.0)
        assert val < 0.3  # well below midpoint
        assert val > 0.0

    def test_large_positive(self):
        val = BondingMeter._sigmoid(200.0, 0.04, 25.0)
        assert val > 0.99

    def test_large_negative(self):
        val = BondingMeter._sigmoid(-100.0, 0.04, 25.0)
        assert val < 0.01

    def test_monotonic(self):
        """Higher reward → higher index."""
        vals = [BondingMeter._sigmoid(r, 0.04, 25.0) for r in range(-50, 150, 10)]
        for i in range(1, len(vals)):
            assert vals[i] > vals[i - 1]

    def test_custom_rate(self):
        """Steeper rate → sharper transition at midpoint."""
        slow = BondingMeter._sigmoid(30.0, 0.02, 25.0)
        fast = BondingMeter._sigmoid(30.0, 0.10, 25.0)
        # Both above 0.5, but faster rate gives higher value
        assert fast > slow

    def test_custom_midpoint(self):
        """Higher midpoint shifts the curve right."""
        low = BondingMeter._sigmoid(25.0, 0.04, 20.0)
        high = BondingMeter._sigmoid(25.0, 0.04, 40.0)
        assert low > high


# ═══════════════════════════════════════════════════════════════════════════
# Formality index with highwater mark
# ═══════════════════════════════════════════════════════════════════════════


class TestFormalityIndex:
    def test_basic_index(self):
        m, _ = _make_meter()
        with patch("bantz.personality.bonding.config") as cfg:
            cfg.bonding_sigmoid_rate = 0.04
            cfg.bonding_sigmoid_midpoint = 25.0
            idx = m.formality_index(25.0)
        assert idx == pytest.approx(0.5, abs=0.05)

    def test_highwater_prevents_crash(self):
        """After seeing high reward, drop in reward can only lower index by 10%."""
        m, _ = _make_meter()
        with patch("bantz.personality.bonding.config") as cfg:
            cfg.bonding_sigmoid_rate = 0.04
            cfg.bonding_sigmoid_midpoint = 25.0
            # Build up high reward
            high_idx = m.formality_index(200.0)  # ~0.999
            # Now sudden drop
            low_idx = m.formality_index(-100.0)  # raw ~0.006
        # Should be clamped to 90% of highwater
        assert low_idx >= high_idx * 0.9 - 0.001

    def test_highwater_monotonic_upward(self):
        """Highwater only ever increases."""
        m, _ = _make_meter()
        with patch("bantz.personality.bonding.config") as cfg:
            cfg.bonding_sigmoid_rate = 0.04
            cfg.bonding_sigmoid_midpoint = 25.0
            m.formality_index(10.0)
            hw1 = m.highwater
            m.formality_index(50.0)
            hw2 = m.highwater
            m.formality_index(20.0)
            hw3 = m.highwater
        assert hw2 > hw1
        assert hw3 == hw2  # doesn't drop

    def test_highwater_persisted(self):
        """Highwater survives close/reopen."""
        m, db_path = _make_meter()
        with patch("bantz.personality.bonding.config") as cfg:
            cfg.bonding_sigmoid_rate = 0.04
            cfg.bonding_sigmoid_midpoint = 25.0
            m.formality_index(100.0)
            hw = m.highwater
        m.close()

        m2 = BondingMeter()
        m2.init(db_path)
        assert m2.highwater == pytest.approx(hw, abs=1e-6)
        m2.close()


# ═══════════════════════════════════════════════════════════════════════════
# Tier selection
# ═══════════════════════════════════════════════════════════════════════════


class TestTierSelection:
    def test_five_tiers(self):
        assert len(TIERS) == 5

    def test_tier_ranges_contiguous(self):
        for i in range(1, len(TIERS)):
            assert TIERS[i][0] == TIERS[i - 1][1]

    def test_tier_starts_at_zero(self):
        assert TIERS[0][0] == 0.0

    def test_tier_ends_at_one(self):
        assert TIERS[-1][1] == 1.0

    def test_labels(self):
        labels = [t[2] for t in TIERS]
        assert labels == ["Ultra Formal", "Formal", "Balanced", "Trusted", "Bonded"]

    def test_ultra_formal_tier_at_low_index(self):
        tier = BondingMeter._tier_for_index(0.05)
        assert tier[2] == "Ultra Formal"

    def test_balanced_tier_at_midpoint(self):
        tier = BondingMeter._tier_for_index(0.5)
        assert tier[2] == "Balanced"

    def test_bonded_tier_at_high_index(self):
        tier = BondingMeter._tier_for_index(0.95)
        assert tier[2] == "Bonded"

    def test_edge_at_boundary(self):
        """Index exactly at boundary goes to next tier."""
        tier = BondingMeter._tier_for_index(0.20)
        assert tier[2] == "Formal"

    def test_index_1_0_is_bonded(self):
        tier = BondingMeter._tier_for_index(1.0)
        assert tier[2] == "Bonded"


# ═══════════════════════════════════════════════════════════════════════════
# formality_hint & formality_label
# ═══════════════════════════════════════════════════════════════════════════


class TestFormalityHint:
    def test_hint_returns_string(self):
        m, _ = _make_meter()
        with patch("bantz.personality.bonding.config") as cfg:
            cfg.bonding_sigmoid_rate = 0.04
            cfg.bonding_sigmoid_midpoint = 25.0
            cfg.bonding_enabled = True
            hint = m.formality_hint(25.0)
        assert isinstance(hint, str)
        assert len(hint) > 20

    def test_label_at_midpoint(self):
        m, _ = _make_meter()
        with patch("bantz.personality.bonding.config") as cfg:
            cfg.bonding_sigmoid_rate = 0.04
            cfg.bonding_sigmoid_midpoint = 25.0
            cfg.bonding_enabled = True
            label = m.formality_label(25.0)
        assert label == "Balanced"

    def test_disabled_returns_default(self):
        m, _ = _make_meter()
        with patch("bantz.personality.bonding.config") as cfg:
            cfg.bonding_enabled = False
            hint = m.formality_hint(200.0)
        assert hint == DEFAULT_TIER[3]  # Balanced hint text

    def test_disabled_label_returns_default(self):
        m, _ = _make_meter()
        with patch("bantz.personality.bonding.config") as cfg:
            cfg.bonding_enabled = False
            label = m.formality_label(200.0)
        assert label == DEFAULT_TIER[2]  # "Balanced"


# ═══════════════════════════════════════════════════════════════════════════
# get_formality_hint convenience
# ═══════════════════════════════════════════════════════════════════════════


class TestGetFormalityHint:
    def test_with_rl_engine(self):
        m, _ = _make_meter()
        mock_rl = MagicMock()
        mock_rl.initialized = True
        mock_rl.cumulative_reward.return_value = 50.0
        with patch("bantz.personality.bonding.config") as cfg:
            cfg.bonding_sigmoid_rate = 0.04
            cfg.bonding_sigmoid_midpoint = 25.0
            cfg.bonding_enabled = True
            with patch("bantz.agent.rl_engine.rl_engine", mock_rl):
                hint = m.get_formality_hint()
        assert isinstance(hint, str)
        assert len(hint) > 10

    def test_rl_not_initialized_returns_default(self):
        m, _ = _make_meter()
        mock_rl = MagicMock()
        mock_rl.initialized = False
        with patch("bantz.agent.rl_engine.rl_engine", mock_rl):
            hint = m.get_formality_hint()
        assert hint == DEFAULT_TIER[3]

    def test_import_error_returns_default(self):
        m, _ = _make_meter()
        with patch.dict("sys.modules", {"bantz.agent.rl_engine": None}):
            hint = m.get_formality_hint()
        assert hint == DEFAULT_TIER[3]


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════


class TestSingleton:
    def test_singleton_exists(self):
        assert bonding_meter is not None
        assert isinstance(bonding_meter, BondingMeter)


# ═══════════════════════════════════════════════════════════════════════════
# Config fields
# ═══════════════════════════════════════════════════════════════════════════


class TestConfig:
    def test_bonding_enabled_field(self):
        from bantz.config import Config
        fields = Config.model_fields
        assert "bonding_enabled" in fields

    def test_bonding_sigmoid_rate_field(self):
        from bantz.config import Config
        fields = Config.model_fields
        assert "bonding_sigmoid_rate" in fields

    def test_bonding_sigmoid_midpoint_field(self):
        from bantz.config import Config
        fields = Config.model_fields
        assert "bonding_sigmoid_midpoint" in fields

    def test_defaults(self):
        from bantz.config import Config
        fields = Config.model_fields
        assert fields["bonding_enabled"].default is True
        assert fields["bonding_sigmoid_rate"].default == 0.04
        assert fields["bonding_sigmoid_midpoint"].default == 25.0


# ═══════════════════════════════════════════════════════════════════════════
# Template integration
# ═══════════════════════════════════════════════════════════════════════════


class TestTemplateIntegration:
    def test_chat_system_has_formality_hint(self):
        from bantz.core.brain import CHAT_SYSTEM
        assert "{formality_hint}" in CHAT_SYSTEM

    def test_finalizer_system_has_formality_hint(self):
        from bantz.core.finalizer import FINALIZER_SYSTEM
        assert "{formality_hint}" in FINALIZER_SYSTEM

    def test_brain_formality_hint_helper(self):
        from bantz.core import brain
        assert callable(getattr(brain, "_formality_hint", None))

    def test_formality_hint_returns_string(self):
        from bantz.core.brain import _formality_hint
        # Without RL engine being initialised, should return "" safely
        result = _formality_hint()
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════
# 1920s Butler character compliance
# ═══════════════════════════════════════════════════════════════════════════


class TestButlerCharacter:
    """All tiers must preserve the 1920s butler character."""

    def test_no_bro_in_tiers(self):
        for _, _, _, hint in TIERS:
            assert "bro" not in hint.lower()

    def test_no_boss_in_tiers(self):
        for _, _, _, hint in TIERS:
            assert "boss" not in hint.lower()

    def test_no_dude_in_tiers(self):
        for _, _, _, hint in TIERS:
            assert "dude" not in hint.lower()

    def test_bonded_tier_still_uses_maam(self):
        """Even at max bonding, Bantz says ma'am."""
        _, _, _, hint = TIERS[-1]
        assert "ma'am" in hint

    def test_bonded_tier_no_modern_slang(self):
        _, _, _, hint = TIERS[-1]
        assert "modern slang" in hint.lower() or "never modern slang" in hint.lower()


# ═══════════════════════════════════════════════════════════════════════════
# .env.example coverage
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvExample:
    def test_env_example_has_bonding_entries(self):
        env_path = Path(__file__).resolve().parents[2] / ".env.example"
        text = env_path.read_text()
        assert "BANTZ_BONDING_ENABLED" in text
        assert "BANTZ_BONDING_SIGMOID_RATE" in text
        assert "BANTZ_BONDING_SIGMOID_MIDPOINT" in text
