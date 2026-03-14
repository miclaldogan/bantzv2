"""Tests for bantz.core.rl_hooks (#226, updated for #221 Affinity Engine).

Covers:
  - rl_reward_hook: positive/negative/uninitialized/error paths
  - rl_feedback_reward: positive/negative sentiment
  - Uses affinity_engine.add_reward instead of old RL engine
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bantz.core.rl_hooks import rl_reward_hook, rl_feedback_reward


# ═══════════════════════════════════════════════════════════════════════════
# rl_reward_hook
# ═══════════════════════════════════════════════════════════════════════════


class TestRLRewardHook:
    """rl_reward_hook sends reward to affinity engine."""

    @pytest.mark.asyncio
    async def test_positive_reward_on_success(self):
        from bantz.tools import ToolResult
        result = ToolResult(success=True, output="ok")

        mock_ae = MagicMock()
        mock_ae.initialized = True

        with patch("bantz.agent.affinity_engine.affinity_engine", mock_ae):
            from bantz.core import rl_hooks
            await rl_hooks.rl_reward_hook("weather", result)

        mock_ae.add_reward.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_negative_reward_on_failure(self):
        from bantz.tools import ToolResult
        result = ToolResult(success=False, output="", error="fail")

        mock_ae = MagicMock()
        mock_ae.initialized = True

        with patch("bantz.agent.affinity_engine.affinity_engine", mock_ae):
            from bantz.core import rl_hooks
            await rl_hooks.rl_reward_hook("shell", result)

        mock_ae.add_reward.assert_called_once_with(-0.5)

    @pytest.mark.asyncio
    async def test_noop_when_uninitialized(self):
        from bantz.tools import ToolResult
        result = ToolResult(success=True, output="ok")

        mock_ae = MagicMock()
        mock_ae.initialized = False

        with patch("bantz.agent.affinity_engine.affinity_engine", mock_ae):
            await rl_reward_hook("shell", result)

        mock_ae.add_reward.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_crash_on_import_error(self):
        from bantz.tools import ToolResult
        result = ToolResult(success=True, output="ok")

        with patch.dict("sys.modules", {"bantz.agent.affinity_engine": None}):
            # Should not raise
            await rl_reward_hook("weather", result)


# ═══════════════════════════════════════════════════════════════════════════
# rl_feedback_reward
# ═══════════════════════════════════════════════════════════════════════════


class TestRLFeedbackReward:
    """rl_feedback_reward sends sentiment signal to affinity engine."""

    @pytest.mark.asyncio
    async def test_positive_feedback(self):
        mock_ae = MagicMock()
        mock_ae.initialized = True

        tc = {"time_segment": "morning", "day_name": "monday", "location": "home"}

        with patch("bantz.agent.affinity_engine.affinity_engine", mock_ae):
            await rl_feedback_reward("positive", tc)

        mock_ae.add_reward.assert_called_once_with(2.0)

    @pytest.mark.asyncio
    async def test_negative_feedback(self):
        mock_ae = MagicMock()
        mock_ae.initialized = True

        tc = {"time_segment": "evening", "day_name": "friday", "location": "office"}

        with patch("bantz.agent.affinity_engine.affinity_engine", mock_ae):
            await rl_feedback_reward("negative", tc)

        mock_ae.add_reward.assert_called_once_with(-2.0)

    @pytest.mark.asyncio
    async def test_noop_when_uninitialized(self):
        mock_ae = MagicMock()
        mock_ae.initialized = False

        tc = {"time_segment": "morning", "day_name": "monday", "location": "home"}

        with patch("bantz.agent.affinity_engine.affinity_engine", mock_ae):
            await rl_feedback_reward("positive", tc)

        mock_ae.add_reward.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_crash_on_import_error(self):
        with patch.dict("sys.modules", {"bantz.agent.affinity_engine": None}):
            await rl_feedback_reward("positive", {})
