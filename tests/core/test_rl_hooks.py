"""Tests for bantz.core.rl_hooks (#226).

Covers:
  - rl_reward_hook: positive/negative/uninitialized/error paths
  - rl_feedback_reward: positive/negative sentiment + offloading
  - AsyncDBExecutor integration (run_in_db is called with write=True)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bantz.core.rl_hooks import rl_reward_hook, rl_feedback_reward


# ═══════════════════════════════════════════════════════════════════════════
# rl_reward_hook
# ═══════════════════════════════════════════════════════════════════════════


class TestRLRewardHook:
    """rl_reward_hook offloads RL engine via run_in_db."""

    @pytest.mark.asyncio
    async def test_positive_reward_on_success(self):
        from bantz.tools import ToolResult
        result = ToolResult(success=True, output="ok")

        mock_engine = MagicMock()
        mock_engine.initialized = True
        mock_encode = MagicMock(return_value=MagicMock(key="morning|monday|home|weather"))

        mock_run_in_db = AsyncMock()

        with patch("bantz.agent.rl_engine.rl_engine", mock_engine), \
             patch("bantz.agent.rl_engine.encode_state", mock_encode), \
             patch("bantz.core.time_context.time_ctx") as mock_tc, \
             patch("bantz.data.async_executor.run_in_db", mock_run_in_db):
            mock_tc.snapshot.return_value = {
                "time_segment": "morning",
                "day_name": "monday",
                "location": "home",
            }
            # Need to re-import to get patched version
            from bantz.core import rl_hooks
            await rl_hooks.rl_reward_hook("weather", result)

        mock_run_in_db.assert_called_once()
        call_kwargs = mock_run_in_db.call_args
        assert call_kwargs[1]["write"] is True

    @pytest.mark.asyncio
    async def test_negative_reward_on_failure(self):
        from bantz.tools import ToolResult
        result = ToolResult(success=False, output="", error="fail")

        mock_engine = MagicMock()
        mock_engine.initialized = True
        mock_encode = MagicMock(return_value=MagicMock(key="morning|monday|home|shell"))

        mock_run_in_db = AsyncMock()

        with patch("bantz.agent.rl_engine.rl_engine", mock_engine), \
             patch("bantz.agent.rl_engine.encode_state", mock_encode), \
             patch("bantz.core.time_context.time_ctx") as mock_tc, \
             patch("bantz.data.async_executor.run_in_db", mock_run_in_db):
            mock_tc.snapshot.return_value = {
                "time_segment": "morning",
                "day_name": "monday",
                "location": "home",
            }
            from bantz.core import rl_hooks
            await rl_hooks.rl_reward_hook("shell", result)

        mock_run_in_db.assert_called_once()

    @pytest.mark.asyncio
    async def test_noop_when_uninitialized(self):
        from bantz.tools import ToolResult
        result = ToolResult(success=True, output="ok")

        mock_engine = MagicMock()
        mock_engine.initialized = False

        mock_run_in_db = AsyncMock()

        with patch("bantz.agent.rl_engine.rl_engine", mock_engine), \
             patch("bantz.data.async_executor.run_in_db", mock_run_in_db):
            await rl_reward_hook("shell", result)

        mock_run_in_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_crash_on_import_error(self):
        from bantz.tools import ToolResult
        result = ToolResult(success=True, output="ok")

        with patch.dict("sys.modules", {"bantz.agent.rl_engine": None}):
            # Should not raise
            await rl_reward_hook("weather", result)


# ═══════════════════════════════════════════════════════════════════════════
# rl_feedback_reward
# ═══════════════════════════════════════════════════════════════════════════


class TestRLFeedbackReward:
    """rl_feedback_reward offloads sentiment RL signal via run_in_db."""

    @pytest.mark.asyncio
    async def test_positive_feedback(self):
        mock_engine = MagicMock()
        mock_engine.initialized = True
        mock_encode = MagicMock(return_value=MagicMock())
        mock_action = MagicMock()

        mock_run_in_db = AsyncMock()

        tc = {"time_segment": "morning", "day_name": "monday", "location": "home"}

        with patch("bantz.agent.rl_engine.rl_engine", mock_engine), \
             patch("bantz.agent.rl_engine.encode_state", mock_encode), \
             patch("bantz.agent.rl_engine.Action", mock_action), \
             patch("bantz.data.async_executor.run_in_db", mock_run_in_db):
            await rl_feedback_reward("positive", tc)

        mock_run_in_db.assert_called_once()
        assert mock_run_in_db.call_args[1]["write"] is True

    @pytest.mark.asyncio
    async def test_negative_feedback(self):
        mock_engine = MagicMock()
        mock_engine.initialized = True
        mock_encode = MagicMock(return_value=MagicMock())
        mock_action = MagicMock()

        mock_run_in_db = AsyncMock()

        tc = {"time_segment": "evening", "day_name": "friday", "location": "office"}

        with patch("bantz.agent.rl_engine.rl_engine", mock_engine), \
             patch("bantz.agent.rl_engine.encode_state", mock_encode), \
             patch("bantz.agent.rl_engine.Action", mock_action), \
             patch("bantz.data.async_executor.run_in_db", mock_run_in_db):
            await rl_feedback_reward("negative", tc)

        mock_run_in_db.assert_called_once()

    @pytest.mark.asyncio
    async def test_noop_when_uninitialized(self):
        mock_engine = MagicMock()
        mock_engine.initialized = False

        mock_run_in_db = AsyncMock()

        tc = {"time_segment": "morning", "day_name": "monday", "location": "home"}

        with patch("bantz.agent.rl_engine.rl_engine", mock_engine), \
             patch("bantz.data.async_executor.run_in_db", mock_run_in_db):
            await rl_feedback_reward("positive", tc)

        mock_run_in_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_crash_on_import_error(self):
        with patch.dict("sys.modules", {"bantz.agent.rl_engine": None}):
            await rl_feedback_reward("positive", {})
