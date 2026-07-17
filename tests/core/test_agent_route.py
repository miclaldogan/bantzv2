"""Tests for the brain's route=="agent" dispatch (#550)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bantz.core.intent import _extract_json


class TestAgentRouteParsing:
    def test_agent_route_survives_extraction(self):
        raw = ('{"route": "agent", "agent_role": "researcher", '
               '"agent_task": "verify X across sources", "confidence": 0.9}')
        data = _extract_json(raw)
        assert data["route"] == "agent"
        assert data["agent_role"] == "researcher"
        assert data["agent_task"] == "verify X across sources"

    def test_unknown_route_still_normalizes_to_tool(self):
        raw = '{"route": "gmail", "tool_name": "", "tool_args": {}}'
        data = _extract_json(raw)
        assert data["route"] == "tool"


class TestExecuteAgentRoute:
    """brain._execute_agent_route behavior with a mocked agent_manager."""

    @pytest.fixture
    def brain(self):
        from bantz.core.brain import brain as b
        return b

    async def test_disabled_returns_none(self, brain):
        with patch("bantz.agent.agent_manager.agent_manager") as mgr:
            mgr.enabled = False
            out = await brain._execute_agent_route(
                {"agent_role": "researcher", "agent_task": "t"}, "t", "t", False,
            )
        assert out is None

    async def test_success_returns_summary_result(self, brain):
        from bantz.agent.sub_agent import SubAgentResult
        result = SubAgentResult(success=True, summary="the answer",
                                tools_used=["web_search"])
        with patch("bantz.agent.agent_manager.agent_manager") as mgr, \
             patch.object(brain, "_graph_store", new=AsyncMock()), \
             patch.object(brain, "_fire_embeddings"):
            mgr.enabled = True
            mgr.delegate = AsyncMock(return_value=result)
            out = await brain._execute_agent_route(
                {"agent_role": "research", "agent_task": "verify X"},
                "verify X", "verify X", False,
            )
        assert out is not None
        assert out.tool_used == "agent:researcher"  # alias resolved
        assert "the answer" in out.response
        assert "web_search" in out.response
        mgr.delegate.assert_awaited_once_with("researcher", "verify X")

    async def test_failed_delegation_returns_none(self, brain):
        from bantz.agent.sub_agent import SubAgentResult
        with patch("bantz.agent.agent_manager.agent_manager") as mgr:
            mgr.enabled = True
            mgr.delegate = AsyncMock(
                return_value=SubAgentResult.failure("model down"))
            out = await brain._execute_agent_route(
                {"agent_role": "researcher"}, "task", "task", False,
            )
        assert out is None

    async def test_low_autonomy_asks_first(self, brain):
        from bantz.config import config
        with patch("bantz.agent.agent_manager.agent_manager") as mgr, \
             patch.object(config, "autonomy", "low"), \
             patch("bantz.core.brain.data_layer") as dl:
            mgr.enabled = True
            mgr.delegate = AsyncMock()
            out = await brain._execute_agent_route(
                {"agent_role": "researcher", "agent_task": "audit disk"},
                "audit disk", "audit disk", False,
            )
        assert out is not None and out.needs_confirm
        assert out.pending_tool == "delegate_task"
        mgr.delegate.assert_not_awaited()
        dl.conversations.add.assert_called_once()

    async def test_low_autonomy_confirmed_proceeds(self, brain):
        from bantz.config import config
        from bantz.agent.sub_agent import SubAgentResult
        result = SubAgentResult(success=True, summary="done")
        with patch("bantz.agent.agent_manager.agent_manager") as mgr, \
             patch.object(config, "autonomy", "low"), \
             patch.object(brain, "_graph_store", new=AsyncMock()), \
             patch.object(brain, "_fire_embeddings"):
            mgr.enabled = True
            mgr.delegate = AsyncMock(return_value=result)
            out = await brain._execute_agent_route(
                {"agent_role": "researcher", "agent_task": "audit disk"},
                "audit disk", "audit disk", True,
            )
        assert out is not None and not out.needs_confirm
        mgr.delegate.assert_awaited_once()

    async def test_empty_task_falls_back_to_input(self, brain):
        from bantz.agent.sub_agent import SubAgentResult
        result = SubAgentResult(success=True, summary="ok")
        with patch("bantz.agent.agent_manager.agent_manager") as mgr, \
             patch.object(brain, "_graph_store", new=AsyncMock()), \
             patch.object(brain, "_fire_embeddings"):
            mgr.enabled = True
            mgr.delegate = AsyncMock(return_value=result)
            await brain._execute_agent_route(
                {"agent_role": "researcher"}, "the english input", "orig", False,
            )
        mgr.delegate.assert_awaited_once_with("researcher", "the english input")
