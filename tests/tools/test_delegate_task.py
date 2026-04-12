"""Tests for bantz.tools.delegate_task — Issue #321 delegate_task tool.

Covers:
  1. Tool registration — present in registry with correct metadata
  2. execute() — happy path delegation via AgentManager
  3. Validation — missing role, missing task_description
  4. Disabled state — returns error when multi-agent is off
  5. Context parsing — string context auto-parsed as JSON
  6. Error propagation — agent failure surfaced correctly
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bantz.tools.delegate_task import DelegateTaskTool
from bantz.agent.sub_agent import SubAgentResult


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

class TestRegistration:
    """delegate_task tool is properly registered."""

    def test_registered_in_registry(self):
        """Tool is findable in the global registry."""
        from bantz.tools import registry
        tool = registry.get("delegate_task")
        assert tool is not None
        assert tool.name == "delegate_task"

    def test_metadata(self):
        tool = DelegateTaskTool()
        assert tool.name == "delegate_task"
        assert tool.risk_level == "safe"
        assert "researcher" in tool.description.lower()
        assert "developer" in tool.description.lower()
        assert "reviewer" in tool.description.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestValidation:
    """Parameter validation."""

    @pytest.mark.asyncio
    async def test_missing_agent_role(self):
        tool = DelegateTaskTool()
        result = await tool.execute(task_description="Find GDP")
        assert result.success is False
        assert "agent_role" in result.error

    @pytest.mark.asyncio
    async def test_missing_task_description(self):
        tool = DelegateTaskTool()
        result = await tool.execute(agent_role="researcher")
        assert result.success is False
        assert "task_description" in result.error

    @pytest.mark.asyncio
    async def test_both_missing(self):
        tool = DelegateTaskTool()
        result = await tool.execute()
        assert result.success is False


# ═══════════════════════════════════════════════════════════════════════════
# Disabled State
# ═══════════════════════════════════════════════════════════════════════════

class TestDisabledState:
    """Tool returns error when multi-agent is disabled."""

    @pytest.mark.asyncio
    async def test_disabled_returns_error(self):
        tool = DelegateTaskTool()
        mock_mgr = MagicMock()
        mock_mgr.enabled = False

        with patch("bantz.tools.delegate_task.agent_manager", mock_mgr, create=True), \
             patch("bantz.agent.agent_manager.agent_manager", mock_mgr):
            result = await tool.execute(
                agent_role="researcher",
                task_description="Find GDP",
            )

        assert result.success is False
        assert "disabled" in result.error.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Happy Path
# ═══════════════════════════════════════════════════════════════════════════

class TestHappyPath:
    """Successful delegation through the tool."""

    @pytest.mark.asyncio
    async def test_delegate_success(self):
        tool = DelegateTaskTool()
        mock_mgr = MagicMock()
        mock_mgr.enabled = True
        mock_mgr.delegate = AsyncMock(return_value=SubAgentResult(
            success=True,
            summary="GDP of Turkey is $1T",
            data={"gdp": 1_000_000_000_000},
            tools_used=["web_search"],
        ))

        with patch("bantz.agent.agent_manager.agent_manager", mock_mgr):
            result = await tool.execute(
                agent_role="researcher",
                task_description="Find Turkey GDP",
                context={"year": 2025},
            )

        assert result.success is True
        assert "GDP" in result.output or "$1T" in result.output
        assert result.data["gdp"] == 1_000_000_000_000
        mock_mgr.delegate.assert_called_once_with(
            role="researcher",
            task="Find Turkey GDP",
            context={"year": 2025},
        )

    @pytest.mark.asyncio
    async def test_tools_used_in_output(self):
        """Tool names appear in the output."""
        tool = DelegateTaskTool()
        mock_mgr = MagicMock()
        mock_mgr.enabled = True
        mock_mgr.delegate = AsyncMock(return_value=SubAgentResult(
            success=True,
            summary="Result found",
            tools_used=["web_search", "read_url"],
        ))

        with patch("bantz.agent.agent_manager.agent_manager", mock_mgr):
            result = await tool.execute(
                agent_role="researcher",
                task_description="Search",
            )

        assert "web_search" in result.output
        assert "read_url" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# Context Parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestContextParsing:
    """Context parameter handling."""

    @pytest.mark.asyncio
    async def test_string_context_parsed_as_json(self):
        """JSON string context is auto-parsed to dict."""
        tool = DelegateTaskTool()
        mock_mgr = MagicMock()
        mock_mgr.enabled = True
        mock_mgr.delegate = AsyncMock(return_value=SubAgentResult(
            success=True, summary="ok",
        ))

        with patch("bantz.agent.agent_manager.agent_manager", mock_mgr):
            await tool.execute(
                agent_role="dev",
                task_description="Write code",
                context='{"lang": "python"}',
            )

        _, kwargs = mock_mgr.delegate.call_args
        assert kwargs["context"] == {"lang": "python"}

    @pytest.mark.asyncio
    async def test_non_json_string_wrapped(self):
        """Non-JSON string context is wrapped in {"raw": ...}."""
        tool = DelegateTaskTool()
        mock_mgr = MagicMock()
        mock_mgr.enabled = True
        mock_mgr.delegate = AsyncMock(return_value=SubAgentResult(
            success=True, summary="ok",
        ))

        with patch("bantz.agent.agent_manager.agent_manager", mock_mgr):
            await tool.execute(
                agent_role="dev",
                task_description="Write code",
                context="just some text",
            )

        _, kwargs = mock_mgr.delegate.call_args
        assert kwargs["context"] == {"raw": "just some text"}

    @pytest.mark.asyncio
    async def test_none_context_becomes_empty_dict(self):
        """None context becomes {}."""
        tool = DelegateTaskTool()
        mock_mgr = MagicMock()
        mock_mgr.enabled = True
        mock_mgr.delegate = AsyncMock(return_value=SubAgentResult(
            success=True, summary="ok",
        ))

        with patch("bantz.agent.agent_manager.agent_manager", mock_mgr):
            await tool.execute(
                agent_role="reviewer",
                task_description="Check code",
                context=None,
            )

        _, kwargs = mock_mgr.delegate.call_args
        assert kwargs["context"] == {}


# ═══════════════════════════════════════════════════════════════════════════
# Error Propagation
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorPropagation:
    """Agent failures surfaced correctly via ToolResult."""

    @pytest.mark.asyncio
    async def test_agent_failure(self):
        tool = DelegateTaskTool()
        mock_mgr = MagicMock()
        mock_mgr.enabled = True
        mock_mgr.delegate = AsyncMock(return_value=SubAgentResult(
            success=False,
            summary="",
            error="Researcher agent timed out after 120s.",
        ))

        with patch("bantz.agent.agent_manager.agent_manager", mock_mgr):
            result = await tool.execute(
                agent_role="researcher",
                task_description="Search",
            )

        assert result.success is False
        assert "timed out" in result.error
