"""Tests for dynamic replanning — executor circuit breaker + Plan B (#216)."""
from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

from bantz.agent.executor import PlanExecutor, PlanExecutionResult, StepResult
from bantz.agent.planner import PlanStep, PlannerAgent


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_step(num: int, tool: str, **params) -> PlanStep:
    return PlanStep(step=num, tool=tool, params=params, description=f"Step {num}")


def ok_tool_result(output: str = "ok"):
    from bantz.tools import ToolResult
    return ToolResult(success=True, output=output)


def fail_tool_result(error: str = "HTTP 403 Forbidden"):
    from bantz.tools import ToolResult
    return ToolResult(success=False, output="", error=error)


# ── 1. Baseline: no failure → no replan ──────────────────────────────────────

@pytest.mark.asyncio
async def test_no_replan_when_all_succeed():
    """All steps succeed → replanned flag stays False."""
    executor = PlanExecutor()
    steps = [make_step(1, "web_search", query="test")]

    with patch("bantz.tools.registry.get") as mock_get:
        mock_tool = AsyncMock()
        mock_tool.execute = AsyncMock(return_value=ok_tool_result("results"))
        mock_get.return_value = mock_tool

        result = await executor.run(steps, original_request="search something")

    assert result.succeeded == 1
    assert not result.replanned
    assert not result.aborted


# ── 2. Failure without original_request → plain abort ────────────────────────

@pytest.mark.asyncio
async def test_no_replan_without_original_request():
    """Failure with no original_request → circuit breaker aborts, no replan."""
    executor = PlanExecutor()
    steps = [make_step(1, "read_url", url="http://x.com"), make_step(2, "summarizer")]

    with patch("bantz.tools.registry.get") as mock_get:
        mock_tool = AsyncMock()
        mock_tool.execute = AsyncMock(return_value=fail_tool_result("HTTP 403"))
        mock_get.return_value = mock_tool

        result = await executor.run(steps)  # no original_request

    assert result.aborted
    assert not result.replanned


# ── 3. Failure triggers replan when original_request provided ─────────────────

@pytest.mark.asyncio
async def test_replan_triggered_on_failure():
    """Step fails + original_request given → replanner is called."""
    executor = PlanExecutor()
    steps = [make_step(1, "read_url", url="http://blocked.com"), make_step(2, "summarizer")]

    plan_b = [make_step(1, "web_search", query="fallback")]

    with patch("bantz.tools.registry.get") as mock_get, \
         patch("bantz.tools.registry.names", return_value=["web_search", "read_url"]), \
         patch("bantz.agent.planner.planner_agent.replan", new_callable=AsyncMock) as mock_replan, \
         patch("bantz.core.notification_manager.notify_toast", side_effect=Exception):

        def tool_side_effect(name):
            if name == "read_url":
                t = AsyncMock()
                t.execute = AsyncMock(return_value=fail_tool_result("HTTP 403"))
                return t
            elif name == "web_search":
                t = AsyncMock()
                t.execute = AsyncMock(return_value=ok_tool_result("fallback results"))
                return t
            return None

        mock_get.side_effect = tool_side_effect
        mock_replan.return_value = plan_b

        result = await executor.run(steps, original_request="find article about AI")

    assert result.replanned
    mock_replan.assert_called_once()


# ── 4. Plan B executes and its results are merged ─────────────────────────────

@pytest.mark.asyncio
async def test_plan_b_results_merged():
    """Plan B steps complete → their results appear in final PlanExecutionResult."""
    executor = PlanExecutor()
    steps = [make_step(1, "read_url", url="http://blocked.com")]
    plan_b = [make_step(1, "web_search", query="alternative")]

    with patch("bantz.tools.registry.get") as mock_get, \
         patch("bantz.tools.registry.names", return_value=["web_search", "read_url"]), \
         patch("bantz.agent.planner.planner_agent.replan", new_callable=AsyncMock) as mock_replan, \
         patch("bantz.core.notification_manager.notify_toast", side_effect=Exception):

        def tool_side_effect(name):
            if name == "read_url":
                t = AsyncMock(); t.execute = AsyncMock(return_value=fail_tool_result("403"))
                return t
            t = AsyncMock(); t.execute = AsyncMock(return_value=ok_tool_result("plan b output"))
            return t

        mock_get.side_effect = tool_side_effect
        mock_replan.return_value = plan_b

        result = await executor.run(steps, original_request="research AI")

    tool_names = [sr.tool for sr in result.step_results]
    assert "web_search" in tool_names
    success_results = [sr for sr in result.step_results if sr.success]
    assert len(success_results) >= 1


# ── 5. Replan not triggered twice (guard against infinite loop) ───────────────

@pytest.mark.asyncio
async def test_no_double_replan():
    """_replanned=True flag prevents recursive replanning."""
    executor = PlanExecutor()
    steps = [make_step(1, "read_url", url="http://blocked.com")]

    with patch("bantz.tools.registry.get") as mock_get, \
         patch("bantz.agent.planner.planner_agent.replan", new_callable=AsyncMock) as mock_replan:

        mock_tool = AsyncMock()
        mock_tool.execute = AsyncMock(return_value=fail_tool_result("error"))
        mock_get.return_value = mock_tool

        # Run with _replanned=True (simulating we're already in Plan B)
        result = await executor.run(
            steps, original_request="test", _replanned=True,
        )

    # Replanner should NOT have been called since _replanned=True
    mock_replan.assert_not_called()
    assert result.aborted


# ── 6. Empty Plan B → plain abort ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_plan_b_falls_back_to_abort():
    """If replanner returns empty steps → abort normally."""
    executor = PlanExecutor()
    steps = [make_step(1, "read_url", url="http://x.com"), make_step(2, "summarizer")]

    with patch("bantz.tools.registry.get") as mock_get, \
         patch("bantz.tools.registry.names", return_value=[]), \
         patch("bantz.agent.planner.planner_agent.replan", new_callable=AsyncMock) as mock_replan, \
         patch("bantz.core.notification_manager.notify_toast", side_effect=Exception):

        mock_tool = AsyncMock()
        mock_tool.execute = AsyncMock(return_value=fail_tool_result("blocked"))
        mock_get.return_value = mock_tool
        mock_replan.return_value = []  # empty Plan B

        result = await executor.run(steps, original_request="test")

    assert result.aborted


# ── 7. PlanExecutionResult.replanned defaults to False ───────────────────────

def test_plan_execution_result_replanned_defaults_false():
    r = PlanExecutionResult()
    assert r.replanned is False


# ── 8. Replan passes correct context to planner ──────────────────────────────

@pytest.mark.asyncio
async def test_replan_receives_context_store():
    """Replanner receives completed step outputs in context_store."""
    executor = PlanExecutor()
    steps = [
        make_step(1, "web_search", query="AI news"),   # succeeds
        make_step(2, "read_url", url="http://x.com"),  # fails
    ]

    captured_kwargs: dict = {}

    async def fake_replan(**kwargs):
        captured_kwargs.update(kwargs)
        return []

    with patch("bantz.tools.registry.get") as mock_get, \
         patch("bantz.tools.registry.names", return_value=["web_search", "read_url"]), \
         patch("bantz.agent.planner.planner_agent.replan", side_effect=fake_replan), \
         patch("bantz.core.notification_manager.notify_toast", side_effect=Exception):

        def tool_side_effect(name):
            if name == "web_search":
                t = AsyncMock(); t.execute = AsyncMock(return_value=ok_tool_result("search done"))
                return t
            t = AsyncMock(); t.execute = AsyncMock(return_value=fail_tool_result("403"))
            return t

        mock_get.side_effect = tool_side_effect
        await executor.run(steps, original_request="AI research")

    # Step 1 output should be in context_store passed to replanner
    assert "context_store" in captured_kwargs
    assert 1 in captured_kwargs["context_store"]
    assert captured_kwargs["context_store"][1]["output"] == "search done"
    assert captured_kwargs["original_request"] == "AI research"
    assert captured_kwargs["failed_step_num"] == 2
