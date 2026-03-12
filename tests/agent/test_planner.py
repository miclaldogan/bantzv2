"""Tests for Issue #187 — Plan-and-Solve Multi-Step Decomposition.

"The Butler's Itinerary": Complex multi-step commands are decomposed
by the LLM into a structured step array and executed sequentially.

Covers:
  1. PlannerAgent — complexity detection heuristics
  2. PlannerAgent — LLM decomposition (mocked Ollama)
  3. PlannerAgent — itinerary formatting (Butler persona)
  4. PlanExecutor — sequential execution with context passing
  5. PlanExecutor — graceful failure handling
  6. PlanExecutor — dependency injection between steps
  7. Brain integration — complex requests route to planner
  8. Brain integration — simple requests bypass planner
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bantz.tools import ToolResult


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PlannerAgent — complexity detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestComplexityDetection:
    """PlannerAgent.is_complex must detect multi-step requests."""

    def _agent(self):
        from bantz.agent.planner import PlannerAgent
        return PlannerAgent()

    def test_search_and_save_is_complex(self):
        """'Search X and then save to a file' → complex."""
        assert self._agent().is_complex(
            "search for quantum computing articles and then save to a file"
        ) is True

    def test_email_then_calendar_is_complex(self):
        """'Check email then check my calendar' → complex."""
        assert self._agent().is_complex(
            "check my email then check my calendar for today"
        ) is True

    def test_search_and_write_file(self):
        """'Find articles about X and save a summary file' → complex."""
        assert self._agent().is_complex(
            "find articles about machine learning and save a summary to a file"
        ) is True

    def test_numbered_steps_complex(self):
        """'First search X, then save to file' → complex."""
        assert self._agent().is_complex(
            "first search for AI news, then write the results to a file"
        ) is True

    def test_simple_greeting_not_complex(self):
        """'Hello, how are you?' → NOT complex."""
        assert self._agent().is_complex("hello how are you") is False

    def test_single_tool_not_complex(self):
        """'What's the weather in Istanbul?' → NOT complex."""
        assert self._agent().is_complex("what is the weather in istanbul") is False

    def test_simple_email_not_complex(self):
        """'Check my email' → NOT complex."""
        assert self._agent().is_complex("check my email") is False

    def test_simple_file_op_not_complex(self):
        """'Create a folder named test' → NOT complex."""
        assert self._agent().is_complex("create a folder named test") is False

    def test_short_ambiguous_not_complex(self):
        """'Tell me about it' → NOT complex."""
        assert self._agent().is_complex("tell me about it") is False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PlannerAgent — LLM decomposition (mocked)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecomposition:
    """PlannerAgent.decompose uses the LLM to generate plan steps."""

    @pytest.mark.asyncio
    async def test_decomposes_search_and_save(self):
        """Search + save file decomposes into 2 steps."""
        from bantz.agent.planner import PlannerAgent

        llm_response = json.dumps([
            {"step": 1, "tool": "web_search", "params": {"query": "AI news"},
             "description": "Search for AI articles", "depends_on": None},
            {"step": 2, "tool": "filesystem", "params": {
                "action": "write", "path": "~/Desktop/ai.txt",
                "content": "{step_1_output}"},
             "description": "Save results to file", "depends_on": 1},
        ])

        agent = PlannerAgent()
        with patch("bantz.agent.planner.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=llm_response)
            steps = await agent.decompose(
                "search for AI news and save to a file",
                ["web_search", "filesystem", "gmail", "shell"],
            )

        assert len(steps) == 2
        assert steps[0].tool == "web_search"
        assert steps[0].step == 1
        assert steps[1].tool == "filesystem"
        assert steps[1].depends_on == 1

    @pytest.mark.asyncio
    async def test_decomposes_three_steps(self):
        """Email + weather + calendar → 3 independent steps."""
        from bantz.agent.planner import PlannerAgent

        llm_response = json.dumps([
            {"step": 1, "tool": "gmail", "params": {"action": "unread"},
             "description": "Check unread emails", "depends_on": None},
            {"step": 2, "tool": "weather", "params": {"city": "Istanbul"},
             "description": "Check weather in Istanbul", "depends_on": None},
            {"step": 3, "tool": "calendar", "params": {"action": "today"},
             "description": "Check today's calendar", "depends_on": None},
        ])

        agent = PlannerAgent()
        with patch("bantz.agent.planner.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=llm_response)
            steps = await agent.decompose(
                "check my email, check the weather, and show my calendar",
                ["gmail", "weather", "calendar", "web_search", "filesystem"],
            )

        assert len(steps) == 3
        assert {s.tool for s in steps} == {"gmail", "weather", "calendar"}

    @pytest.mark.asyncio
    async def test_filters_unknown_tools(self):
        """Steps with unknown tool names are dropped."""
        from bantz.agent.planner import PlannerAgent

        llm_response = json.dumps([
            {"step": 1, "tool": "web_search", "params": {"query": "test"},
             "description": "Search", "depends_on": None},
            {"step": 2, "tool": "magic_wand", "params": {},
             "description": "Cast spell", "depends_on": None},
        ])

        agent = PlannerAgent()
        with patch("bantz.agent.planner.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=llm_response)
            steps = await agent.decompose("test", ["web_search", "filesystem"])

        assert len(steps) == 1
        assert steps[0].tool == "web_search"

    @pytest.mark.asyncio
    async def test_handles_markdown_fenced_json(self):
        """LLM wraps response in ```json fences → still parsed."""
        from bantz.agent.planner import PlannerAgent

        llm_response = '```json\n[{"step": 1, "tool": "weather", "params": {"city": "London"}, "description": "Check weather", "depends_on": null}]\n```'

        agent = PlannerAgent()
        with patch("bantz.agent.planner.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=llm_response)
            steps = await agent.decompose("check weather", ["weather"])

        assert len(steps) == 1
        assert steps[0].tool == "weather"

    @pytest.mark.asyncio
    async def test_returns_empty_on_garbage(self):
        """LLM returns garbage → empty list, no crash."""
        from bantz.agent.planner import PlannerAgent

        agent = PlannerAgent()
        with patch("bantz.agent.planner.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value="I'm just a butler, I can't do that.")
            steps = await agent.decompose("do something", ["web_search"])

        assert steps == []


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PlannerAgent — itinerary formatting
# ═══════════════════════════════════════════════════════════════════════════════


class TestItineraryFormatting:
    """format_itinerary must produce butler-style announcements."""

    def test_formats_two_step_plan(self):
        from bantz.agent.planner import PlannerAgent, PlanStep

        steps = [
            PlanStep(step=1, tool="web_search", params={"query": "AI"},
                     description="Search for AI articles"),
            PlanStep(step=2, tool="filesystem", params={"action": "write"},
                     description="Save results to a file", depends_on=1),
        ]
        text = PlannerAgent().format_itinerary(steps)
        assert "itinerary" in text.lower()
        assert "1. Search for AI articles" in text
        assert "2. Save results to a file" in text
        assert "Commencing forthwith" in text

    def test_shows_dependency(self):
        from bantz.agent.planner import PlannerAgent, PlanStep

        steps = [
            PlanStep(step=1, tool="web_search", description="Search"),
            PlanStep(step=2, tool="filesystem", description="Save", depends_on=1),
        ]
        text = PlannerAgent().format_itinerary(steps)
        assert "results from step 1" in text

    def test_empty_plan_returns_empty(self):
        from bantz.agent.planner import PlannerAgent
        assert PlannerAgent().format_itinerary([]) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PlanExecutor — sequential execution
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanExecutor:
    """PlanExecutor.run executes steps and tracks results."""

    @pytest.mark.asyncio
    async def test_executes_two_steps_sequentially(self):
        """Two independent steps both succeed → summary says 2/2."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="weather", params={"city": "London"},
                     description="Check London weather"),
            PlanStep(step=2, tool="news", params={"source": "all"},
                     description="Get headlines"),
        ]

        mock_weather = AsyncMock(return_value=ToolResult(
            success=True, output="London: 15°C, cloudy"))
        mock_news = AsyncMock(return_value=ToolResult(
            success=True, output="Top stories: AI, Tech"))

        executor = PlanExecutor()
        with patch("bantz.agent.executor.registry") as mock_reg:
            def _get(name):
                tools = {
                    "weather": MagicMock(execute=mock_weather),
                    "news": MagicMock(execute=mock_news),
                }
                return tools.get(name)
            mock_reg.get = _get

            result = await executor.run(steps)

        assert result.all_success is True
        assert result.succeeded == 2
        assert result.total == 2
        assert "2" in result.summary()

    @pytest.mark.asyncio
    async def test_handles_step_failure_gracefully(self):
        """One step fails → execution continues, summary shows partial."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="web_search",
                     params={"query": "test"}, description="Search"),
            PlanStep(step=2, tool="weather",
                     params={"city": "Paris"}, description="Weather"),
        ]

        mock_search = AsyncMock(return_value=ToolResult(
            success=False, output="", error="Connection timeout"))
        mock_weather = AsyncMock(return_value=ToolResult(
            success=True, output="Paris: 20°C"))

        executor = PlanExecutor()
        with patch("bantz.agent.executor.registry") as mock_reg:
            def _get(name):
                tools = {
                    "web_search": MagicMock(execute=mock_search),
                    "weather": MagicMock(execute=mock_weather),
                }
                return tools.get(name)
            mock_reg.get = _get

            result = await executor.run(steps)

        assert result.succeeded == 1
        assert result.total == 2
        assert result.all_success is False
        assert "1 of 2" in result.summary()

    @pytest.mark.asyncio
    async def test_unknown_tool_doesnt_crash(self):
        """Step references unknown tool → noted as failure, doesn't crash."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="nonexistent", params={},
                     description="Do something"),
        ]

        executor = PlanExecutor()
        with patch("bantz.agent.executor.registry") as mock_reg:
            mock_reg.get = MagicMock(return_value=None)
            result = await executor.run(steps)

        assert result.succeeded == 0
        assert result.total == 1
        assert "not found" in result.step_results[0].error


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PlanExecutor — context passing between steps
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextPassing:
    """Output from step N must be available to step N+1 when depends_on."""

    @pytest.mark.asyncio
    async def test_step2_receives_step1_output(self):
        """Step 2 depends on step 1 → step 1 output injected into step 2 params."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        search_output = "Article: Quantum computing is the future of..."

        steps = [
            PlanStep(step=1, tool="web_search",
                     params={"query": "quantum computing"},
                     description="Search"),
            PlanStep(step=2, tool="filesystem",
                     params={"action": "write", "path": "~/Desktop/q.txt",
                             "content": "{step_1_output}"},
                     description="Save results", depends_on=1),
        ]

        captured_params: dict = {}

        async def mock_fs_execute(**kwargs):
            captured_params.update(kwargs)
            return ToolResult(success=True, output="File written")

        mock_search = AsyncMock(return_value=ToolResult(
            success=True, output=search_output))
        mock_fs = MagicMock()
        mock_fs.execute = mock_fs_execute

        executor = PlanExecutor()
        with patch("bantz.agent.executor.registry") as mock_reg:
            def _get(name):
                tools = {"web_search": MagicMock(execute=mock_search),
                         "filesystem": mock_fs}
                return tools.get(name)
            mock_reg.get = _get

            result = await executor.run(steps)

        assert result.all_success is True
        # The content should have been replaced with step 1 output
        assert "quantum computing" in captured_params.get("content", "").lower()

    @pytest.mark.asyncio
    async def test_independent_steps_no_context_leak(self):
        """Steps without depends_on don't get injected context."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="weather",
                     params={"city": "Tokyo"}, description="Weather"),
            PlanStep(step=2, tool="news",
                     params={"source": "hn"}, description="News"),
        ]

        captured: list[dict] = []

        async def mock_execute(**kwargs):
            captured.append(dict(kwargs))
            return ToolResult(success=True, output="OK")

        executor = PlanExecutor()
        with patch("bantz.agent.executor.registry") as mock_reg:
            mock_tool = MagicMock()
            mock_tool.execute = mock_execute
            mock_reg.get = MagicMock(return_value=mock_tool)

            result = await executor.run(steps)

        # Step 2 params should NOT have step 1 output injected
        assert "content" not in captured[1]  # news has no content key

    @pytest.mark.asyncio
    async def test_inject_context_replaces_placeholder(self):
        """_inject_context replaces {step_N_output} placeholders in params."""
        from bantz.agent.executor import PlanExecutor

        params = {"content": "{step_1_output}", "path": "~/file.txt"}
        result = PlanExecutor._inject_context(params, "Hello World")
        assert result["content"] == "Hello World"
        assert result["path"] == "~/file.txt"  # unchanged


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PlanExecutor — result summary formatting
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecutionSummary:
    """PlanExecutionResult.summary must produce butler-style summaries."""

    def test_all_success_summary(self):
        from bantz.agent.executor import PlanExecutionResult, StepResult

        result = PlanExecutionResult(step_results=[
            StepResult(step_number=1, tool="weather", description="Check weather",
                       success=True, output="London: 15°C"),
            StepResult(step_number=2, tool="news", description="Get news",
                       success=True, output="Headlines here"),
        ])
        s = result.summary()
        assert "all 2 tasks completed" in s.lower()
        assert "✓" in s

    def test_partial_failure_summary(self):
        from bantz.agent.executor import PlanExecutionResult, StepResult

        result = PlanExecutionResult(step_results=[
            StepResult(step_number=1, tool="web_search", description="Search",
                       success=True, output="Results"),
            StepResult(step_number=2, tool="filesystem", description="Save",
                       success=False, output="", error="Permission denied"),
        ])
        s = result.summary()
        assert "1 of 2" in s
        assert "✗" in s

    def test_empty_result(self):
        from bantz.agent.executor import PlanExecutionResult

        result = PlanExecutionResult()
        assert "empty" in result.summary().lower()

    def test_all_failed_summary(self):
        from bantz.agent.executor import PlanExecutionResult, StepResult

        result = PlanExecutionResult(step_results=[
            StepResult(step_number=1, tool="web_search", description="Search",
                       success=False, output="", error="Timeout"),
        ])
        s = result.summary()
        assert "difficulties" in s.lower() or "regret" in s.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Brain integration — _execute_plan routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestBrainPlannerIntegration:
    """Brain.process routes complex requests to the planner."""

    @pytest.mark.asyncio
    async def test_complex_request_uses_planner(self):
        """Multi-tool request triggers planner path."""
        from bantz.agent.planner import PlanStep

        plan_steps = [
            PlanStep(step=1, tool="web_search",
                     params={"query": "AI news"}, description="Search"),
            PlanStep(step=2, tool="filesystem",
                     params={"action": "write", "path": "~/Desktop/ai.txt",
                             "content": "{step_1_output}"},
                     description="Save to file", depends_on=1),
        ]

        from bantz.agent.executor import PlanExecutionResult, StepResult
        exec_result = PlanExecutionResult(step_results=[
            StepResult(step_number=1, tool="web_search",
                       description="Search", success=True, output="AI stuff"),
            StepResult(step_number=2, tool="filesystem",
                       description="Save to file", success=True,
                       output="File written"),
        ])

        with patch("bantz.agent.planner.planner_agent") as mock_planner, \
             patch("bantz.agent.executor.plan_executor") as mock_executor, \
             patch("bantz.core.brain.data_layer") as mock_dal, \
             patch("bantz.core.brain.ollama") as mock_ollama, \
             patch("bantz.core.brain.cot_route") as mock_cot:

            mock_planner.is_complex = MagicMock(return_value=True)
            mock_planner.decompose = AsyncMock(return_value=plan_steps)
            mock_planner.format_itinerary = MagicMock(
                return_value="Allow me a moment...\n1. Search\n2. Save")
            mock_executor.run = AsyncMock(return_value=exec_result)
            mock_dal.conversations = MagicMock()
            mock_dal.init = MagicMock()
            mock_ollama.chat = AsyncMock(return_value="")

            from bantz.core.brain import Brain
            brain = Brain.__new__(Brain)
            brain._memory_ready = True
            brain._graph_ready = True
            brain._feedback_ctx = ""
            brain._last_messages = []
            brain._last_events = []
            brain._last_draft = None
            brain._is_remote = False
            brain._bridge = None
            brain._to_en = AsyncMock(return_value="search for AI news and save to a file")
            brain._graph_store = AsyncMock()
            brain._fire_embeddings = MagicMock()

            # Mock workflow_engine to not detect anything
            with patch("bantz.core.workflow.workflow_engine") as mock_wf:
                mock_wf.detect = MagicMock(return_value=None)

                result = await brain.process(
                    "search for AI news and save to a file")

            assert result.tool_used == "planner"
            assert "allow me a moment" in result.response.lower()
            assert "completed successfully" in result.response.lower()

    @pytest.mark.asyncio
    async def test_simple_request_bypasses_planner(self):
        """Simple single-tool request should NOT trigger planner."""
        from bantz.agent.planner import PlannerAgent

        agent = PlannerAgent()
        assert agent.is_complex("what is the weather in istanbul") is False


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PLANNER_SYSTEM prompt validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlannerPrompt:
    """The planner system prompt must contain required elements."""

    def test_prompt_mentions_butler(self):
        from bantz.agent.planner import PLANNER_SYSTEM
        assert "butler" in PLANNER_SYSTEM.lower()

    def test_prompt_mentions_json(self):
        from bantz.agent.planner import PLANNER_SYSTEM
        assert "JSON" in PLANNER_SYSTEM

    def test_prompt_has_tool_reference(self):
        from bantz.agent.planner import PLANNER_SYSTEM
        assert "web_search" in PLANNER_SYSTEM
        assert "filesystem" in PLANNER_SYSTEM
        assert "gmail" in PLANNER_SYSTEM

    def test_prompt_has_depends_on(self):
        from bantz.agent.planner import PLANNER_SYSTEM
        assert "depends_on" in PLANNER_SYSTEM

    def test_prompt_has_examples(self):
        from bantz.agent.planner import PLANNER_SYSTEM
        assert "quantum computing" in PLANNER_SYSTEM.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. PlanStep parsing edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanStepParsing:
    """Edge cases in _parse_steps."""

    def test_parse_valid_array(self):
        from bantz.agent.planner import PlannerAgent
        data = PlannerAgent._parse_steps(
            '[{"step": 1, "tool": "weather", "params": {"city": "X"}, "description": "Y", "depends_on": null}]'
        )
        assert len(data) == 1
        assert data[0]["tool"] == "weather"

    def test_parse_strips_fences(self):
        from bantz.agent.planner import PlannerAgent
        data = PlannerAgent._parse_steps(
            '```json\n[{"step": 1, "tool": "a", "params": {}, "description": "", "depends_on": null}]\n```'
        )
        assert len(data) == 1

    def test_parse_rejects_non_array(self):
        from bantz.agent.planner import PlannerAgent
        with pytest.raises(ValueError, match="Expected JSON array"):
            PlannerAgent._parse_steps('{"not": "an array"}')

    def test_parse_rejects_empty_array(self):
        from bantz.agent.planner import PlannerAgent
        with pytest.raises(ValueError, match="Empty"):
            PlannerAgent._parse_steps('[]')

    def test_parse_rejects_garbage(self):
        from bantz.agent.planner import PlannerAgent
        with pytest.raises(Exception):
            PlannerAgent._parse_steps("I'm just a butler, I can't do that.")
