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

    def test_summarize_and_save_is_complex(self):
        """'Search X, summarize it, and save' → complex (process_text keyword)."""
        assert self._agent().is_complex(
            "search for AI news, summarize it, and save to a file"
        ) is True

    def test_translate_and_write_is_complex(self):
        """'Translate this text and write to a file' → complex."""
        assert self._agent().is_complex(
            "translate the document results and write to a file"
        ) is True


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
    async def test_process_text_step_not_filtered(self):
        """Steps with tool='process_text' must NOT be filtered out."""
        from bantz.agent.planner import PlannerAgent

        llm_response = json.dumps([
            {"step": 1, "tool": "web_search", "params": {"query": "AI news"},
             "description": "Search", "depends_on": None},
            {"step": 2, "tool": "process_text",
             "params": {"instruction": "Summarize: {step_1_output}"},
             "description": "Summarize results", "depends_on": 1},
            {"step": 3, "tool": "filesystem",
             "params": {"action": "write", "path": "~/ai.txt",
                        "content": "{step_2_output}"},
             "description": "Save summary", "depends_on": 2},
        ])

        agent = PlannerAgent()
        with patch("bantz.agent.planner.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=llm_response)
            steps = await agent.decompose(
                "search AI news, summarize, save to file",
                ["web_search", "filesystem"],
            )

        assert len(steps) == 3
        assert steps[1].tool == "process_text"
        assert steps[1].depends_on == 1
        assert steps[2].depends_on == 2

    @pytest.mark.asyncio
    async def test_read_url_step_not_filtered(self):
        """Steps with tool='read_url' must NOT be filtered out."""
        from bantz.agent.planner import PlannerAgent

        llm_response = json.dumps([
            {"step": 1, "tool": "web_search", "params": {"query": "AI"},
             "description": "Search", "depends_on": None},
            {"step": 2, "tool": "read_url",
             "params": {"url": "https://example.com/article"},
             "description": "Read full article", "depends_on": 1},
            {"step": 3, "tool": "process_text",
             "params": {"instruction": "Summarize: {step_2_output}"},
             "description": "Summarize", "depends_on": 2},
        ])

        agent = PlannerAgent()
        with patch("bantz.agent.planner.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=llm_response)
            steps = await agent.decompose(
                "search AI, read the article, summarize",
                ["web_search", "filesystem"],
            )

        assert len(steps) == 3
        assert steps[1].tool == "read_url"
        assert steps[1].depends_on == 1
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

    def test_prompt_has_process_text_tool(self):
        """TOOL REFERENCE must include process_text."""
        from bantz.agent.planner import PLANNER_SYSTEM
        assert "process_text" in PLANNER_SYSTEM

    def test_prompt_has_critical_tool_rules(self):
        """Prompt must contain CRITICAL TOOL RULES forbidding web_search misuse."""
        from bantz.agent.planner import PLANNER_SYSTEM
        assert "CRITICAL TOOL RULES" in PLANNER_SYSTEM
        assert "NEVER use" in PLANNER_SYSTEM
        assert "summarize" in PLANNER_SYSTEM.lower()

    def test_example_uses_three_step_flow(self):
        """The examples must show web_search → read_url → process_text → filesystem (4-step)."""
        from bantz.agent.planner import PLANNER_SYSTEM
        # process_text must appear in the example section
        idx_examples = PLANNER_SYSTEM.index("EXAMPLES:")
        example_block = PLANNER_SYSTEM[idx_examples:]
        assert "process_text" in example_block
        assert "read_url" in example_block
        assert "step_2_output" in example_block
        assert "step_3_output" in example_block

    def test_prompt_has_read_url_tool(self):
        """TOOL REFERENCE must include read_url."""
        from bantz.agent.planner import PLANNER_SYSTEM
        assert "read_url" in PLANNER_SYSTEM
        assert "full text" in PLANNER_SYSTEM.lower() or "full content" in PLANNER_SYSTEM.lower()

    def test_prompt_has_deep_reading_rule(self):
        """CRITICAL TOOL RULES must mention read_url for thorough research."""
        from bantz.agent.planner import PLANNER_SYSTEM
        assert "read_url" in PLANNER_SYSTEM
        # The rule about using read_url for full article text
        rules_start = PLANNER_SYSTEM.index("CRITICAL TOOL RULES")
        # Find the next section header after CRITICAL TOOL RULES
        rules_end = PLANNER_SYSTEM.index("\nRULES:\n", rules_start)
        rules_block = PLANNER_SYSTEM[rules_start:rules_end]
        assert "read_url" in rules_block


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


# ═══════════════════════════════════════════════════════════════════════════════
# 10. PlanExecutor — process_text virtual tool
# ═══════════════════════════════════════════════════════════════════════════════


class TestProcessTextVirtualTool:
    """The executor must route process_text steps to the LLM, not the registry."""

    @pytest.mark.asyncio
    async def test_process_text_calls_llm(self):
        """process_text step invokes the llm_fn with the instruction."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="web_search",
                     params={"query": "AI news"}, description="Search"),
            PlanStep(step=2, tool="process_text",
                     params={"instruction": "Summarize: {step_1_output}"},
                     description="Summarize results", depends_on=1),
        ]

        mock_search = AsyncMock(return_value=ToolResult(
            success=True, output="Article about AI breakthroughs"))
        mock_llm = AsyncMock(return_value="AI is advancing rapidly.")

        executor = PlanExecutor()
        with patch("bantz.agent.executor.registry") as mock_reg:
            mock_reg.get = MagicMock(
                side_effect=lambda n: MagicMock(execute=mock_search) if n == "web_search" else None
            )
            result = await executor.run(steps, llm_fn=mock_llm)

        assert result.all_success is True
        assert result.step_results[1].tool == "process_text"
        assert result.step_results[1].output == "AI is advancing rapidly."
        # LLM was called with messages containing the instruction
        mock_llm.assert_called_once()
        call_args = mock_llm.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert "summarize" in call_args[1]["content"].lower()

    @pytest.mark.asyncio
    async def test_process_text_no_llm_fn(self):
        """process_text without llm_fn → graceful failure."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="process_text",
                     params={"instruction": "Summarize something"},
                     description="Summarize"),
        ]

        executor = PlanExecutor()
        result = await executor.run(steps)  # no llm_fn

        assert result.succeeded == 0
        assert "No LLM function" in result.step_results[0].error

    @pytest.mark.asyncio
    async def test_process_text_no_instruction(self):
        """process_text without 'instruction' param → graceful failure."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="process_text",
                     params={}, description="Summarize"),
        ]

        mock_llm = AsyncMock(return_value="should not be called")
        executor = PlanExecutor()
        result = await executor.run(steps, llm_fn=mock_llm)

        assert result.succeeded == 0
        assert "instruction" in result.step_results[0].error
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_text_llm_exception(self):
        """LLM raises an exception → process_text fails gracefully."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="process_text",
                     params={"instruction": "Summarize"},
                     description="Summarize"),
        ]

        mock_llm = AsyncMock(side_effect=RuntimeError("LLM offline"))
        executor = PlanExecutor()
        result = await executor.run(steps, llm_fn=mock_llm)

        assert result.succeeded == 0
        assert "LLM offline" in result.step_results[0].error

    @pytest.mark.asyncio
    async def test_process_text_context_passed_to_next_step(self):
        """Output from process_text is stored and injected into the next step."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="process_text",
                     params={"instruction": "Say hello"},
                     description="Generate text"),
            PlanStep(step=2, tool="filesystem",
                     params={"action": "write", "path": "~/out.txt",
                             "content": "{step_1_output}"},
                     description="Save output", depends_on=1),
        ]

        captured_params: dict = {}

        async def mock_fs_execute(**kwargs):
            captured_params.update(kwargs)
            return ToolResult(success=True, output="File written")

        mock_llm = AsyncMock(return_value="Hello from the LLM!")

        executor = PlanExecutor()
        with patch("bantz.agent.executor.registry") as mock_reg:
            mock_fs = MagicMock()
            mock_fs.execute = mock_fs_execute
            mock_reg.get = MagicMock(return_value=mock_fs)

            result = await executor.run(steps, llm_fn=mock_llm)

        assert result.all_success is True
        assert captured_params["content"] == "Hello from the LLM!"

    @pytest.mark.asyncio
    async def test_three_step_search_summarize_save(self):
        """Full 3-step flow: web_search → process_text → filesystem."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="web_search",
                     params={"query": "quantum computing"},
                     description="Search for articles"),
            PlanStep(step=2, tool="process_text",
                     params={"instruction": "Summarize: {step_1_output}"},
                     description="Summarize search results", depends_on=1),
            PlanStep(step=3, tool="filesystem",
                     params={"action": "write", "path": "~/summary.txt",
                             "content": "{step_2_output}"},
                     description="Save summary to file", depends_on=2),
        ]

        search_output = "Quantum computing uses qubits for parallel processing..."
        summary_output = "TL;DR: Quantum computing leverages qubits."
        file_params: dict = {}

        mock_search = AsyncMock(return_value=ToolResult(
            success=True, output=search_output))
        mock_llm = AsyncMock(return_value=summary_output)

        async def mock_fs_execute(**kwargs):
            file_params.update(kwargs)
            return ToolResult(success=True, output="File written")

        executor = PlanExecutor()
        with patch("bantz.agent.executor.registry") as mock_reg:
            mock_fs = MagicMock()
            mock_fs.execute = mock_fs_execute
            def _get(name):
                if name == "web_search":
                    return MagicMock(execute=mock_search)
                if name == "filesystem":
                    return mock_fs
                return None
            mock_reg.get = _get

            result = await executor.run(steps, llm_fn=mock_llm)

        assert result.all_success is True
        assert result.succeeded == 3
        # process_text got the search output in its instruction
        llm_call_msgs = mock_llm.call_args[0][0]
        assert search_output[:100] in llm_call_msgs[1]["content"]
        # filesystem got the LLM summary
        assert file_params["content"] == summary_output

    @pytest.mark.asyncio
    async def test_four_step_deep_research_flow(self):
        """Full 4-step: web_search → read_url → process_text → filesystem."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="web_search",
                     params={"query": "quantum computing"},
                     description="Search for articles"),
            PlanStep(step=2, tool="read_url",
                     params={"url": "https://example.com/quantum"},
                     description="Read full article", depends_on=1),
            PlanStep(step=3, tool="process_text",
                     params={"instruction": "Summarize: {step_2_output}"},
                     description="Summarize the article", depends_on=2),
            PlanStep(step=4, tool="filesystem",
                     params={"action": "write", "path": "~/summary.txt",
                             "content": "{step_3_output}"},
                     description="Save summary", depends_on=3),
        ]

        article_text = "Full article about quantum computing...\n\nTelegraph Reference: https://example.com/quantum"
        summary_output = "Summary of quantum computing.\n\nTelegraph Reference: https://example.com/quantum"
        file_params: dict = {}

        mock_search = AsyncMock(return_value=ToolResult(
            success=True, output="1. Quantum Article\n   URL: https://example.com/quantum"))
        mock_reader = AsyncMock(return_value=ToolResult(
            success=True, output=article_text))
        mock_llm = AsyncMock(return_value=summary_output)

        async def mock_fs_execute(**kwargs):
            file_params.update(kwargs)
            return ToolResult(success=True, output="File written")

        executor = PlanExecutor()
        with patch("bantz.agent.executor.registry") as mock_reg:
            mock_fs = MagicMock()
            mock_fs.execute = mock_fs_execute
            def _get(name):
                if name == "web_search":
                    return MagicMock(execute=mock_search)
                if name == "read_url":
                    return MagicMock(execute=mock_reader)
                if name == "filesystem":
                    return mock_fs
                return None
            mock_reg.get = _get

            result = await executor.run(steps, llm_fn=mock_llm)

        assert result.all_success is True
        assert result.succeeded == 4
        # process_text received the full article text (from read_url)
        llm_call_msgs = mock_llm.call_args[0][0]
        assert "quantum computing" in llm_call_msgs[1]["content"].lower()
        # filesystem received the summary with citation preserved
        assert "Telegraph Reference" in file_params["content"]


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Brain integration — process_text with LLM wiring
# ═══════════════════════════════════════════════════════════════════════════════


class TestBrainProcessTextIntegration:
    """Brain._execute_plan passes ollama.chat as llm_fn to executor."""

    @pytest.mark.asyncio
    async def test_execute_plan_passes_llm_fn(self):
        """_execute_plan must call plan_executor.run with llm_fn=ollama.chat."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutionResult, StepResult

        plan_steps = [
            PlanStep(step=1, tool="web_search",
                     params={"query": "AI"}, description="Search"),
            PlanStep(step=2, tool="process_text",
                     params={"instruction": "Summarize {step_1_output}"},
                     description="Summarize", depends_on=1),
        ]

        exec_result = PlanExecutionResult(step_results=[
            StepResult(step_number=1, tool="web_search",
                       description="Search", success=True, output="AI data"),
            StepResult(step_number=2, tool="process_text",
                       description="Summarize", success=True,
                       output="Summary of AI"),
        ])

        with patch("bantz.agent.planner.planner_agent") as mock_planner, \
             patch("bantz.agent.executor.plan_executor") as mock_executor, \
             patch("bantz.core.brain.data_layer") as mock_dal, \
             patch("bantz.core.brain.ollama") as mock_ollama:

            mock_planner.decompose = AsyncMock(return_value=plan_steps)
            mock_planner.format_itinerary = MagicMock(
                return_value="Itinerary...")
            mock_executor.run = AsyncMock(return_value=exec_result)
            mock_dal.conversations = MagicMock()
            mock_ollama.chat = AsyncMock(return_value="llm output")

            from bantz.core.brain import Brain
            brain = Brain.__new__(Brain)
            brain._graph_store = AsyncMock()
            brain._fire_embeddings = MagicMock()

            result = await brain._execute_plan(
                "search AI and summarize",
                "search AI and summarize",
                {},
            )

        # Verify llm_fn was passed
        mock_executor.run.assert_called_once()
        call_kwargs = mock_executor.run.call_args[1]
        assert "llm_fn" in call_kwargs
        assert call_kwargs["llm_fn"] is mock_ollama.chat

    @pytest.mark.asyncio
    async def test_tool_names_include_process_text(self):
        """Brain._execute_plan adds 'process_text' to tool_names."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutionResult

        with patch("bantz.agent.planner.planner_agent") as mock_planner, \
             patch("bantz.agent.executor.plan_executor") as mock_executor, \
             patch("bantz.core.brain.data_layer") as mock_dal, \
             patch("bantz.core.brain.ollama") as mock_ollama, \
             patch("bantz.core.brain.registry") as mock_registry:

            mock_registry.names.return_value = ["web_search", "filesystem"]
            mock_planner.decompose = AsyncMock(return_value=[
                PlanStep(step=1, tool="web_search", params={}, description="X"),
                PlanStep(step=2, tool="process_text", params={}, description="Y", depends_on=1),
            ])
            mock_planner.format_itinerary = MagicMock(return_value="...")
            mock_executor.run = AsyncMock(
                return_value=PlanExecutionResult(step_results=[]))
            mock_dal.conversations = MagicMock()
            mock_ollama.chat = AsyncMock()

            from bantz.core.brain import Brain
            brain = Brain.__new__(Brain)
            brain._graph_store = AsyncMock()
            brain._fire_embeddings = MagicMock()

            await brain._execute_plan("test", "test", {})

        # decompose must receive process_text in tool_names
        call_args = mock_planner.decompose.call_args[0]
        tool_names = call_args[1]
        assert "process_text" in tool_names
        assert "web_search" in tool_names


# ═══════════════════════════════════════════════════════════════════════════════
# 12. process_text citation preservation (#182)
# ═══════════════════════════════════════════════════════════════════════════════


class TestProcessTextCitation:
    """process_text system prompt must enforce Telegraph Reference preservation."""

    @pytest.mark.asyncio
    async def test_system_prompt_has_citation_rule(self):
        """The system prompt sent to LLM must mention Telegraph References."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="process_text",
                     params={"instruction": "Summarize this text"},
                     description="Summarize"),
        ]

        captured_messages: list = []

        async def mock_llm(msgs):
            captured_messages.extend(msgs)
            return "Summary here"

        executor = PlanExecutor()
        await executor.run(steps, llm_fn=mock_llm)

        # System prompt must contain citation rule
        sys_prompt = captured_messages[0]["content"]
        assert "Telegraph Reference" in sys_prompt
        assert "dereliction of duty" in sys_prompt
        assert "Markdown" in sys_prompt  # no markdown links rule

    @pytest.mark.asyncio
    async def test_system_prompt_forbids_markdown_links(self):
        """Citation rule must explicitly forbid Markdown link syntax."""
        from bantz.agent.planner import PlanStep
        from bantz.agent.executor import PlanExecutor

        steps = [
            PlanStep(step=1, tool="process_text",
                     params={"instruction": "Summarize"},
                     description="Summarize"),
        ]

        captured_messages: list = []

        async def mock_llm(msgs):
            captured_messages.extend(msgs)
            return "Summary"

        executor = PlanExecutor()
        await executor.run(steps, llm_fn=mock_llm)

        sys_prompt = captured_messages[0]["content"]
        assert "[text](url)" in sys_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# 13. read_url in _TOOL_KEYWORDS
# ═══════════════════════════════════════════════════════════════════════════════


class TestReadUrlKeywords:
    """read_url keywords for complexity detection."""

    def _agent(self):
        from bantz.agent.planner import PlannerAgent
        return PlannerAgent()

    def test_read_url_keywords_in_dict(self):
        from bantz.agent.planner import _TOOL_KEYWORDS
        assert "read_url" in _TOOL_KEYWORDS
        assert any("read" in kw for kw in _TOOL_KEYWORDS["read_url"])

    def test_open_link_and_save_is_complex(self):
        """'open link and save' → complex (read_url + filesystem)."""
        assert self._agent().is_complex(
            "open link https://example.com and save the full article to a file"
        ) is True
