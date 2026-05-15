"""
Tests for bantz.workflows.runner — step execution, interpolation, retry, etc.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bantz.tools import BaseTool, ToolResult
from bantz.workflows.models import (
    StepDef,
    WorkflowDef,
    RetryPolicy,
)
from bantz.workflows.runner import WorkflowRunner


# ── Helpers ──────────────────────────────────────────────────────────────────

class FakeTool(BaseTool):
    name = "fake"
    description = "A fake tool for testing"

    def __init__(self, output="ok", success=True, error="", raise_exc=None):
        self._output = output
        self._success = success
        self._error = error
        self._raise_exc = raise_exc

    async def execute(self, **kwargs):
        if self._raise_exc:
            raise self._raise_exc
        return ToolResult(
            success=self._success,
            output=self._output,
            data={"kwargs": kwargs},
            error=self._error,
        )


class SlowTool(BaseTool):
    name = "slow"
    description = "A tool that takes forever"

    async def execute(self, **kwargs):
        await asyncio.sleep(999)
        return ToolResult(success=True, output="done")


def _make_wf(steps, **kwargs):
    defaults = {"name": "test-wf"}
    defaults.update(kwargs)
    return WorkflowDef(steps=steps, **defaults)


# ── Template interpolation ───────────────────────────────────────────────────

class TestInterpolation:
    def test_simple_input(self):
        runner = WorkflowRunner()
        ctx = {"inputs": {"city": "NYC"}, "steps": {}, "variables": {}}
        result = runner._interpolate("Weather in {{ inputs.city }}", ctx)
        assert result == "Weather in NYC"

    def test_step_output(self):
        runner = WorkflowRunner()
        ctx = {
            "inputs": {},
            "steps": {"s1": {"output": "sunny", "success": True}},
            "variables": {},
        }
        result = runner._interpolate("Result: {{ steps.s1.output }}", ctx)
        assert result == "Result: sunny"

    def test_variable(self):
        runner = WorkflowRunner()
        ctx = {"inputs": {}, "steps": {}, "variables": {"x": "42"}}
        result = runner._interpolate("x = {{ variables.x }}", ctx)
        assert result == "x = 42"

    def test_unresolved_kept(self):
        runner = WorkflowRunner()
        ctx = {"inputs": {}, "steps": {}, "variables": {}}
        result = runner._interpolate("{{ inputs.missing }}", ctx)
        assert result == "{{ inputs.missing }}"

    def test_dict_interpolation(self):
        runner = WorkflowRunner()
        ctx = {"inputs": {"q": "test"}, "steps": {}, "variables": {}}
        d = {"query": "{{ inputs.q }}", "count": 5}
        result = runner._interpolate_dict(d, ctx)
        assert result == {"query": "test", "count": 5}


# ── Condition evaluation ─────────────────────────────────────────────────────

class TestEvalCondition:
    def test_equal_true(self):
        assert WorkflowRunner._eval_condition("true == true") is True

    def test_equal_false(self):
        assert WorkflowRunner._eval_condition("false == false") is True

    def test_string_equal(self):
        assert WorkflowRunner._eval_condition('hello == "hello"') is True

    def test_string_not_equal(self):
        assert WorkflowRunner._eval_condition('hello != "world"') is True

    def test_truthy(self):
        assert WorkflowRunner._eval_condition("something") is True

    def test_falsy_false(self):
        assert WorkflowRunner._eval_condition("false") is False

    def test_falsy_empty(self):
        assert WorkflowRunner._eval_condition("") is False


# ── Runner — tool steps ──────────────────────────────────────────────────────

class TestRunToolStep:
    @pytest.mark.asyncio
    async def test_single_tool_step(self):
        wf = _make_wf([{"name": "s1", "action": "tool", "tool": "fake"}])
        runner = WorkflowRunner()
        with patch("bantz.workflows.runner.WorkflowRunner._run_tool") as mock:
            from bantz.workflows.models import StepResult
            mock.return_value = StepResult(step_name="s1", success=True, output="ok")
            result = await runner.run(wf)
        assert result.success is True
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_tool_step_with_registry(self):
        fake = FakeTool(output="weather is sunny")
        wf = _make_wf([{"name": "s1", "action": "tool", "tool": "fake"}])
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = fake
            result = await runner.run(wf)
        assert result.success is True
        assert result.final_output == "weather is sunny"

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        wf = _make_wf([{"name": "s1", "action": "tool", "tool": "nonexistent"}])
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = None
            result = await runner.run(wf)
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_tool_raises_exception(self):
        fake = FakeTool(raise_exc=RuntimeError("boom"))
        wf = _make_wf([{"name": "s1", "action": "tool", "tool": "fake"}])
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = fake
            result = await runner.run(wf)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_tool_args_interpolated(self):
        fake = FakeTool(output="NYC weather")
        wf = _make_wf(
            [{"name": "s1", "action": "tool", "tool": "fake", "args": {"city": "{{ inputs.city }}"}}],
            inputs={"city": {"type": "string", "default": "NYC"}},
        )
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = fake
            result = await runner.run(wf, {"city": "Istanbul"})
        assert result.success is True


# ── Runner — shell steps ─────────────────────────────────────────────────────

class TestRunShellStep:
    @pytest.mark.asyncio
    async def test_shell_step(self):
        fake_shell = FakeTool(output="/home/user")
        wf = _make_wf([{"name": "s1", "action": "shell_command", "command": "pwd"}])
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = fake_shell
            result = await runner.run(wf)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_shell_tool_not_available(self):
        wf = _make_wf([{"name": "s1", "action": "shell_command", "command": "pwd"}])
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = None
            result = await runner.run(wf)
        assert result.success is False
        assert "not available" in result.error


# ── Runner — set_variable ────────────────────────────────────────────────────

class TestRunSetVariable:
    @pytest.mark.asyncio
    async def test_set_variable(self):
        wf = _make_wf([
            {"name": "s1", "action": "set_variable", "variable": "x", "value": "42"},
        ])
        runner = WorkflowRunner()
        result = await runner.run(wf)
        assert result.success is True
        assert result.variables.get("x") == "42"

    @pytest.mark.asyncio
    async def test_set_variable_with_interpolation(self):
        wf = _make_wf(
            [
                {"name": "s1", "action": "set_variable", "variable": "greeting", "value": "Hello {{ inputs.name }}"},
            ],
            inputs={"name": {"type": "string", "default": "World"}},
        )
        runner = WorkflowRunner()
        result = await runner.run(wf, {"name": "Bantz"})
        assert result.variables.get("greeting") == "Hello Bantz"


# ── Runner — conditional ─────────────────────────────────────────────────────

class TestRunConditional:
    @pytest.mark.asyncio
    async def test_conditional_true_path(self):
        wf = _make_wf([
            {"name": "check", "action": "conditional", "condition": "true == true", "then_step": "yes", "else_step": "no"},
            {"name": "no", "action": "set_variable", "variable": "path", "value": "wrong"},
            {"name": "yes", "action": "set_variable", "variable": "path", "value": "correct"},
        ])
        runner = WorkflowRunner()
        result = await runner.run(wf)
        assert result.success is True
        assert result.variables.get("path") == "correct"

    @pytest.mark.asyncio
    async def test_conditional_false_path(self):
        wf = _make_wf([
            {"name": "check", "action": "conditional", "condition": "false == true", "then_step": "yes", "else_step": "no"},
            {"name": "yes", "action": "set_variable", "variable": "path", "value": "wrong"},
            {"name": "no", "action": "set_variable", "variable": "path", "value": "fallback"},
        ])
        runner = WorkflowRunner()
        result = await runner.run(wf)
        assert result.success is True
        assert result.variables.get("path") == "fallback"


# ── Runner — retry logic ─────────────────────────────────────────────────────

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        call_count = 0
        class FlakyTool(BaseTool):
            name = "flaky"
            description = "fails first, succeeds second"
            async def execute(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return ToolResult(success=False, output="", error="flaky")
                return ToolResult(success=True, output="ok")

        wf = _make_wf([{
            "name": "s1", "action": "tool", "tool": "flaky",
            "retry": {"max_retries": 2, "delay_seconds": 0},
        }])
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = FlakyTool()
            result = await runner.run(wf)
        assert result.success is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        class AlwaysFail(BaseTool):
            name = "failing"
            description = "always fails"
            async def execute(self, **kwargs):
                return ToolResult(success=False, output="", error="nope")

        wf = _make_wf([{
            "name": "s1", "action": "tool", "tool": "failing",
            "retry": {"max_retries": 1, "delay_seconds": 0},
        }])
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = AlwaysFail()
            result = await runner.run(wf)
        assert result.success is False


# ── Runner — on_failure modes ────────────────────────────────────────────────

class TestOnFailure:
    @pytest.mark.asyncio
    async def test_abort_stops_on_failure(self):
        class FailTool(BaseTool):
            name = "fail"
            description = ""
            async def execute(self, **kwargs):
                return ToolResult(success=False, output="", error="fail")

        wf = _make_wf(
            [
                {"name": "s1", "action": "tool", "tool": "fail"},
                {"name": "s2", "action": "set_variable", "variable": "x", "value": "should not run"},
            ],
            on_failure="abort",
        )
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = FailTool()
            result = await runner.run(wf)
        assert result.success is False
        assert len(result.steps) == 1  # s2 never ran

    @pytest.mark.asyncio
    async def test_continue_skips_failure(self):
        class FailTool(BaseTool):
            name = "fail"
            description = ""
            async def execute(self, **kwargs):
                return ToolResult(success=False, output="", error="oops")

        wf = _make_wf(
            [
                {"name": "s1", "action": "tool", "tool": "fail"},
                {"name": "s2", "action": "set_variable", "variable": "x", "value": "ran"},
            ],
            on_failure="continue",
        )
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = FailTool()
            result = await runner.run(wf)
        assert result.success is True
        assert len(result.steps) == 2
        assert result.variables.get("x") == "ran"


# ── Runner — step timeout ────────────────────────────────────────────────────

class TestStepTimeout:
    @pytest.mark.asyncio
    async def test_step_timeout(self):
        wf = _make_wf([{
            "name": "s1", "action": "tool", "tool": "slow",
            "timeout_seconds": 1,
        }])
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = SlowTool()
            result = await runner.run(wf)
        assert result.success is False
        assert "timed out" in result.steps[0].error


# ── Runner — workflow timeout ────────────────────────────────────────────────

class TestWorkflowTimeout:
    @pytest.mark.asyncio
    async def test_workflow_timeout(self):
        wf = _make_wf(
            [{"name": "s1", "action": "tool", "tool": "slow", "timeout_seconds": 600}],
            timeout_seconds=1,
        )
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = SlowTool()
            result = await runner.run(wf)
        assert result.success is False
        assert "timed out" in result.error


# ── Runner — depends_on ──────────────────────────────────────────────────────

class TestDependsOn:
    @pytest.mark.asyncio
    async def test_context_passed_between_steps(self):
        fake = FakeTool(output="step1 output")
        wf = _make_wf([
            {"name": "s1", "action": "tool", "tool": "fake"},
            {"name": "s2", "action": "set_variable", "variable": "prev", "value": "{{ steps.s1.output }}", "depends_on": ["s1"]},
        ])
        runner = WorkflowRunner()
        with patch("bantz.tools.registry") as mock_reg:
            mock_reg.get.return_value = fake
            result = await runner.run(wf)
        assert result.success is True
        assert result.variables.get("prev") == "step1 output"


# ── Runner — multi-step sequential ──────────────────────────────────────────

class TestMultiStep:
    @pytest.mark.asyncio
    async def test_three_step_pipeline(self):
        wf = _make_wf([
            {"name": "s1", "action": "set_variable", "variable": "a", "value": "1"},
            {"name": "s2", "action": "set_variable", "variable": "b", "value": "2"},
            {"name": "s3", "action": "set_variable", "variable": "c", "value": "{{ variables.a }}-{{ variables.b }}"},
        ])
        runner = WorkflowRunner()
        result = await runner.run(wf)
        assert result.success is True
        assert result.variables == {"a": "1", "b": "2", "c": "1-2"}
        assert len(result.steps) == 3
        assert result.total_duration_ms > 0


# ── Runner — inputs ──────────────────────────────────────────────────────────

class TestInputs:
    @pytest.mark.asyncio
    async def test_default_inputs(self):
        wf = _make_wf(
            [{"name": "s1", "action": "set_variable", "variable": "city", "value": "{{ inputs.city }}"}],
            inputs={"city": {"type": "string", "default": "NYC"}},
        )
        runner = WorkflowRunner()
        result = await runner.run(wf)
        assert result.variables.get("city") == "NYC"

    @pytest.mark.asyncio
    async def test_override_inputs(self):
        wf = _make_wf(
            [{"name": "s1", "action": "set_variable", "variable": "city", "value": "{{ inputs.city }}"}],
            inputs={"city": {"type": "string", "default": "NYC"}},
        )
        runner = WorkflowRunner()
        result = await runner.run(wf, {"city": "Istanbul"})
        assert result.variables.get("city") == "Istanbul"
