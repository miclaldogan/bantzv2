"""
Tests for bantz.workflows.models — Pydantic schema validation.
"""
from __future__ import annotations

import pytest

from bantz.workflows.models import (
    InputDef,
    RetryPolicy,
    StepDef,
    StepResult,
    TriggerDef,
    WorkflowDef,
    WorkflowResult,
)


# ── RetryPolicy ──────────────────────────────────────────────────────────────

class TestRetryPolicy:
    def test_defaults(self):
        rp = RetryPolicy()
        assert rp.max_retries == 0
        assert rp.delay_seconds == 1.0

    def test_custom_values(self):
        rp = RetryPolicy(max_retries=3, delay_seconds=2.5)
        assert rp.max_retries == 3
        assert rp.delay_seconds == 2.5

    def test_max_retries_ceiling(self):
        with pytest.raises(Exception):
            RetryPolicy(max_retries=11)


# ── StepDef ──────────────────────────────────────────────────────────────────

class TestStepDef:
    def test_tool_step_valid(self):
        s = StepDef(name="fetch", action="tool", tool="weather", args={"city": "NYC"})
        assert s.name == "fetch"
        assert s.tool == "weather"

    def test_tool_step_missing_tool_field(self):
        with pytest.raises(Exception, match="requires 'tool' field"):
            StepDef(name="bad", action="tool")

    def test_shell_step_valid(self):
        s = StepDef(name="run", action="shell_command", command="ls -la")
        assert s.command == "ls -la"

    def test_shell_step_missing_command(self):
        with pytest.raises(Exception, match="requires 'command' field"):
            StepDef(name="bad", action="shell_command")

    def test_http_step_valid(self):
        s = StepDef(name="req", action="http_request", url="https://example.com")
        assert s.method == "GET"

    def test_http_step_missing_url(self):
        with pytest.raises(Exception, match="requires 'url' field"):
            StepDef(name="bad", action="http_request")

    def test_ask_llm_valid(self):
        s = StepDef(name="llm", action="ask_llm", prompt="Summarize this")
        assert s.prompt == "Summarize this"

    def test_ask_llm_missing_prompt(self):
        with pytest.raises(Exception, match="requires 'prompt' field"):
            StepDef(name="bad", action="ask_llm")

    def test_conditional_valid(self):
        s = StepDef(
            name="branch", action="conditional",
            condition="true", then_step="a", else_step="b",
        )
        assert s.condition == "true"

    def test_conditional_missing_condition(self):
        with pytest.raises(Exception, match="requires 'condition' field"):
            StepDef(name="bad", action="conditional")

    def test_set_variable_valid(self):
        s = StepDef(name="setvar", action="set_variable", variable="x", value="42")
        assert s.variable == "x"

    def test_set_variable_missing_variable(self):
        with pytest.raises(Exception, match="requires 'variable' field"):
            StepDef(name="bad", action="set_variable")

    def test_timeout_defaults(self):
        s = StepDef(name="t", action="tool", tool="weather")
        assert s.timeout_seconds == 30.0

    def test_retry_defaults(self):
        s = StepDef(name="t", action="tool", tool="weather")
        assert s.retry.max_retries == 0

    def test_depends_on(self):
        s = StepDef(
            name="t", action="tool", tool="weather",
            depends_on=["step1", "step2"],
        )
        assert s.depends_on == ["step1", "step2"]


# ── WorkflowDef ──────────────────────────────────────────────────────────────

class TestWorkflowDef:
    def _minimal(self, **overrides):
        defaults = {
            "name": "test-wf",
            "steps": [{"name": "s1", "action": "tool", "tool": "weather"}],
        }
        defaults.update(overrides)
        return WorkflowDef(**defaults)

    def test_minimal_valid(self):
        wf = self._minimal()
        assert wf.name == "test-wf"
        assert len(wf.steps) == 1

    def test_defaults(self):
        wf = self._minimal()
        assert wf.version == "1.0"
        assert wf.on_failure == "abort"
        assert wf.timeout_seconds == 300.0

    def test_no_steps_raises(self):
        with pytest.raises(Exception):
            WorkflowDef(name="empty", steps=[])

    def test_duplicate_step_names_raises(self):
        with pytest.raises(Exception, match="Duplicate step names"):
            WorkflowDef(
                name="dupes",
                steps=[
                    {"name": "a", "action": "tool", "tool": "weather"},
                    {"name": "a", "action": "tool", "tool": "news"},
                ],
            )

    def test_depends_on_unknown_step_raises(self):
        with pytest.raises(Exception, match="depends_on unknown step"):
            WorkflowDef(
                name="bad-dep",
                steps=[
                    {"name": "a", "action": "tool", "tool": "weather", "depends_on": ["b"]},
                ],
            )

    def test_depends_on_valid(self):
        wf = WorkflowDef(
            name="good-dep",
            steps=[
                {"name": "a", "action": "tool", "tool": "weather"},
                {"name": "b", "action": "tool", "tool": "news", "depends_on": ["a"]},
            ],
        )
        assert wf.steps[1].depends_on == ["a"]

    def test_inputs(self):
        wf = self._minimal(inputs={"city": {"type": "string", "default": "NYC"}})
        assert "city" in wf.inputs
        assert wf.inputs["city"].default == "NYC"

    def test_triggers(self):
        wf = self._minimal(triggers=[{"schedule": "08:00"}, {"manual": True}])
        assert len(wf.triggers) == 2

    def test_on_failure_modes(self):
        for mode in ("abort", "ask_llm", "continue"):
            wf = self._minimal(on_failure=mode)
            assert wf.on_failure == mode


# ── Runtime result types ─────────────────────────────────────────────────────

class TestStepResult:
    def test_defaults(self):
        sr = StepResult(step_name="s1", success=True)
        assert sr.output == ""
        assert sr.error == ""
        assert sr.data == {}
        assert sr.duration_ms == 0.0


class TestWorkflowResult:
    def test_defaults(self):
        wr = WorkflowResult(workflow_name="wf", success=True)
        assert wr.steps == []
        assert wr.final_output == ""
        assert wr.variables == {}


# ── InputDef, TriggerDef ─────────────────────────────────────────────────────

class TestInputDef:
    def test_defaults(self):
        i = InputDef()
        assert i.type == "string"
        assert i.required is False


class TestTriggerDef:
    def test_schedule(self):
        t = TriggerDef(schedule="08:00")
        assert t.schedule == "08:00"

    def test_manual(self):
        t = TriggerDef(manual=True)
        assert t.manual is True

    def test_event(self):
        t = TriggerDef(event="on_wake")
        assert t.event == "on_wake"
