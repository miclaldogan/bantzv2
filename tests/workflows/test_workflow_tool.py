"""
Tests for bantz.tools.workflow_tool — the agent-facing run_workflow tool.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from pathlib import Path

from bantz.tools.workflow_tool import WorkflowTool
from bantz.workflows.models import WorkflowDef, StepResult, WorkflowResult


@pytest.fixture
def tool():
    return WorkflowTool()


SAMPLE_YAML = """\
name: test-wf
description: "Test workflow"
steps:
  - name: s1
    action: set_variable
    variable: x
    value: "42"
"""


class TestListAction:
    @pytest.mark.asyncio
    async def test_list_empty(self, tool):
        with patch("bantz.tools.workflow_tool.workflow_registry") as mock_reg:
            mock_reg.list_all.return_value = []
            result = await tool.execute(action="list")
        assert result.success is True
        assert "No workflows" in result.output

    @pytest.mark.asyncio
    async def test_list_with_workflows(self, tool):
        with patch("bantz.tools.workflow_tool.workflow_registry") as mock_reg:
            mock_reg.list_all.return_value = [
                {"name": "morning-briefing", "description": "Morning news", "inputs": {}, "steps": 4, "version": "1.0"},
            ]
            result = await tool.execute(action="list")
        assert result.success is True
        assert "morning-briefing" in result.output
        assert result.data["workflows"]


class TestRunAction:
    @pytest.mark.asyncio
    async def test_run_missing_name(self, tool):
        result = await tool.execute(action="run")
        assert result.success is False
        assert "Missing 'name'" in result.error

    @pytest.mark.asyncio
    async def test_run_success(self, tool):
        wf = WorkflowDef(
            name="test-wf",
            steps=[{"name": "s1", "action": "set_variable", "variable": "x", "value": "1"}],
        )
        wf_result = WorkflowResult(
            workflow_name="test-wf",
            success=True,
            steps=[StepResult(step_name="s1", success=True, output="Set x = 1", duration_ms=5)],
            final_output="Set x = 1",
            total_duration_ms=5,
            variables={"x": "1"},
        )
        with patch("bantz.tools.workflow_tool.workflow_registry") as mock_reg, \
             patch("bantz.tools.workflow_tool._runner") as mock_runner:
            mock_reg.get.return_value = wf
            mock_runner.run = AsyncMock(return_value=wf_result)
            result = await tool.execute(action="run", name="test-wf")
        assert result.success is True
        assert "Set x = 1" in result.output
        assert "✓" in result.output

    @pytest.mark.asyncio
    async def test_run_failure(self, tool):
        wf = WorkflowDef(
            name="test-wf",
            steps=[{"name": "s1", "action": "set_variable", "variable": "x", "value": "1"}],
        )
        wf_result = WorkflowResult(
            workflow_name="test-wf",
            success=False,
            steps=[StepResult(step_name="s1", success=False, error="boom")],
            error="Step 's1': boom",
            total_duration_ms=5,
        )
        with patch("bantz.tools.workflow_tool.workflow_registry") as mock_reg, \
             patch("bantz.tools.workflow_tool._runner") as mock_runner:
            mock_reg.get.return_value = wf
            mock_runner.run = AsyncMock(return_value=wf_result)
            result = await tool.execute(action="run", name="test-wf")
        assert result.success is False
        assert "failed" in result.output

    @pytest.mark.asyncio
    async def test_run_not_found(self, tool):
        from bantz.workflows.errors import WorkflowNotFoundError
        with patch("bantz.tools.workflow_tool.workflow_registry") as mock_reg:
            mock_reg.get.side_effect = WorkflowNotFoundError("nope")
            result = await tool.execute(action="run", name="missing")
        assert result.success is False
        assert "nope" in result.error

    @pytest.mark.asyncio
    async def test_run_with_json_string_inputs(self, tool):
        wf = WorkflowDef(
            name="test-wf",
            steps=[{"name": "s1", "action": "set_variable", "variable": "x", "value": "42"}],
        )
        wf_result = WorkflowResult(
            workflow_name="test-wf", success=True,
            steps=[StepResult(step_name="s1", success=True, output="done", duration_ms=1)],
            final_output="done", total_duration_ms=1,
        )
        with patch("bantz.tools.workflow_tool.workflow_registry") as mock_reg, \
             patch("bantz.tools.workflow_tool._runner") as mock_runner:
            mock_reg.get.return_value = wf
            mock_runner.run = AsyncMock(return_value=wf_result)
            result = await tool.execute(action="run", name="test-wf", inputs='{"city": "NYC"}')
        assert result.success is True


class TestCreateAction:
    @pytest.mark.asyncio
    async def test_create_missing_params(self, tool):
        result = await tool.execute(action="create")
        assert result.success is False
        assert "requires" in result.error

    @pytest.mark.asyncio
    async def test_create_valid(self, tool, tmp_path):
        with patch("bantz.tools.workflow_tool.workflow_registry") as mock_reg, \
             patch("bantz.tools.workflow_tool.Path") as mock_path_cls:
            mock_reg.parse_yaml.return_value = WorkflowDef(
                name="test-wf",
                steps=[{"name": "s1", "action": "set_variable", "variable": "x", "value": "1"}],
            )
            # Mock config import to use tmp_path
            with patch("bantz.config.config") as mock_cfg:
                mock_cfg.workflows_dir = str(tmp_path)
                result = await tool.execute(action="create", name="test-wf", yaml_content=SAMPLE_YAML)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_create_invalid_yaml(self, tool):
        from bantz.workflows.errors import WorkflowValidationError
        with patch("bantz.tools.workflow_tool.workflow_registry") as mock_reg:
            mock_reg.parse_yaml.side_effect = WorkflowValidationError("bad yaml")
            result = await tool.execute(action="create", name="bad", yaml_content="invalid: [")
        assert result.success is False


class TestUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        result = await tool.execute(action="bogus")
        assert result.success is False
        assert "Unknown action" in result.error


class TestRegistration:
    def test_tool_registered(self):
        from bantz.tools import registry
        t = registry.get("run_workflow")
        assert t is not None
        assert t.name == "run_workflow"
