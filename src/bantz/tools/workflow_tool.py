"""
Bantz Tool — ``run_workflow`` (#323)

Lets the agent execute a predefined YAML workflow by name, or list
available workflows.

Actions:
  - ``run``   : execute a named workflow with optional inputs
  - ``list``  : list all available workflows
  - ``create``: save a new workflow YAML (agent-generated)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


from bantz.tools import BaseTool, ToolResult, registry
from bantz.workflows import (
    WorkflowRunner,
    workflow_registry,
    WorkflowError,
    WorkflowNotFoundError,
)

log = logging.getLogger(__name__)

_runner = WorkflowRunner()


class WorkflowTool(BaseTool):
    name = "run_workflow"
    description = (
        "Execute a predefined YAML workflow by name, list available workflows, "
        "or create a new workflow. "
        "action=run name=<workflow_name> inputs={...} to execute. "
        "action=list to see available workflows. "
        "action=create name=<name> yaml_content=<raw YAML string> to save a new workflow."
    )
    risk_level = "moderate"

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "run")
        try:
            if action == "list":
                return self._list_workflows()
            elif action == "run":
                return await self._run_workflow(kwargs)
            elif action == "create":
                return self._create_workflow(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown action '{action}'. Use: run, list, create.",
                )
        except WorkflowNotFoundError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except WorkflowError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except Exception as exc:
            log.exception("Workflow tool error")
            return ToolResult(success=False, output="", error=str(exc))

    # ── action handlers ───────────────────────────────────────────────────

    def _list_workflows(self) -> ToolResult:
        workflows = workflow_registry.list_all()
        if not workflows:
            return ToolResult(
                success=True,
                output="No workflows available. Create one with action=create.",
                data={"workflows": []},
            )
        lines = []
        for wf in workflows:
            desc = f" — {wf['description']}" if wf.get("description") else ""
            inputs_str = ""
            if wf.get("inputs"):
                input_names = ", ".join(wf["inputs"].keys())
                inputs_str = f" (inputs: {input_names})"
            lines.append(f"• **{wf['name']}** v{wf['version']}{desc}{inputs_str} [{wf['steps']} steps]")
        output = "**Available Workflows:**\n" + "\n".join(lines)
        return ToolResult(success=True, output=output, data={"workflows": workflows})

    async def _run_workflow(self, kwargs: dict[str, Any]) -> ToolResult:
        name = kwargs.get("name", "")
        if not name:
            return ToolResult(
                success=False,
                output="",
                error="Missing 'name' parameter. Use action=list to see available workflows.",
            )
        wf = workflow_registry.get(name)
        inputs = kwargs.get("inputs", {})
        if isinstance(inputs, str):
            # LLM might send JSON as string
            import json
            try:
                inputs = json.loads(inputs)
            except json.JSONDecodeError:
                inputs = {}

        result = await _runner.run(wf, inputs)
        if result.success:
            output = result.final_output or "Workflow completed successfully."
            # Append step summary
            step_summary = []
            for sr in result.steps:
                status = "✓" if sr.success else "✗"
                step_summary.append(f"  {status} {sr.step_name} ({sr.duration_ms:.0f}ms)")
            output += "\n\n**Steps:**\n" + "\n".join(step_summary)
            output += f"\n\nTotal: {result.total_duration_ms:.0f}ms"
        else:
            output = f"Workflow '{name}' failed: {result.error}"
            if result.steps:
                step_summary = []
                for sr in result.steps:
                    status = "✓" if sr.success else "✗"
                    step_summary.append(f"  {status} {sr.step_name}")
                output += "\n\n**Steps:**\n" + "\n".join(step_summary)

        return ToolResult(
            success=result.success,
            output=output,
            data={
                "workflow": name,
                "steps": [
                    {"name": s.step_name, "success": s.success, "duration_ms": s.duration_ms}
                    for s in result.steps
                ],
                "variables": result.variables,
                "total_ms": result.total_duration_ms,
            },
            error=result.error if not result.success else "",
        )

    def _create_workflow(self, kwargs: dict[str, Any]) -> ToolResult:
        name = kwargs.get("name", "")
        yaml_content = kwargs.get("yaml_content", "")
        if not name or not yaml_content:
            return ToolResult(
                success=False,
                output="",
                error="action=create requires 'name' and 'yaml_content' parameters.",
            )
        # Validate by parsing
        wf = workflow_registry.parse_yaml(yaml_content, source=f"<agent:{name}>")

        # Save to user workflow dir
        try:
            from bantz.config import config
            wf_dir = Path(config.workflows_dir) if config.workflows_dir else None
            if wf_dir is None:
                wf_dir = config.db_path.parent / "workflows"
            wf_dir.mkdir(parents=True, exist_ok=True)
            path = wf_dir / f"{name}.yaml"
            path.write_text(yaml_content, encoding="utf-8")
            workflow_registry.register(wf)
            return ToolResult(
                success=True,
                output=f"Workflow '{wf.name}' saved to {path} and registered.",
                data={"path": str(path), "name": wf.name},
            )
        except Exception as exc:
            # Still register in memory even if file save fails
            workflow_registry.register(wf)
            return ToolResult(
                success=True,
                output=f"Workflow '{wf.name}' registered in memory (file save failed: {exc}).",
                data={"name": wf.name, "in_memory_only": True},
            )


registry.register(WorkflowTool())
