"""
Bantz — Delegate Task Tool (#321)

Allows the Brain orchestrator to delegate sub-tasks to specialised
sub-agents (Researcher, Developer, Reviewer) via the AgentManager.

This tool is registered in the global ToolRegistry and is available
to the CoT router and planner for multi-step task decomposition.

Usage by the LLM:
    {"tool": "delegate_task", "args": {
        "agent_role": "researcher",
        "task_description": "Find the current GDP of Turkey",
        "context": {"year": 2025}
    }}
"""
from __future__ import annotations

import json
import logging
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tools.delegate")


class DelegateTaskTool(BaseTool):
    """Delegate a sub-task to a specialised agent.

    The orchestrator (Brain) can use this tool to spin up a sub-agent
    with its own system prompt and restricted tool access, wait for
    results, and incorporate the findings.

    Parameters:
        agent_role: "researcher" | "developer" | "reviewer" (or alias)
        task_description: Natural language description of what to do
        context: Optional dict with extra context from the orchestrator

    Returns:
        ToolResult with the agent's summary and structured data.
    """

    name = "delegate_task"
    description = (
        "Delegate a sub-task to a specialised agent. "
        "Roles: researcher (web search, info gathering), "
        "developer (code, shell, files), "
        "reviewer (validation, quality check). "
        "Use when a task requires focused expertise that a specialist would handle better."
    )
    risk_level = "safe"

    async def execute(
        self,
        agent_role: str = "",
        task_description: str = "",
        context: dict[str, Any] | str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        # ── Validation ────────────────────────────────────────────
        if not agent_role:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: agent_role",
            )
        if not task_description:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: task_description",
            )

        # Parse context if it's a JSON string (LLMs sometimes serialise)
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except (json.JSONDecodeError, ValueError):
                context = {"raw": context}

        context = context or {}

        # ── Delegate via AgentManager ─────────────────────────────
        from bantz.agent.agent_manager import agent_manager

        if not agent_manager.enabled:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Multi-agent system is disabled. "
                    "Set BANTZ_MULTI_AGENT_ENABLED=true in your .env file."
                ),
            )

        result = await agent_manager.delegate(
            role=agent_role,
            task=task_description,
            context=context,
        )

        if not result.success:
            return ToolResult(
                success=False,
                output=result.error,
                error=result.error,
            )

        # ── Build output ──────────────────────────────────────────
        output_parts = [result.summary]
        if result.tools_used:
            output_parts.append(
                f"\n[Tools used: {', '.join(result.tools_used)}]"
            )

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            data=result.data,
        )


# ── Register ──────────────────────────────────────────────────────────────

registry.register(DelegateTaskTool())
