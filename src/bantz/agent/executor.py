"""
Bantz v3 — Plan-and-Solve Executor (#187)

"The Butler Carries Out His Itinerary"

Executes a list of PlanSteps sequentially, passing context from one step
to the next.  If a step fails, the executor notes it and continues with
remaining steps (graceful degradation).

Usage:
    from bantz.agent.executor import plan_executor
    result = await plan_executor.run(steps, tool_registry)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from bantz.tools import registry, ToolResult

log = logging.getLogger("bantz.executor")


@dataclass
class StepResult:
    """Result of executing a single plan step."""
    step_number: int
    tool: str
    description: str
    success: bool
    output: str
    error: str = ""


@dataclass
class PlanExecutionResult:
    """Aggregate result of executing an entire plan."""
    step_results: list[StepResult] = field(default_factory=list)
    aborted: bool = False

    @property
    def succeeded(self) -> int:
        return sum(1 for s in self.step_results if s.success)

    @property
    def total(self) -> int:
        return len(self.step_results)

    @property
    def all_success(self) -> bool:
        return self.succeeded == self.total and self.total > 0

    def summary(self) -> str:
        """A butler-style completion summary."""
        if not self.step_results:
            return "I'm afraid the itinerary was empty — nothing to do."

        lines: list[str] = []
        for sr in self.step_results:
            icon = "✓" if sr.success else "✗"
            short = sr.output[:200].replace("\n", " ") if sr.output else sr.error[:200]
            lines.append(f"  {icon} Step {sr.step_number}: {sr.description}")
            if short:
                lines.append(f"    → {short}")

        ok, total = self.succeeded, self.total
        if ok == total:
            header = f"Very good — all {total} tasks completed successfully."
        elif ok == 0:
            header = f"I regret to report that all {total} tasks encountered difficulties."
        else:
            header = f"I have completed {ok} of {total} tasks. Some required improvisation."

        return header + "\n" + "\n".join(lines)


class PlanExecutor:
    """Runs plan steps sequentially, threading context between them."""

    async def run(
        self,
        steps: list[Any],  # list[PlanStep] from planner.py
        *,
        on_step_start: Any = None,  # optional callback(step_number, description)
    ) -> PlanExecutionResult:
        """Execute all steps in order.

        - Output from step N is stored and injected into step N+1 when
          step N+1 declares ``depends_on: N``.
        - If a step fails, execution continues with remaining steps.
        - ``on_step_start`` is an optional async callback for progress updates.
        """
        result = PlanExecutionResult()
        context_store: dict[int, str] = {}  # step_number → output text

        for plan_step in steps:
            step_num = plan_step.step
            tool_name = plan_step.tool
            params = dict(plan_step.params)  # shallow copy
            description = plan_step.description
            depends = plan_step.depends_on

            # Progress callback
            if on_step_start is not None:
                try:
                    await on_step_start(step_num, description)
                except Exception:
                    pass

            # Inject dependency context
            if depends is not None and depends in context_store:
                prev_output = context_store[depends]
                params = self._inject_context(params, prev_output)

            # Look up tool
            tool = registry.get(tool_name)
            if not tool:
                sr = StepResult(
                    step_number=step_num,
                    tool=tool_name,
                    description=description,
                    success=False,
                    output="",
                    error=f"Tool '{tool_name}' not found in registry.",
                )
                result.step_results.append(sr)
                log.warning("Plan step %d: tool '%s' not found", step_num, tool_name)
                continue

            # Execute
            try:
                tool_result: ToolResult = await tool.execute(**params)
                sr = StepResult(
                    step_number=step_num,
                    tool=tool_name,
                    description=description,
                    success=tool_result.success,
                    output=tool_result.output,
                    error=tool_result.error,
                )
                if tool_result.success:
                    context_store[step_num] = tool_result.output[:2000]
                    log.info("Plan step %d [%s]: success", step_num, tool_name)
                else:
                    log.warning("Plan step %d [%s]: failed — %s",
                                step_num, tool_name, tool_result.error)
            except Exception as exc:
                sr = StepResult(
                    step_number=step_num,
                    tool=tool_name,
                    description=description,
                    success=False,
                    output="",
                    error=str(exc),
                )
                log.warning("Plan step %d [%s]: exception — %s",
                            step_num, tool_name, exc)

            result.step_results.append(sr)

        return result

    @staticmethod
    def _inject_context(params: dict, prev_output: str) -> dict:
        """Replace {step_N_output} placeholders and add context key."""
        # Replace any placeholder references in string values
        for key, val in params.items():
            if isinstance(val, str) and re.search(r"\{step_\d+_output\}", val):
                params[key] = re.sub(r"\{step_\d+_output\}", prev_output[:1500], val)

        # Also provide as explicit "content" for filesystem writes
        # if content is a placeholder or empty and we have prior output
        if "content" in params:
            c = params["content"]
            if not c or (isinstance(c, str) and "{step_" in c):
                params["content"] = prev_output[:2000]

        return params


# ── Singleton ────────────────────────────────────────────────────────────────

plan_executor = PlanExecutor()
