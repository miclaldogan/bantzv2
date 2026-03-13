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
from typing import Any, Callable, Awaitable

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
        llm_fn: Callable[..., Awaitable[str]] | None = None,
    ) -> PlanExecutionResult:
        """Execute all steps in order.

        - Output from step N is stored and injected into step N+1 when
          step N+1 declares ``depends_on: N``.
        - If a step fails, execution continues with remaining steps.
        - ``on_step_start`` is an optional async callback for progress updates.
        - ``llm_fn`` is an async callable used by the virtual ``process_text``
          tool.  Signature: ``await llm_fn(messages) -> str``.
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

            # ── Virtual tool: process_text ──────────────────────────────
            if tool_name == "process_text":
                sr = await self._handle_process_text(
                    step_num, params, description, llm_fn,
                )
                if sr.success:
                    context_store[step_num] = sr.output[:2000]
                result.step_results.append(sr)
                continue

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

    # ── Virtual tool: process_text ───────────────────────────────────────

    @staticmethod
    async def _handle_process_text(
        step_num: int,
        params: dict,
        description: str,
        llm_fn: Callable[..., Awaitable[str]] | None,
    ) -> StepResult:
        """Route *process_text* to the LLM instead of the tool registry."""
        instruction = params.get("instruction", "")
        if not instruction:
            return StepResult(
                step_number=step_num,
                tool="process_text",
                description=description,
                success=False,
                output="",
                error="process_text requires an 'instruction' param.",
            )
        if llm_fn is None:
            return StepResult(
                step_number=step_num,
                tool="process_text",
                description=description,
                success=False,
                output="",
                error="No LLM function provided for process_text.",
            )
        try:
            messages = [
                {"role": "system", "content": (
                    "You are a helpful text-processing assistant. "
                    "Follow the user's instruction precisely.\n\n"
                    "CRITICAL: If the input text contains 'Telegraph References' or URLs, "
                    "you MUST preserve them. ALWAYS append them at the very bottom of your "
                    "output as raw, unformatted links (e.g., Telegraph Reference: https...). "
                    "Do NOT use Markdown links [text](url). "
                    "Omitting the source link is a dereliction of duty."
                )},
                {"role": "user", "content": instruction},
            ]
            llm_output = await llm_fn(messages)
            log.info("Plan step %d [process_text]: success", step_num)
            return StepResult(
                step_number=step_num,
                tool="process_text",
                description=description,
                success=True,
                output=llm_output,
            )
        except Exception as exc:
            log.warning("Plan step %d [process_text]: exception — %s", step_num, exc)
            return StepResult(
                step_number=step_num,
                tool="process_text",
                description=description,
                success=False,
                output="",
                error=str(exc),
            )

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
