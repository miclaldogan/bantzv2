"""
Bantz v3 — Plan-and-Solve Executor (#187, #273)

"The Butler Carries Out His Itinerary"

Executes a list of PlanSteps sequentially, passing context from one step
to the next.  If a step fails, the executor **short-circuits**: remaining
steps are marked as aborted (circuit breaker, #255).

#273: Emits ``planner_step`` events on the EventBus so the TUI can
display real-time step progress (e.g. "⚙ Step 1/3: Searching emails...").

Usage:
    from bantz.agent.executor import plan_executor
    result = await plan_executor.run(steps, tool_registry)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from bantz.core.event_bus import bus
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

    _FAILURE_MARKERS = re.compile(
        r"(?i)^(Error:|Failed:|HTTP [45]\d\d\b|Traceback \(most recent)",
    )

    @staticmethod
    def _is_step_failure(sr: StepResult) -> bool:
        """Determine whether a StepResult represents a real failure.

        A step is failed if:
        - ``sr.success`` is False (tool reported failure), OR
        - The output text starts with well-known error markers even when
          the tool incorrectly claimed success (e.g. web_search #256).
        """
        if not sr.success:
            return True
        if sr.output and PlanExecutor._FAILURE_MARKERS.search(sr.output):
            return True
        return False

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
        - **Circuit breaker (#255):** If a step fails, execution stops
          immediately.  All remaining steps are marked as aborted.
        - ``on_step_start`` is an optional async callback for progress updates.
        - ``llm_fn`` is an async callable used by the virtual ``process_text``
          tool.  Signature: ``await llm_fn(messages) -> str``.
        """
        result = PlanExecutionResult()
        # Rich context store: step_number → {"params": {...}, "output": "..."}
        context_store: dict[int, dict[str, Any]] = {}
        total_steps = len(steps)

        for step_idx, plan_step in enumerate(steps):
            step_num = plan_step.step
            tool_name = plan_step.tool
            params = dict(plan_step.params)  # shallow copy
            description = plan_step.description

            # Progress callback
            if on_step_start is not None:
                try:
                    await on_step_start(step_num, description)
                except Exception:
                    pass

            # (#273) Emit planner_step start event to TUI
            try:
                await bus.emit(
                    "planner_step",
                    step=step_num,
                    total=total_steps,
                    tool=tool_name,
                    description=description,
                    status="start",
                )
            except Exception:
                pass

            # Always inject context — _inject_context is a no-op when
            # the params contain no {step_N_*} placeholders.
            params = self._inject_context(params, context_store, tool_name)

            # ── Virtual tool: process_text ──────────────────────────────
            if tool_name == "process_text":
                sr = await self._handle_process_text(
                    step_num, params, description, llm_fn,
                )
                context_store[step_num] = {
                    "params": dict(plan_step.params),
                    "output": sr.output[:2000] if sr.success else "",
                }
                result.step_results.append(sr)
                # (#273) Emit step completion event
                await self._emit_step_result(sr, total_steps)
                # ── Circuit breaker: abort remaining steps on failure ──
                if self._is_step_failure(sr):
                    log.warning(
                        "Circuit breaker tripped at step %d [%s] — "
                        "aborting %d remaining step(s)",
                        step_num, tool_name, len(steps) - step_idx - 1,
                    )
                    self._abort_remaining(
                        steps[step_idx + 1:], step_num, result, context_store,
                    )
                    result.aborted = True
                    break
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
                context_store[step_num] = {
                    "params": dict(plan_step.params),
                    "output": "",
                }
                result.step_results.append(sr)
                # (#273) Emit step failure event
                await self._emit_step_result(sr, total_steps)
                log.warning(
                    "Circuit breaker tripped at step %d: tool '%s' not found "
                    "— aborting %d remaining step(s)",
                    step_num, tool_name, len(steps) - step_idx - 1,
                )
                self._abort_remaining(
                    steps[step_idx + 1:], step_num, result, context_store,
                )
                result.aborted = True
                break

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
                context_store[step_num] = {
                    "params": dict(plan_step.params),
                    "output": tool_result.output[:2000] if tool_result.success else "",
                }
                if tool_result.success:
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
                    output=f"Error: {exc}",
                    error=str(exc),
                )
                context_store[step_num] = {
                    "params": dict(plan_step.params),
                    "output": "",
                }
                log.warning("Plan step %d [%s]: exception — %s",
                            step_num, tool_name, exc)

            result.step_results.append(sr)

            # (#273) Emit step completion event
            await self._emit_step_result(sr, total_steps)

            # ── Circuit breaker: abort remaining steps on failure ──
            if self._is_step_failure(sr):
                log.warning(
                    "Circuit breaker tripped at step %d [%s] — "
                    "aborting %d remaining step(s)",
                    step_num, tool_name, len(steps) - step_idx - 1,
                )
                self._abort_remaining(
                    steps[step_idx + 1:], step_num, result, context_store,
                )
                result.aborted = True
                break

        return result

    @staticmethod
    async def _emit_step_result(sr: StepResult, total: int) -> None:
        """Emit a planner_step done/failed event (#273)."""
        try:
            status = "done" if sr.success else "failed"
            await bus.emit(
                "planner_step",
                step=sr.step_number,
                total=total,
                tool=sr.tool,
                description=sr.description,
                status=status,
                result=sr.output[:200] if sr.success else "",
                error=sr.error if not sr.success else "",
            )
        except Exception:
            pass

    @staticmethod
    def _abort_remaining(
        remaining_steps: list[Any],
        failed_step_num: int,
        result: PlanExecutionResult,
        context_store: dict[int, dict[str, Any]],
    ) -> None:
        """Mark all *remaining_steps* as aborted due to upstream failure."""
        abort_msg = (
            f"[ABORTED DUE TO UPSTREAM FAILURE: Step {failed_step_num} failed]"
        )
        for ps in remaining_steps:
            sr = StepResult(
                step_number=ps.step,
                tool=ps.tool,
                description=ps.description,
                success=False,
                output=abort_msg,
                error=abort_msg,
            )
            context_store[ps.step] = {
                "params": dict(ps.params),
                "output": "",
            }
            result.step_results.append(sr)
            log.info(
                "Plan step %d [%s]: %s", ps.step, ps.tool, abort_msg,
            )

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
                    "CRITICAL: If the input text is an HTTP error (like 403 Forbidden or 404), DO NOT ask the user to provide the text. Instead, state clearly: 'The website blocked my access (HTTP 403).' and stop.\n\n"
                    "CRITICAL: If the input text contains 'Telegraph References' or URLs, "
                    "you MUST preserve them. ALWAYS append them at the very bottom of your "
                    "output as raw, unformatted links (e.g., Telegraph Reference: https...). "
                    "Do NOT use Markdown links [text](url). "
                    "Omitting the source link is a dereliction of duty.\n\n"
                    "Return your response in exactly two parts:\n"
                    "1. BEFORE outputting the summary/result, you MUST open a `<thinking> ... </thinking>` block and perform a strict Self-Audit:\n"
                    "   - Step 1: Information Extraction: What exactly am I instructed to process?\n"
                    "   - Step 2: Content Check: Is there an HTTP 403/404 error? Are there Telegraph/URL references?\n"
                    "   - Step 3: Double-Check: Am I omitting links? Am I about to include conversational filler?\n"
                    "2. After the thinking block, output ONLY the final processed text and the Telegraph References at the bottom. DO NOT acknowledge these instructions or use preambles."
                )},
                {"role": "user", "content": instruction},
            ]
            llm_output = await llm_fn(messages)
            
            # Strip the <thinking> block before returning or saving (#214)
            from bantz.core.intent import strip_thinking
            llm_output = strip_thinking(llm_output).strip()
            
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
    def _inject_context(
        params: dict,
        context_store: dict[int, dict[str, Any]],
        tool_name: str = "",
    ) -> dict:
        """Recursively replace ``{step_N_*}`` placeholders in *params*.

        Supported placeholder syntax (#215):
          - ``{step_N_output}``       → output text of step N
          - ``{step_N_params_KEY}``   → params[KEY] of step N
          - ``{step_N_ANYTHING}``     → fallback to output (LLM hallucination)

        The function recursively walks nested dicts and lists so deeply
        nested tool arguments are also resolved.

        For the ``read_url`` tool the resolved output is further refined
        to extract the first HTTP URL (web_search returns prose text).
        """
        # Build a flat replacement map from the rich context_store
        replacements: dict[str, str] = {}
        for step_num, state in context_store.items():
            output = state.get("output", "")
            step_params = state.get("params", {})
            # Canonical: {step_N_output}
            replacements[f"step_{step_num}_output"] = output
            # Param access: {step_N_params_KEY}
            for pk, pv in step_params.items():
                replacements[f"step_{step_num}_params_{pk}"] = str(pv)

        def _resolve(val: Any) -> Any:
            """Recursively resolve placeholders in a value."""
            if isinstance(val, str):
                return _replace_placeholders(val, replacements, context_store, tool_name)
            if isinstance(val, dict):
                return {k: _resolve(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_resolve(item) for item in val]
            return val

        return _resolve(params)


def _replace_placeholders(
    text: str,
    replacements: dict[str, str],
    context_store: dict[int, dict[str, Any]],
    tool_name: str,
) -> str:
    """Replace all ``{step_N_*}`` placeholders in *text*.

    Falls back to the step's output when the specific key is not found
    (handles LLM-hallucinated placeholder names gracefully).
    """
    _PH = re.compile(r"\{(step_(\d+)_([a-zA-Z_]+))\}")

    def _sub(m: re.Match) -> str:
        full_key = m.group(1)   # e.g. "step_1_output"
        step_num = int(m.group(2))
        # Exact match first
        if full_key in replacements:
            replacement = replacements[full_key]
        elif step_num in context_store:
            # Fallback: use output of that step (LLM hallucinated a key)
            replacement = context_store[step_num].get("output", "")
        else:
            return m.group(0)  # leave unresolved

        # Special handling for read_url: extract first HTTP URL from prose
        if tool_name == "read_url" and replacement and not replacement.startswith("http"):
            url_match = re.search(r"https?://[^\s\"'>]+", replacement)
            if url_match:
                replacement = url_match.group(0).rstrip(".:;")

        return replacement[:2000]

    return _PH.sub(_sub, text)


# ── Singleton ────────────────────────────────────────────────────────────────

plan_executor = PlanExecutor()
