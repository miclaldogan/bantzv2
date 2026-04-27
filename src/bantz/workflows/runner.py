"""
Workflow Runner — state machine that executes a ``WorkflowDef`` step by step.

Handles:
  - Template variable interpolation (``{{ inputs.city }}``, ``{{ steps.X.output }}``)
  - Sequential execution with ``depends_on`` ordering
  - Retry with configurable delay
  - Step-level and workflow-level timeouts
  - ``on_failure`` strategies: abort, continue, ask_llm (LLM self-heal)
  - ``conditional`` branching
  - ``set_variable`` for context variables
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from bantz.workflows.errors import (
    StepExecutionError,
    WorkflowError,
)
from bantz.workflows.models import (
    StepDef,
    StepResult,
    WorkflowDef,
    WorkflowResult,
)

log = logging.getLogger(__name__)

# Matches {{ path.to.var }}
_TEMPLATE_RE = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


class WorkflowRunner:
    """Executes a ``WorkflowDef`` and returns a ``WorkflowResult``."""

    def __init__(self) -> None:
        # Lazily resolved at execution time
        self._tool_registry: Any = None

    # ── public API ────────────────────────────────────────────────────────

    async def run(
        self,
        workflow: WorkflowDef,
        inputs: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Execute all steps in order and return the result."""
        t0 = time.monotonic()
        ctx = self._build_context(workflow, inputs or {})
        results: list[StepResult] = []
        _step_map = {s.name: s for s in workflow.steps}
        completed: set[str] = set()
        jump_target: str | None = None

        try:
            async with asyncio.timeout(workflow.timeout_seconds):
                for step in workflow.steps:
                    # Handle conditional jumps
                    if jump_target is not None:
                        if step.name != jump_target:
                            continue
                        jump_target = None

                    # Wait for dependencies
                    for dep in step.depends_on:
                        if dep not in completed:
                            sr = self._find_result(results, dep)
                            if sr is None or not sr.success:
                                raise StepExecutionError(
                                    step.name,
                                    f"dependency '{dep}' did not complete successfully",
                                )

                    sr = await self._execute_step(step, ctx)
                    results.append(sr)
                    ctx["steps"][step.name] = {
                        "output": sr.output,
                        "data": sr.data,
                        "success": sr.success,
                        "error": sr.error,
                    }

                    if sr.success:
                        completed.add(step.name)
                    else:
                        if workflow.on_failure == "abort":
                            raise StepExecutionError(step.name, sr.error)
                        elif workflow.on_failure == "ask_llm":
                            healed = await self._llm_heal(step, sr, ctx)
                            results.append(healed)
                            ctx["steps"][step.name] = {
                                "output": healed.output,
                                "data": healed.data,
                                "success": healed.success,
                                "error": healed.error,
                            }
                            if healed.success:
                                completed.add(step.name)
                            else:
                                raise StepExecutionError(step.name, healed.error)
                        # on_failure == "continue" → skip and proceed

                    # Handle conditional jumps
                    if step.action == "conditional" and sr.success:
                        jump_target = sr.data.get("jump_to")

        except asyncio.TimeoutError:
            return WorkflowResult(
                workflow_name=workflow.name,
                success=False,
                steps=results,
                error=f"Workflow timed out after {workflow.timeout_seconds}s",
                total_duration_ms=(time.monotonic() - t0) * 1000,
                variables=ctx.get("variables", {}),
            )
        except WorkflowError as exc:
            return WorkflowResult(
                workflow_name=workflow.name,
                success=False,
                steps=results,
                error=str(exc),
                total_duration_ms=(time.monotonic() - t0) * 1000,
                variables=ctx.get("variables", {}),
            )

        # Build final output from last successful step
        final_output = ""
        for sr in reversed(results):
            if sr.success and sr.output:
                final_output = sr.output
                break

        return WorkflowResult(
            workflow_name=workflow.name,
            success=True,
            steps=results,
            final_output=final_output,
            total_duration_ms=(time.monotonic() - t0) * 1000,
            variables=ctx.get("variables", {}),
        )

    # ── step execution ────────────────────────────────────────────────────

    async def _execute_step(
        self, step: StepDef, ctx: dict[str, Any],
    ) -> StepResult:
        """Execute a single step with retry logic."""
        last_error = ""
        attempts = step.retry.max_retries + 1

        for attempt in range(attempts):
            if attempt > 0:
                await asyncio.sleep(step.retry.delay_seconds)
                log.info(
                    "Retrying step '%s' (attempt %d/%d)",
                    step.name, attempt + 1, attempts,
                )

            t0 = time.monotonic()
            try:
                async with asyncio.timeout(step.timeout_seconds):
                    sr = await self._dispatch_step(step, ctx)
                    sr.duration_ms = (time.monotonic() - t0) * 1000
                    if sr.success:
                        return sr
                    last_error = sr.error
            except asyncio.TimeoutError:
                last_error = f"timed out after {step.timeout_seconds}s"
                log.warning("Step '%s' timed out", step.name)
            except Exception as exc:
                last_error = str(exc)
                log.warning("Step '%s' failed: %s", step.name, exc)

        duration = (time.monotonic() - t0) * 1000
        return StepResult(
            step_name=step.name,
            success=False,
            error=last_error,
            duration_ms=duration,
        )

    async def _dispatch_step(
        self, step: StepDef, ctx: dict[str, Any],
    ) -> StepResult:
        """Route a step to the correct handler based on its action type."""
        action = step.action
        if action == "tool":
            return await self._run_tool(step, ctx)
        elif action == "shell_command":
            return await self._run_shell(step, ctx)
        elif action == "http_request":
            return await self._run_http(step, ctx)
        elif action == "ask_llm":
            return await self._run_llm(step, ctx)
        elif action == "conditional":
            return self._run_conditional(step, ctx)
        elif action == "set_variable":
            return self._run_set_variable(step, ctx)
        else:
            return StepResult(
                step_name=step.name,
                success=False,
                error=f"Unknown action: {action}",
            )

    # ── action handlers ───────────────────────────────────────────────────

    async def _run_tool(self, step: StepDef, ctx: dict[str, Any]) -> StepResult:
        """Invoke a Bantz tool from the global registry."""
        from bantz.tools import registry as tool_registry

        tool = tool_registry.get(step.tool)
        if tool is None:
            return StepResult(
                step_name=step.name,
                success=False,
                error=f"Tool '{step.tool}' not found in registry",
            )
        args = self._interpolate_dict(step.args, ctx)
        try:
            result = await tool.execute(**args)
            return StepResult(
                step_name=step.name,
                success=result.success,
                output=result.output,
                data=result.data,
                error=result.error,
            )
        except Exception as exc:
            return StepResult(
                step_name=step.name,
                success=False,
                error=str(exc),
            )

    async def _run_shell(self, step: StepDef, ctx: dict[str, Any]) -> StepResult:
        """Run a shell command via the shell tool."""
        from bantz.tools import registry as tool_registry

        shell_tool = tool_registry.get("shell")
        if shell_tool is None:
            return StepResult(
                step_name=step.name,
                success=False,
                error="Shell tool not available",
            )
        cmd = self._interpolate(step.command, ctx)
        try:
            result = await shell_tool.execute(command=cmd)
            return StepResult(
                step_name=step.name,
                success=result.success,
                output=result.output,
                data=result.data,
                error=result.error,
            )
        except Exception as exc:
            return StepResult(
                step_name=step.name,
                success=False,
                error=str(exc),
            )

    async def _run_http(self, step: StepDef, ctx: dict[str, Any]) -> StepResult:
        """Make an HTTP request."""
        import aiohttp

        url = self._interpolate(step.url, ctx)
        headers = {k: self._interpolate(v, ctx) for k, v in step.headers.items()}
        body = self._interpolate_dict(step.body, ctx) if step.body else None

        try:
            async with aiohttp.ClientSession() as session:
                method = step.method.upper()
                kwargs: dict[str, Any] = {"headers": headers}
                if body and method in ("POST", "PUT", "PATCH"):
                    kwargs["json"] = body
                async with session.request(method, url, **kwargs) as resp:
                    text = await resp.text()
                    return StepResult(
                        step_name=step.name,
                        success=200 <= resp.status < 400,
                        output=text[:4000],
                        data={"status": resp.status, "headers": dict(resp.headers)},
                        error="" if resp.status < 400 else f"HTTP {resp.status}",
                    )
        except Exception as exc:
            return StepResult(
                step_name=step.name,
                success=False,
                error=str(exc),
            )

    async def _run_llm(self, step: StepDef, ctx: dict[str, Any]) -> StepResult:
        """Generate text via the Bantz LLM layer."""
        prompt = self._interpolate(step.prompt, ctx)
        try:
            from bantz.llm.ollama_client import generate
            text = await generate(prompt)
            return StepResult(
                step_name=step.name,
                success=True,
                output=text,
            )
        except ImportError:
            return StepResult(
                step_name=step.name,
                success=False,
                error="LLM not available (ollama_client not importable)",
            )
        except Exception as exc:
            return StepResult(
                step_name=step.name,
                success=False,
                error=str(exc),
            )

    def _run_conditional(self, step: StepDef, ctx: dict[str, Any]) -> StepResult:
        """Evaluate a condition and set jump target."""
        expr = self._interpolate(step.condition, ctx)
        result = self._eval_condition(expr)
        jump = step.then_step if result else step.else_step
        return StepResult(
            step_name=step.name,
            success=True,
            output=f"Condition '{step.condition}' → {result}",
            data={"result": result, "jump_to": jump or ""},
        )

    def _run_set_variable(self, step: StepDef, ctx: dict[str, Any]) -> StepResult:
        """Set a variable in the workflow context."""
        val = self._interpolate(step.value, ctx)
        ctx.setdefault("variables", {})[step.variable] = val
        return StepResult(
            step_name=step.name,
            success=True,
            output=f"Set {step.variable} = {val}",
            data={"variable": step.variable, "value": val},
        )

    # ── LLM self-heal ─────────────────────────────────────────────────────

    async def _llm_heal(
        self, step: StepDef, failed: StepResult, ctx: dict[str, Any],
    ) -> StepResult:
        """Ask the LLM to recover from a failed step."""
        prompt = (
            f"A workflow step failed. The step was '{step.name}' "
            f"(action={step.action}). Error: {failed.error}\n\n"
            f"Context so far: {ctx.get('variables', {})}\n\n"
            f"Please provide a corrected output or alternative approach. "
            f"Respond with just the result text."
        )
        try:
            from bantz.llm.ollama_client import generate
            text = await generate(prompt)
            return StepResult(
                step_name=f"{step.name}_healed",
                success=True,
                output=text,
                data={"healed": True},
            )
        except Exception as exc:
            return StepResult(
                step_name=f"{step.name}_healed",
                success=False,
                error=f"LLM heal failed: {exc}",
            )

    # ── template interpolation ────────────────────────────────────────────

    def _build_context(
        self, workflow: WorkflowDef, user_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the initial execution context with inputs and defaults."""
        inputs: dict[str, Any] = {}
        for name, idef in workflow.inputs.items():
            if name in user_inputs:
                inputs[name] = user_inputs[name]
            elif idef.default is not None:
                inputs[name] = idef.default
            elif idef.required:
                raise WorkflowError(f"Required input '{name}' not provided")
        return {
            "inputs": inputs,
            "steps": {},
            "variables": {},
        }

    def _interpolate(self, template: str, ctx: dict[str, Any]) -> str:
        """Replace ``{{ path.to.var }}`` placeholders with context values."""
        def _replacer(m: re.Match) -> str:
            path = m.group(1)
            val = self._resolve_path(path, ctx)
            return str(val) if val is not None else m.group(0)
        return _TEMPLATE_RE.sub(_replacer, template)

    def _interpolate_dict(
        self, d: dict[str, Any], ctx: dict[str, Any],
    ) -> dict[str, Any]:
        """Recursively interpolate all string values in a dict."""
        out: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = self._interpolate(v, ctx)
            elif isinstance(v, dict):
                out[k] = self._interpolate_dict(v, ctx)
            elif isinstance(v, list):
                out[k] = [
                    self._interpolate(i, ctx) if isinstance(i, str) else i
                    for i in v
                ]
            else:
                out[k] = v
        return out

    @staticmethod
    def _resolve_path(path: str, ctx: dict[str, Any]) -> Any:
        """Resolve a dotted path like ``steps.get_weather.output``."""
        parts = path.split(".")
        cur: Any = ctx
        for p in parts:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return None
            if cur is None:
                return None
        return cur

    @staticmethod
    def _eval_condition(expr: str) -> bool:
        """Safely evaluate a simple condition string.

        Supports: ``val == "literal"``, ``val == true/false``,
        ``val != "literal"``, truthiness checks.
        """
        expr = expr.strip()
        if "==" in expr:
            left, right = expr.split("==", 1)
            left, right = left.strip(), right.strip()
            # Handle boolean literals
            if right.lower() == "true":
                return left.lower() in ("true", "1", "yes")
            if right.lower() == "false":
                return left.lower() in ("false", "0", "no", "")
            # String comparison (strip quotes)
            right = right.strip("\"'")
            left = left.strip("\"'")
            return left == right
        if "!=" in expr:
            left, right = expr.split("!=", 1)
            left, right = left.strip(), right.strip()
            right = right.strip("\"'")
            left = left.strip("\"'")
            return left != right
        # Truthiness
        return bool(expr) and expr.lower() not in ("false", "0", "no", "none", "")

    @staticmethod
    def _find_result(results: list[StepResult], name: str) -> StepResult | None:
        for r in results:
            if r.step_name == name:
                return r
        return None
