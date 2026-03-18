"""
Bantz — ComputerUseTool: visual desktop/web automation via AutonomousVisionLoop (#188)

Registered in the tool registry as "computer_use".  Allows the planner and
cot_route() to delegate visual navigation tasks to the VLM loop.

Usage (via tool registry):
    result = await registry.get("computer_use").execute(
        task="Open Firefox and navigate to wikipedia.org",
        timeout=120,
    )
"""
from __future__ import annotations

import logging
from typing import Any

from bantz.tools import BaseTool, ToolResult

log = logging.getLogger("bantz.tool.computer_use")


class ComputerUseTool(BaseTool):
    """Visual desktop / web automation via AutonomousVisionLoop.

    The tool captures screenshots, sends them to a VLM for visual grounding,
    and executes the decided actions (click, type, scroll, hotkey) in a loop
    until the goal is achieved or safety limits are hit.
    """

    name = "computer_use"
    description = (
        "Visually navigate and interact with desktop applications and web pages "
        "using screen vision.  Provide a plain-English task description."
    )
    risk_level = "moderate"

    def __init__(self) -> None:
        super().__init__()
        self._loop: Any = None  # lazily initialised

    def _get_loop(self):
        if self._loop is None:
            from bantz.vision.computer_use import vision_loop
            self._loop = vision_loop
        return self._loop

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run a visual navigation task.

        Keyword args:
            task (str):          Plain-English description of what to do.
            success_criteria (str, optional): How to know when done.
                                             Defaults to "Task completed: {task}".
            app (str, optional): Hint for which application to focus first.
            timeout (int):       Maximum seconds to spend (default 120).
            max_steps (int):     Maximum loop iterations (default 15).

        Returns:
            ToolResult with output = final_observation + extracted_text.
        """
        task = str(kwargs.get("task", "")).strip()
        if not task:
            return ToolResult(success=False, output="", error="task parameter is required")

        timeout = int(kwargs.get("timeout", 120))
        max_steps = int(kwargs.get("max_steps", 15))
        success_criteria = str(
            kwargs.get("success_criteria", f"Task completed: {task}")
        )

        from bantz.vision.computer_use import VisionGoal

        goal = VisionGoal(
            description=task,
            success_criteria=success_criteria,
            max_steps=max_steps,
            timeout_s=float(timeout),
        )

        step_log: list[str] = []

        async def _on_step(vs):
            step_log.append(
                f"Step {vs.step_num}: {vs.action} {vs.target!r} "
                f"— {'ok' if vs.success else vs.error}"
            )
            # Notify TUI toast if available
            try:
                from bantz.core.notification_manager import notify_toast
                notify_toast(
                    f"👁 Step {vs.step_num}/{goal.max_steps}: "
                    f"{vs.action} {vs.target or '…'}",
                    toast_type="info",
                )
            except Exception:
                pass

        try:
            result = await self._get_loop().execute(goal, on_step=_on_step)
        except Exception as exc:
            log.warning("ComputerUseTool.execute error: %s", exc)
            return ToolResult(success=False, output="", error=str(exc))

        output_parts = []
        if result.final_observation:
            output_parts.append(result.final_observation)
        if result.extracted_text.strip():
            output_parts.append(f"Extracted text:\n{result.extracted_text.strip()}")
        if step_log:
            output_parts.append("Steps:\n" + "\n".join(step_log))

        return ToolResult(
            success=result.success,
            output="\n\n".join(output_parts),
            error=result.error if not result.success else "",
            data={
                "steps": len(result.steps),
                "total_time_s": round(result.total_time_s, 2),
            },
        )


# ── Auto-register ─────────────────────────────────────────────────────────────

from bantz.tools import registry as _registry  # noqa: E402
_registry.register(ComputerUseTool())
