"""
Bantz v2 — Workflow Engine

Detects and executes chained multi-tool commands.
"send mail to prof, add it to calendar, remind me tomorrow" → 3 sequential tool calls.

Usage:
    from bantz.core.workflow import workflow_engine
    steps = await workflow_engine.detect(user_input, en_input)
    if steps:
        result = await workflow_engine.execute(steps, en_input, tc)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bantz.tools import registry, ToolResult


# ── Splitter patterns ─────────────────────────────────────────────────────────

# Conjunctions/delimiters that separate independent tool intents
_SPLIT_PATTERN = re.compile(
    r"""
    \s*(?:                          # whitespace before delimiter
        ,\s*(?:and\s+|then\s+)?    # ", and" / ", then" / ","
      | \s+and\s+then\s+           # " and then"
      | \s+then\s+                 # " then"
      | \s+and\s+also\s+           # " and also"
      | \s+also\s+                 # " also"
      | \s+after\s+that\s+         # " after that"
      | \.\s+                      # ". " (sentence boundary)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Tool-indicating keywords — if a sub-sentence contains one, it's likely actionable
_TOOL_KEYWORDS = {
    "mail": "gmail", "email": "gmail", "send": "gmail", "compose": "gmail",
    "inbox": "gmail", "reply": "gmail",
    "calendar": "calendar", "event": "calendar", "meeting": "calendar",
    "schedule": "calendar", "appointment": "calendar",
    "weather": "weather", "temperature": "weather", "forecast": "weather",
    "news": "news", "headlines": "news",
    "assignment": "classroom", "homework": "classroom", "classroom": "classroom",
    "deadline": "classroom",
    "remind": "calendar", "reminder": "calendar",
    "search": "web_search", "look up": "web_search", "google": "web_search",
    "file": "filesystem", "folder": "filesystem",
    "create file": "filesystem", "create folder": "shell",
    "delete": "shell", "remove": "shell",
    "system": "system", "cpu": "system", "ram": "system", "memory": "system",
}


@dataclass
class WorkflowStep:
    """A single step in a workflow chain."""
    sub_input: str                     # the original sub-sentence
    tool_hint: str = ""                # hinted tool name from keywords
    tool_name: str = ""                # resolved tool name (from router/quick_route)
    tool_args: dict = field(default_factory=dict)
    result: ToolResult | None = None   # filled after execution
    success: bool = False


class WorkflowEngine:

    def detect(self, orig: str, en: str) -> list[WorkflowStep] | None:
        """
        Detect if the input contains multiple independent tool intents.
        Returns a list of WorkflowStep if multi-tool, or None if single intent.
        """
        # Split on conjunctions/delimiters
        parts = _SPLIT_PATTERN.split(en)
        parts = [p.strip() for p in parts if p and p.strip() and len(p.strip()) > 3]

        if len(parts) < 2:
            return None

        # Check that at least 2 parts map to different tools
        steps: list[WorkflowStep] = []
        seen_tools: set[str] = set()

        for part in parts:
            hint = self._detect_tool(part)
            if hint:
                seen_tools.add(hint)
                steps.append(WorkflowStep(sub_input=part, tool_hint=hint))
            else:
                # Might be context/filler — attach to previous step or create generic
                if steps:
                    steps[-1].sub_input += f", {part}"
                else:
                    steps.append(WorkflowStep(sub_input=part))

        # Only return multi-step if we have 2+ steps with at least 2 different intents
        # OR 2+ steps even with same tool (e.g. "send mail to A and send mail to B")
        if len(steps) >= 2 and len(seen_tools) >= 2:
            return steps
        # Same tool but clearly different actions
        if len(steps) >= 2 and all(s.tool_hint for s in steps):
            return steps

        return None

    @staticmethod
    def _detect_tool(text: str) -> str:
        """Map a sub-sentence to a likely tool name via keyword matching."""
        t = text.lower()
        for keyword, tool in _TOOL_KEYWORDS.items():
            if keyword in t:
                return tool
        return ""

    async def execute(
        self,
        steps: list[WorkflowStep],
        brain_ref: Any,
        en_input: str,
        tc: dict,
    ) -> str:
        """
        Execute workflow steps sequentially.
        Each step's result provides context to the next.
        Returns a combined summary string.
        """
        outputs: list[str] = []
        prev_context = ""

        for i, step in enumerate(steps, 1):
            # Route this sub-step through brain's quick_route
            quick = brain_ref._quick_route(step.sub_input, step.sub_input)

            if quick and quick["tool"].startswith("_"):
                # Built-in pseudo-tools (briefing, schedule, etc.)
                text = await self._run_builtin(quick, brain_ref)
                step.result = ToolResult(success=True, output=text)
                step.success = True
                outputs.append(f"Step {i}: {text}")
                prev_context = text
                continue

            if quick:
                step.tool_name = quick["tool"]
                step.tool_args = quick["args"]
            elif step.tool_hint:
                # Fallback: use the hinted tool with the sub-input as intent
                step.tool_name = step.tool_hint
                step.tool_args = {"intent": step.sub_input}
            else:
                outputs.append(f"Step {i}: Couldn't determine action for '{step.sub_input}'")
                continue

            tool = registry.get(step.tool_name)
            if not tool:
                outputs.append(f"Step {i}: Tool '{step.tool_name}' not found")
                continue

            # Inject previous context if this step needs it
            if prev_context and "intent" in step.tool_args:
                step.tool_args["context"] = prev_context

            try:
                result = await tool.execute(**step.tool_args)
                step.result = result
                step.success = result.success

                if result.success:
                    # Store context for next step
                    prev_context = result.output[:500]
                    short = result.output[:200].replace("\n", " ")
                    outputs.append(f"✓ [{step.tool_name}] {short}")
                else:
                    outputs.append(f"✗ [{step.tool_name}] {result.error}")
            except Exception as exc:
                outputs.append(f"✗ [{step.tool_name}] Error: {exc}")

        # Build summary
        n_ok = sum(1 for s in steps if s.success)
        header = f"Workflow complete: {n_ok}/{len(steps)} steps succeeded."
        return header + "\n" + "\n".join(outputs)

    @staticmethod
    async def _run_builtin(quick: dict, brain_ref: Any) -> str:
        """Handle built-in pseudo-tools like _briefing, _schedule_*."""
        tool = quick["tool"]
        if tool == "_briefing":
            from bantz.core.briefing import briefing as _briefing
            return await _briefing.generate()
        if tool == "_schedule_today":
            from bantz.core.schedule import schedule as _sched
            return _sched.format_today()
        if tool == "_schedule_next":
            from bantz.core.schedule import schedule as _sched
            return _sched.format_next()
        if tool == "_schedule_week":
            from bantz.core.schedule import schedule as _sched
            return _sched.format_week()
        return f"(built-in: {tool})"


workflow_engine = WorkflowEngine()
