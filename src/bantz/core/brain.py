"""
Bantz v2 — Brain (Orchestrator)
input → router (Ollama) → tool executor → finalizer → output

Phase 1: Intentionally kept simple. QualityGate, TieredFinalizer etc. comes in Phase 2.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from bantz.config import config
from bantz.llm.ollama import ollama
from bantz.tools import registry
from bantz.tools import ToolResult


# ── Router prompt ─────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """You are Bantz's routing brain. Analyze the user message and return a JSON routing decision.

Available tools: {tool_list}

RULES:
- If a tool fits, use it. If not, use "chat" (just talk).
- Return ONLY valid JSON, no markdown fences, no explanation.
- For shell commands: set risk_level correctly.

Response format:
{{
  "route": "tool" | "chat",
  "tool_name": "<name or null>",
  "tool_args": {{<args or empty>}},
  "risk_level": "safe" | "moderate" | "destructive",
  "reasoning": "<one sentence>"
}}"""


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM output. Remove Markdown fences if present."""
    text = text.strip()
    # ```json ... ``` or ``` ... ``` Remove
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ── Dataclass: Result of a brain operation ──────────────────────────────────

@dataclass
class BrainResult:
    response: str              # Final text to be shown to the user
    tool_used: str | None      # Which tool was used (or None)
    tool_result: ToolResult | None = None
    needs_confirm: bool = False  # Destructive operation requires confirmation
    pending_command: str = ""    # Command awaiting confirmation


# ── Brain ─────────────────────────────────────────────────────────────────────

class Brain:
    def __init__(self) -> None:
        # Import tools to register them
        import bantz.tools.shell       # noqa: F401
        import bantz.tools.system      # noqa: F401
        import bantz.tools.filesystem  # noqa: F401

    async def process(self, user_input: str, confirmed: bool = False) -> BrainResult:
        """
        Main processing loop.
        confirmed=True: destructive operation confirmed by the user.
        """

        # 1. Router — which tool?
        tool_schemas = registry.all_schemas()
        tool_list = ", ".join(f"{t['name']}({t['risk_level']})" for t in tool_schemas)

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM.format(tool_list=tool_list)},
            {"role": "user", "content": user_input},
        ]

        try:
            raw = await ollama.chat(messages)
            plan = _extract_json(raw)
        except Exception as exc:
            # Router failed → fallback to chat
            return BrainResult(
                response=await self._fallback_chat(user_input),
                tool_used=None,
            )

        route = plan.get("route", "chat")
        tool_name = plan.get("tool_name")
        tool_args = plan.get("tool_args", {})
        risk = plan.get("risk_level", "safe")

        # 2. No tool or just chat
        if route != "tool" or not tool_name:
            return BrainResult(
                response=await self._fallback_chat(user_input),
                tool_used=None,
            )

        # 3. Destructive confirmation check
        if risk == "destructive" and config.shell_confirm_destructive and not confirmed:
            cmd = tool_args.get("command", tool_name)
            return BrainResult(
                response=f"⚠️  This operation might be destructive: `{cmd}`\nDo you confirm? (yes/no)",
                tool_used=tool_name,
                needs_confirm=True,
                pending_command=cmd,
            )

        # 4. Execute tool
        tool = registry.get(tool_name)
        if not tool:
            return BrainResult(
                response=f"Tool not found: {tool_name}",
                tool_used=None,
            )

        result = await tool.execute(**tool_args)

        # 5. Finalize — convert result to natural language
        response = await self._finalize(user_input, result)
        return BrainResult(response=response, tool_used=tool_name, tool_result=result)

    async def _fallback_chat(self, user_input: str) -> str:
        """If no tool, chat directly with Ollama."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Bantz — the user's personal terminal assistant. "

                    "Provide short, clear, and helpful responses. Speak in Turkish."
                ),
            },
            {"role": "user", "content": user_input},
        ]
        try:
            return await ollama.chat(messages)
        except Exception as exc:
            return f"(Ollama connection error: {exc})"

    async def _finalize(self, user_input: str, result: ToolResult) -> str:
        """Convert tool result to natural language."""
        if not result.success:
            return f"❌ Error: {result.error}"

        # Show short outputs directly
        if len(result.output) < 500:
            return result.output

        # Summarize long outputs with LLM
        messages = [
            {
                "role": "system",
                "content": "Summarize the following tool output based on the user's question. Turkish, short.",
            },
            {
                "role": "user",
                "content": f"Question: {user_input}\n\nOutput:\n{result.output[:2000]}",
            },
        ]
        try:
            return await ollama.chat(messages)
        except Exception:
            return result.output[:1000] + "\n... (truncated)"


brain = Brain()