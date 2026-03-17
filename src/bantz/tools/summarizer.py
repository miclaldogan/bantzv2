"""
Bantz — Summarizer Tool (Architect's Revision for Planner #187)

Native text-processing tool that replaces the virtual ``process_text``
hook.  Unlike the old ``__llm__`` approach (which gave the model
open-ended freedom and risked hallucination loops), this tool has a
**focused, citation-preserving system prompt** and strict output rules.

Usage from the Planner:
    {"step": 3, "tool": "summarizer",
     "params": {"instruction": "Summarize this article: $REF:step_2"},
     "description": "Summarize the fetched article"}

The tool uses the fastest available LLM (Gemini Flash → Ollama fallback)
with a low temperature for deterministic output.
"""
from __future__ import annotations

import logging
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tools.summarizer")

# ── Focused system prompt — no open-ended generation ─────────────────────

_SUMMARIZER_SYSTEM = """\
You are a precise text-processing assistant.  Follow the user's
instruction EXACTLY.  You may summarize, rewrite, translate, extract,
or analyze text — but ONLY the text provided to you.

HARD RULES:
1. NEVER fabricate data, facts, statistics, or quotes that are not
   present in the input text.
2. If the input contains an HTTP error (403 Forbidden, 404 Not Found,
   connection refused), state the error clearly and STOP.  Do NOT
   ask the user to provide the text instead.
3. PRESERVE all source URLs / references.  Append them at the bottom
   of your output as raw unformatted links:
       Telegraph Reference: https://...
   Omitting a source link is a dereliction of duty.
4. NO conversational filler — do not greet, do not sign off.
5. Output ONLY the processed result.  Plain text, no Markdown.\
"""


class SummarizerTool(BaseTool):
    """Text summarization / analysis / rewriting tool.

    Replaces the virtual ``process_text`` with a properly registered
    tool that has a focused system prompt and citation preservation.
    """

    name = "summarizer"
    description = (
        "Summarize, analyze, rewrite, or transform text.  "
        "Param: instruction (str) — what to do with the text.  "
        "Example: 'Summarize the following article: <text>'"
    )
    risk_level = "safe"

    async def execute(self, **kwargs: Any) -> ToolResult:
        instruction = kwargs.get("instruction", "")
        if not instruction:
            return ToolResult(
                success=False, output="",
                error="'instruction' parameter is required.",
            )

        messages = [
            {"role": "system", "content": _SUMMARIZER_SYSTEM},
            {"role": "user", "content": instruction},
        ]

        # Try Gemini Flash first (lower latency for summarization)
        raw: str | None = None
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                raw = await gemini.chat(messages, temperature=0.2)
        except Exception:
            pass  # fall through to Ollama

        if raw is None:
            try:
                from bantz.llm.ollama import ollama
                raw = await ollama.chat(messages)
            except Exception as exc:
                return ToolResult(
                    success=False, output="",
                    error=f"LLM unavailable: {exc}",
                )

        # Strip leaked <thinking> blocks (#214)
        try:
            from bantz.core.intent import strip_thinking
            raw = strip_thinking(raw).strip()
        except Exception:
            raw = raw.strip()

        log.info("Summarizer: processed %d chars → %d chars", len(instruction), len(raw))
        return ToolResult(success=True, output=raw)


# Auto-register on import
registry.register(SummarizerTool())
