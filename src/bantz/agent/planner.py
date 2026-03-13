"""
Bantz v3 — Plan-and-Solve Multi-Step Decomposition (#187)

"The Butler's Itinerary"

When the user gives a complex, multi-step command (e.g. "Find 3 articles
about AI, summarize them, and save to a file in the Research folder"),
the PlannerAgent uses the LLM to break it down into a structured JSON
array of steps — an "itinerary" — that the Executor runs sequentially.

Usage:
    from bantz.agent.planner import planner_agent
    steps = await planner_agent.decompose(en_input, tool_schemas)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from bantz.llm.ollama import ollama

log = logging.getLogger("bantz.planner")

# ── System prompt — enforces 1920s Butler persona + valid JSON output ────────

PLANNER_SYSTEM = """\
You are Bantz, a 1920s-era English butler who is also a meticulous planner.
The user has given you a complex request that requires MULTIPLE steps.

Your task: break the request into a numbered list of discrete steps.
Each step uses exactly ONE tool from the available set.

AVAILABLE TOOLS: {tool_names}

TOOL REFERENCE:
- web_search: search the internet for information. Params: {{"query": "..."}}
- filesystem: read/write/create files and folders. Params: {{"action": "write|read|create_folder_and_file", "path": "...", "content": "...", "folder_path": "...", "file_name": "..."}}
- shell: run a bash command. Params: {{"command": "..."}}
- gmail: send/read email. Params: {{"action": "compose_and_send|unread|search", "to": "...", "intent": "...", "subject": "..."}}
- calendar: manage events. Params: {{"action": "create|today|week", "title": "...", "date": "...", "time": "..."}}
- weather: check weather. Params: {{"city": "..."}}
- news: get headlines. Params: {{"source": "all|hn"}}
- system: check CPU/RAM/disk. Params: {{"metric": "all|cpu|ram|disk"}}
- document: summarize/read a document. Params: {{"path": "...", "action": "summarize|read|ask", "question": "..."}}
- read_url: fetch and read the full text content of a specific URL/webpage. Params: {{"url": "https://..."}}
- process_text: summarize, analyze, rewrite, or transform text from a previous step. Params: {{"instruction": "Summarize the following: {{step_N_output}}"}}

CRITICAL TOOL RULES:
- NEVER use `web_search` or `news` to summarize, rewrite, translate, or analyze text. Those tools are STRICTLY for fetching new external information.
- If a step requires summarizing, analyzing, or modifying text from a previous step, you MUST use the `process_text` tool. Put your exact instructions (e.g., "Summarize the following: {{step_1_output}}") in the "instruction" param.
- When the user wants a THOROUGH research report (not just snippets), use `web_search` first, then `read_url` to fetch the full article text from the best URL, then `process_text` to summarize.

TOOL USAGE BEST PRACTICES:
- For `web_search`: Keep queries SHORT and broad (e.g., "Google quantum computer study", NOT "Search for the article titled Google Quantum Computer Makes Breakthrough in Quantum Error Correction"). The search engine works best with concise keywords.
- For `read_url`: The parameter MUST be a valid HTTP/HTTPS URL. Do NOT pass natural language or search queries to this tool. You get this URL from `{{step_N_output}}` of a previous `web_search` step.
- Keep plans as SHORT as possible. A complete deep reading workflow should only be 3 or 4 steps: `web_search` -> `read_url` -> `process_text` -> `filesystem`.

RULES:
1. Each step must use exactly ONE tool.
2. Steps execute in order. Later steps can reference output of earlier steps.
3. If a step needs the output of a previous step, note it in "depends_on".
4. Keep it minimal — don't add unnecessary steps.
5. For file writing, default to ~/Desktop/ if no path is specified.
6. Return ONLY a valid JSON array. No markdown fences. No explanation.
7. CRITICAL: When referencing output from a previous step, you MUST use the EXACT format `{{step_N_output}}` (e.g. `{{step_1_output}}`, `{{step_2_output}}`). Do NOT invent custom variable names like `{{step_1_url}}`, `{{step_1_best_article_url}}`, or `{{step_1_summary}}`. The ONLY valid placeholder is `{{step_N_output}}`.

OUTPUT FORMAT (return a JSON array of objects):
[
  {{"step": 1, "tool": "<tool_name>", "params": {{...}}, "description": "Brief description", "depends_on": null}},
  {{"step": 2, "tool": "<tool_name>", "params": {{...}}, "description": "Brief description", "depends_on": 1}},
  ...
]

EXAMPLES:

User: "Search for articles about quantum computing, summarize the best one, and save to a file"
[
  {{"step": 1, "tool": "web_search", "params": {{"query": "quantum computing breakthroughs"}}, "description": "Search for quantum computing articles (returns a list of URLs)", "depends_on": null}},
  {{"step": 2, "tool": "read_url", "params": {{"url": "{{step_1_output}}"}}, "description": "Fetch the full article text from the URL returned by web_search", "depends_on": 1}},
  {{"step": 3, "tool": "process_text", "params": {{"instruction": "Summarize the following article into a concise report. Preserve any source URLs at the bottom: {{step_2_output}}"}}, "description": "Summarize the full article text", "depends_on": 2}},
  {{"step": 4, "tool": "filesystem", "params": {{"action": "create_folder_and_file", "folder_path": "~/Desktop/research", "file_name": "quantum_computing_summary.txt", "content": "{{step_3_output}}"}}, "description": "Save the summary to a file", "depends_on": 3}}
]

User: "Check my emails, then check the weather in Istanbul, and tell me what's on my calendar"
[
  {{"step": 1, "tool": "gmail", "params": {{"action": "unread"}}, "description": "Check unread emails", "depends_on": null}},
  {{"step": 2, "tool": "weather", "params": {{"city": "Istanbul"}}, "description": "Check weather in Istanbul", "depends_on": null}},
  {{"step": 3, "tool": "calendar", "params": {{"action": "today"}}, "description": "Check today's calendar", "depends_on": null}}
]\
"""

# ── Complexity detection heuristics ──────────────────────────────────────────

# Conjunctions / sequence markers that hint at multi-step intent
_MULTI_STEP_MARKERS = re.compile(
    r"""
    \b(?:then|after\s+that|next|also|additionally|finally|afterwards)\b
    | \b(?:and\s+(?:then|also|afterwards))\b
    | \b(?:first|second|third)\b.*\b(?:then|next|after)\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Tool-indicating keywords — grouped by tool for counting distinct tools
_TOOL_KEYWORDS: dict[str, list[str]] = {
    "web_search": ["search", "find", "look up", "google", "articles", "research"],
    "gmail": ["email", "mail", "inbox", "compose", "send mail"],
    "calendar": ["calendar", "event", "meeting", "schedule", "appointment"],
    "filesystem": ["save", "file", "folder", "write to", "create file", "create folder"],
    "weather": ["weather", "temperature", "forecast"],
    "news": ["news", "headlines"],
    "system": ["cpu", "ram", "memory", "disk", "system status"],
    "shell": ["run command", "terminal", "bash"],
    "document": ["pdf", "document"],
    "read_url": ["read url", "read page", "open link", "fetch page", "full article", "full text"],
    "process_text": ["summarize", "analyze", "rewrite", "translate", "transform"],
}


@dataclass
class PlanStep:
    """A single step in the butler's itinerary."""
    step: int
    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    depends_on: int | None = None


class PlannerAgent:
    """Decomposes complex user requests into structured step arrays."""

    def is_complex(self, en_input: str) -> bool:
        """Heuristic check: does this input likely need multi-step planning?

        Returns True if:
        - Input has sequence markers (then, after that, next...)
        - Input references 2+ distinct tool categories
        - Input is long (>15 words) with multiple verb phrases
        """
        text = en_input.lower()

        # Check for explicit sequence markers
        if _MULTI_STEP_MARKERS.search(text):
            # Also needs at least 2 different tool intents
            distinct = self._count_distinct_tools(text)
            if distinct >= 2:
                return True

        # Even without markers, 2+ distinct tool intents in a long sentence
        distinct = self._count_distinct_tools(text)
        word_count = len(text.split())
        if distinct >= 2 and word_count >= 8:
            return True

        # Very explicit multi-step: numbered instructions
        if re.search(r"\b(?:1\.|step\s*1|first)\b", text) and \
           re.search(r"\b(?:2\.|step\s*2|second|then)\b", text):
            return True

        return False

    @staticmethod
    def _count_distinct_tools(text: str) -> int:
        """Count how many distinct tool categories are mentioned."""
        found: set[str] = set()
        for tool_name, keywords in _TOOL_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    found.add(tool_name)
                    break
        return len(found)

    async def decompose(
        self,
        en_input: str,
        tool_names: list[str],
    ) -> list[PlanStep]:
        """Use the LLM to decompose a complex request into plan steps.

        Args:
            en_input: User request (in English).
            tool_names: Available tool names from the registry.

        Returns:
            List of PlanStep objects. Empty list if decomposition fails.
        """
        prompt = PLANNER_SYSTEM.format(tool_names=", ".join(tool_names))

        raw = await ollama.chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": en_input},
        ])

        try:
            steps_data = self._parse_steps(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Planner decomposition failed (attempt 1): %s", exc)
            # Retry with correction
            try:
                raw2 = await ollama.chat([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": en_input},
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        "That was not valid JSON. Return ONLY a JSON array of step "
                        "objects. No markdown. No explanation."
                    )},
                ])
                steps_data = self._parse_steps(raw2)
            except Exception as exc2:
                log.warning("Planner decomposition failed (attempt 2): %s", exc2)
                return []

        # Convert raw dicts to PlanStep objects
        steps: list[PlanStep] = []
        for i, s in enumerate(steps_data, 1):
            steps.append(PlanStep(
                step=s.get("step", i),
                tool=s.get("tool", ""),
                params=s.get("params", {}),
                description=s.get("description", ""),
                depends_on=s.get("depends_on"),
            ))

        # Validate: each step must reference a known tool or virtual tool
        _virtual_tools = {"process_text", "read_url"}
        allowed = set(tool_names) | _virtual_tools
        valid_steps = [s for s in steps if s.tool in allowed]
        if len(valid_steps) < len(steps):
            dropped = len(steps) - len(valid_steps)
            log.warning("Planner dropped %d steps with unknown tools", dropped)

        return valid_steps

    @staticmethod
    def _parse_steps(raw: str) -> list[dict]:
        """Extract a JSON array from the LLM response."""
        text = raw.strip().strip("`")
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # Try to find a JSON array
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            result = json.loads(m.group())
        else:
            result = json.loads(text)

        if not isinstance(result, list):
            raise ValueError(f"Expected JSON array, got {type(result).__name__}")
        if len(result) == 0:
            raise ValueError("Empty step array")
        return result

    def format_itinerary(self, steps: list[PlanStep]) -> str:
        """Format the plan as a butler-style announcement for the user.

        Returns a persona-appropriate message like:
        'Allow me a moment to draft an itinerary for this endeavor, ma'am.
         Here is my plan:
           1. Search for quantum computing articles
           2. Save the results to a file'
        """
        if not steps:
            return ""
        lines = [
            "Allow me a moment to draft an itinerary for this endeavor.",
            "Here is my plan:",
        ]
        for s in steps:
            dep = f" (using results from step {s.depends_on})" if s.depends_on else ""
            lines.append(f"  {s.step}. {s.description}{dep}")
        lines.append("")
        lines.append("Commencing forthwith.")
        return "\n".join(lines)


# ── Singleton ────────────────────────────────────────────────────────────────

planner_agent = PlannerAgent()
