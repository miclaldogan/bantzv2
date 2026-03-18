"""
Bantz v3 — Plan-and-Solve Multi-Step Decomposition (#187, #273)

"The Butler's Itinerary"

When the user gives a complex, multi-step command (e.g. "Find 3 articles
about AI, summarize them, and save to a file in the Research folder"),
the PlannerAgent uses the LLM to break it down into a structured JSON
array of steps — an "itinerary" — that the Executor runs sequentially.

#273: decompose() now uses streaming via ``_stream_and_collect()`` to
emit ``<thinking>`` tokens to the TUI in real-time.

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
- summarizer: summarize, analyze, rewrite, or transform text from a previous step. Params: {{"instruction": "Summarize the following: $REF_STEP_N"}}
- browser_control: open apps, navigate browser, take screenshots, click UI elements, type in page inputs. Params: {{"action": "open|navigate|new_tab|screenshot|find_and_click|type_in_element|hotkey|type|scroll", "app": "firefox", "url": "https://...", "target": "element name", "text": "...", "keys": "ctrl+t", "wait": 2.0}}
  • action=navigate → go to url= in browser
  • action=find_and_click → click a visible UI element (target= required)
  • action=type_in_element → find element by name, click it, type text= into it (target= text= required)
  • action=screenshot → capture current screen state
  Use for: "open firefox", "go to a website", "search on YouTube", "click a button", "type in a form", any GUI automation.
  (Legacy alias: "process_text" also accepted by the executor.)

CRITICAL TOOL RULES:
- NEVER use `web_search` or `news` to summarize, rewrite, translate, or analyze text. Those tools are STRICTLY for fetching new external information.
- If a step requires summarizing, analyzing, or modifying text from a previous step, you MUST use the `summarizer` tool. Put your exact instructions (e.g., "Summarize the following: $REF_STEP_1") in the "instruction" param.
- When the user wants a THOROUGH research report (not just snippets), use `web_search` first, then `read_url` to fetch the full article text from the best URL, then `summarizer` to summarize.
- NEVER use the `news` tool for specific research, reading full articles, or when you need a URL. The `news` tool provides a text-only summary of today's headlines without source links. It is ONLY for general "What's in the news today?" questions.
- For finding specific articles, news, or topics to read, you MUST use `web_search`. `web_search` returns the raw URLs required for the `read_url` tool.
- Do NOT invent intermediate steps like "Extract URL from results". Just pass `$REF_STEP_N` from `web_search` directly to `read_url` — the executor handles extraction automatically.

TOOL USAGE BEST PRACTICES:
- For `web_search`: Keep queries SHORT and broad (e.g., "Google quantum computer study", NOT "Search for the article titled Google Quantum Computer Makes Breakthrough in Quantum Error Correction"). The search engine works best with concise keywords.
- For `read_url`: The `read_url` tool strictly requires ONLY a valid 'http' URL. Do NOT pass natural language to it. However, `$REF_STEP_N` from `web_search` contains full text snippets. Therefore it is heavily recommended to instruct the system accurately. Actually, the rule is: ALWAYS use `$REF_STEP_N` from `web_search` as the `url` parameter for `read_url`. The system will automatically extract the best URL from the search results text.
- Keep plans as SHORT as possible. A complete deep reading workflow should only be 3 or 4 steps: `web_search` -> `read_url` -> `summarizer` -> `filesystem`.

STANDARD OPERATING PROCEDURES (SOP) & FALLBACKS:
- Rule 1 (Strict Path Selection): NEVER create conditional steps (e.g., 'If Step 1 fails...'). The executor runs ALL steps sequentially. You must choose ONE path before planning: EITHER use the simple specialist tool (like `weather`) OR use the deep research flow (`web_search` -> `read_url` -> `process_text`). Do NOT mix them in the same plan.
- Deep Research bypass: If the user explicitly asks you to search the internet, read a site, or bypass your normal tools, or asks for "latest news about [topic]", choose the deep research flow. Always chain `web_search` -> `read_url` -> `process_text` to extract detailed facts from the actual articles.

COREFERENCE RESOLUTION (CRITICAL):
Before generating ANY tool arguments, you MUST resolve pronouns and
references by analyzing the RECENT CONVERSATION provided below.
- "him" / "her" → find the person's name from recent context
- "it" / "that" / "this" → find the specific object/file/topic mentioned
- "yesterday's file" / "that report" → find the exact filename or path
- "the same city" → find which city was discussed
If you cannot resolve a reference, use the most recent relevant entity
from the conversation. NEVER leave pronouns unresolved in tool params.

{recent_history_block}

RULES:
1. Each step must use exactly ONE tool.
2. Steps execute in order. Later steps can reference output of earlier steps.
3. If a step needs the output of a previous step, note it in "depends_on".
4. Keep it minimal — don't add unnecessary steps.
5. For file writing, default to ~/Desktop/ if no path is specified.
6. Return your response in exactly two parts:
   a. BEFORE outputting the JSON, you MUST open a `<thinking> ... </thinking>` block and perform a strict Self-Audit:
      - Step 1: Goal & Information Extraction: What is the final goal? What strict parameters (file names, exact URLs, locations) did the user provide?
      - Step 2: Path Selection & Tool Matching: Which seq of tools achieves this? If reading an article, do I have the URL or do I need to search first?
      - Step 3: Double-Check / Self-Correction: Have I skipped any required tools? Am I faking variables? Are my params correct and based on real input?
   b. After the thinking block, output ONLY a valid JSON array. No markdown fences. No explanation.
7. CRITICAL: When referencing output from a previous step, use `$REF_STEP_N` (e.g. `$REF_STEP_1`, `$REF_STEP_2`). Do NOT invent custom variable names like `$REF_STEP_1_url` or `$REF_STEP_1_summary`. The ONLY valid output reference is `$REF_STEP_N`. (Legacy format `{{step_N_output}}` is also accepted but deprecated.)
8. PATH CHAINING: When a later step needs the file path or folder path from an earlier step, use `$REF_STEP_N_PARAMS_KEY` where KEY is the exact param name from that step. For example:
   - If Step 1 used `filesystem` with `"folder_path": "~/Desktop/research"`, Step 2 can reference that folder as `$REF_STEP_1_PARAMS_folder_path/summary.txt`.
   - If Step 1 used `filesystem` with `"path": "~/report.txt"`, Step 2 can use `$REF_STEP_1_PARAMS_path` to read the same file.
   - NEVER use `$REF_STEP_N` to construct file paths. Tool output is human-readable text like "Folder created successfully", NOT a path. Always use `$REF_STEP_N_PARAMS_path` or `$REF_STEP_N_PARAMS_folder_path` for paths.

OUTPUT FORMAT (return a JSON array of objects):
<thinking>
Step 1: The user wants to [objective here]. Key entities observed: [x, y].
Step 2: The optimal chain of tools is tool_A -> tool_B.
Step 3: Double-Check: I must not hallucinate the output of tool_A. I will use $REF_STEP_1.
</thinking>
[
  {{"step": 1, "tool": "<tool_name>", "params": {{...}}, "description": "Brief description", "depends_on": null}},
  {{"step": 2, "tool": "<tool_name>", "params": {{...}}, "description": "Brief description", "depends_on": 1}},
  ...
]

EXAMPLES:

User: "Search for articles about quantum computing, summarize the best one, and save to a file"
<thinking>
Step 1: Goal is to find, read, summarize, and save an article. Entities: Query="quantum computing breakthrough", Target File=Quantum summary.
Step 2: Needs deep research flow: web_search -> read_url -> process_text -> filesystem.
Step 3: Double-Check: read_url needs a URL, which I will dynamically get from $REF_STEP_1. summarizer will summarize $REF_STEP_2. Correct.
</thinking>
[
  {{"step": 1, "tool": "web_search", "params": {{"query": "quantum computing breakthrough"}}, "description": "Search for quantum computing articles — returns a list of URLs", "depends_on": null}},
  {{"step": 2, "tool": "read_url", "params": {{"url": "$REF_STEP_1"}}, "description": "Read the full article from the URL returned by step 1", "depends_on": 1}},
  {{"step": 3, "tool": "summarizer", "params": {{"instruction": "Summarize this article in detail, preserving source URLs at the bottom: $REF_STEP_2"}}, "description": "Summarize the article text from step 2", "depends_on": 2}},
  {{"step": 4, "tool": "filesystem", "params": {{"action": "create_folder_and_file", "folder_path": "~/Desktop/research", "file_name": "quantum_computing_summary.txt", "content": "$REF_STEP_3"}}, "description": "Save the summary to a file", "depends_on": 3}}
]

User: "Create a research folder and save a summary of quantum computing into it"
<thinking>
Step 1: Goal is to create a folder, then search, summarize, and save into that folder.
Step 2: filesystem -> web_search -> process_text -> filesystem.
Step 3: Double-Check: Step 4 needs the folder path from Step 1. I must use $REF_STEP_1_PARAMS_folder_path NOT $REF_STEP_1 (which would be "Folder created" text).
</thinking>
[
  {{"step": 1, "tool": "filesystem", "params": {{"action": "create_folder_and_file", "folder_path": "~/Desktop/research", "file_name": ".gitkeep", "content": ""}}, "description": "Create the research folder", "depends_on": null}},
  {{"step": 2, "tool": "web_search", "params": {{"query": "quantum computing breakthroughs 2025"}}, "description": "Search for articles", "depends_on": null}},
  {{"step": 3, "tool": "summarizer", "params": {{"instruction": "Summarize this article in detail: $REF_STEP_2"}}, "description": "Summarize the search results", "depends_on": 2}},
  {{"step": 4, "tool": "filesystem", "params": {{"action": "write", "path": "$REF_STEP_1_PARAMS_folder_path/quantum_summary.txt", "content": "$REF_STEP_3"}}, "description": "Save summary into the research folder", "depends_on": 3}}
]

User: "Open Firefox and go to youtube.com, then open the GülDür channel"
<thinking>
Step 1: Goal is to open Firefox, navigate to YouTube, then navigate to a specific channel.
Step 2: This is a GUI automation task. Use browser_control for all steps. action=open to launch Firefox, action=navigate to go to YouTube, action=navigate again for the channel page.
Step 3: Double-Check: I must use browser_control with action=open NOT summarizer. The params MUST include "action" key.
</thinking>
[
  {{"step": 1, "tool": "browser_control", "params": {{"action": "open", "app": "firefox", "wait": 3.0}}, "description": "Open Firefox browser", "depends_on": null}},
  {{"step": 2, "tool": "browser_control", "params": {{"action": "navigate", "app": "firefox", "url": "https://www.youtube.com", "wait": 2.5}}, "description": "Navigate to YouTube", "depends_on": 1}},
  {{"step": 3, "tool": "browser_control", "params": {{"action": "navigate", "app": "firefox", "url": "https://www.youtube.com/@GulDur", "wait": 2.5}}, "description": "Go to GülDür channel", "depends_on": 2}}
]

User: "Go to YouTube and search for funny cats"
<thinking>
Step 1: Goal is to navigate to YouTube then search.
Step 2: Use browser_control. Navigate to YouTube, then use type_in_element with site=youtube so it uses the direct search URL.
Step 3: Pass site= hint so type_in_element constructs the direct YouTube search URL internally.
</thinking>
[
  {{"step": 1, "tool": "browser_control", "params": {{"action": "navigate", "app": "firefox", "url": "https://www.youtube.com", "wait": 2.5}}, "description": "Navigate to YouTube", "depends_on": null}},
  {{"step": 2, "tool": "browser_control", "params": {{"action": "type_in_element", "app": "firefox", "target": "search box", "text": "funny cats", "site": "youtube", "press_enter": "true"}}, "description": "Search YouTube for funny cats", "depends_on": 1}}
]

User: "Search for 'Python tutorial' on Google"
<thinking>
Step 1: Goal is to go to Google and search.
Step 2: Navigate to Google, then type_in_element with site=google.
Step 3: site= hint lets type_in_element use the direct Google search URL.
</thinking>
[
  {{"step": 1, "tool": "browser_control", "params": {{"action": "navigate", "app": "firefox", "url": "https://www.google.com", "wait": 2.0}}, "description": "Navigate to Google", "depends_on": null}},
  {{"step": 2, "tool": "browser_control", "params": {{"action": "type_in_element", "app": "firefox", "target": "search input", "text": "Python tutorial", "site": "google", "press_enter": "true"}}, "description": "Search on Google", "depends_on": 1}}
]

User: "open me a dr strange video on youtube" (or "find X video on youtube", "play X on youtube")
<thinking>
Step 1: Goal is to find and play a video on YouTube. This needs: navigate to YouTube search results for "dr strange", wait for results to load, then find and click the first video.
Step 2: browser_control for all steps. type_in_element with site=youtube uses direct search URL. wait_for_load waits for results. find_and_click uses VLM to find the first video result.
Step 3: 3 steps total — search, wait, click first result.
</thinking>
[
  {{"step": 1, "tool": "browser_control", "params": {{"action": "type_in_element", "app": "firefox", "target": "search box", "text": "dr strange", "site": "youtube", "press_enter": "false"}}, "description": "Navigate to YouTube search results for dr strange", "depends_on": null}},
  {{"step": 2, "tool": "browser_control", "params": {{"action": "wait_for_load", "app": "firefox", "max_wait": 8.0}}, "description": "Wait for search results to load", "depends_on": 1}},
  {{"step": 3, "tool": "browser_control", "params": {{"action": "find_and_click", "app": "firefox", "target": "first video result", "site": "youtube", "wait_load": "false"}}, "description": "Click the first video result", "depends_on": 2}}
]

User: "Check my emails, then check the weather in Istanbul, and tell me what's on my calendar"
<thinking>
Step 1: Goal is three independent tasks: Email unread, Weather in Istanbul, Today's Calendar.
Step 2: Tools needed: gmail In parallel to weather In parallel to calendar. Series execution is fine.
Step 3: Double-Check: I have the city "Istanbul". I don't need any complex variables between steps as they don't depend on each other.
</thinking>
[
  {{"step": 1, "tool": "gmail", "params": {{"action": "unread"}}, "description": "Check unread emails", "depends_on": null}},
  {{"step": 2, "tool": "weather", "params": {{"city": "Istanbul"}}, "description": "Check weather in Istanbul", "depends_on": null}},
  {{"step": 3, "tool": "calendar", "params": {{"action": "today"}}, "description": "Check today's calendar", "depends_on": null}}
]\
"""

REPLAN_SYSTEM = """\
You are Bantz, a 1920s-era English butler. The original multi-step plan has
encountered an obstacle at one of its steps.  You must devise a concise Plan B.

ORIGINAL REQUEST: {original_request}

COMPLETED STEPS AND THEIR OUTPUTS:
{completed_summary}

FAILED STEP:
  Step {failed_step_num}: tool={failed_tool}, error="{failed_error}"

AVAILABLE TOOLS: {tool_names}

Your task: return a NEW JSON array of steps (Plan B) that accomplishes the
original request given what has already succeeded.  Number steps from 1.
Rules:
- Do NOT repeat already-completed steps.
- If no alternative exists, return a single step using the "summarizer" tool
  that honestly reports the failure to the user.
- Return ONLY valid JSON — no markdown fences, no explanation.
- Maximum 4 steps.

Example output:
[
  {{"step": 1, "tool": "web_search", "params": {{"query": "alternative source"}}, "description": "Try alternative source", "depends_on": null}},
  {{"step": 2, "tool": "summarizer", "params": {{"instruction": "Summarize: $REF_STEP_1"}}, "description": "Summarize result", "depends_on": 1}}
]\
"""


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

    @staticmethod
    def _format_recent_history(recent_history: list[dict] | None) -> str:
        """Format recent conversation for coreference resolution block."""
        if not recent_history:
            return "RECENT CONVERSATION: (none)"
        lines = ["RECENT CONVERSATION:"]
        for msg in recent_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]
            lines.append(f"  {role}: {content}")
        return "\n".join(lines)

    async def decompose(
        self,
        en_input: str,
        tool_names: list[str],
        *,
        recent_history: list[dict] | None = None,
    ) -> list[PlanStep]:
        """Use the LLM to decompose a complex request into plan steps.

        #273: Now uses streaming via ``_stream_and_collect()`` so the
        ``<thinking>`` block is emitted to the TUI in real-time.

        Args:
            en_input: User request (in English).
            tool_names: Available tool names from the registry.
            recent_history: Last few conversation turns for coreference
                resolution (e.g. resolving 'him', 'that file').

        Returns:
            List of PlanStep objects. Empty list if decomposition fails.
        """
        from bantz.core.intent import _stream_and_collect

        history_block = self._format_recent_history(recent_history)
        prompt = PLANNER_SYSTEM.format(
            tool_names=", ".join(tool_names),
            recent_history_block=history_block,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": en_input},
        ]

        raw = await _stream_and_collect(
            messages, emit_thinking=True, source="planner",
        )

        try:
            steps_data = self._parse_steps(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Planner decomposition failed (attempt 1): %s", exc)
            # Retry with correction (no thinking events for retry)
            try:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": (
                    "That was not valid JSON. Return ONLY a JSON array of step "
                    "objects. No markdown. No explanation."
                )})
                raw2 = await _stream_and_collect(
                    messages, emit_thinking=False, source="planner_retry",
                )
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

        # Normalise tool names — LLMs sometimes return "firefox", "browser", etc.
        _PLANNER_ALIASES: dict[str, str] = {
            "firefox": "browser_control",
            "chrome": "browser_control",
            "chromium": "browser_control",
            "browser": "browser_control",
            "browse": "browser_control",
            "navigate": "browser_control",
            "email": "gmail",
            "mail": "gmail",
            "bash": "shell",
            "terminal": "shell",
            "file": "filesystem",
            "files": "filesystem",
            "search": "web_search",
            "google": "web_search",
            "news_tool": "news",
            "remind": "reminder",
            "events": "calendar",
        }
        for s in steps:
            normed = _PLANNER_ALIASES.get(s.tool.lower())
            if normed and s.tool != normed:
                log.info("Planner normalised tool '%s' → '%s' (step %d)", s.tool, normed, s.step)
                s.tool = normed

        # Validate: each step must reference a known tool or virtual tool
        _virtual_tools = {"process_text", "summarizer", "read_url"}
        allowed = set(tool_names) | _virtual_tools
        valid_steps = []
        for s in steps:
            if s.tool in allowed:
                valid_steps.append(s)
            else:
                log.warning("Planner dropped step %d: unknown tool '%s'", s.step, s.tool)

        return valid_steps

    @staticmethod
    def _parse_steps(raw: str) -> list[dict]:
        """Extract a JSON array from the LLM response."""
        from bantz.core.intent import strip_thinking

        text = strip_thinking(raw)  # #214 — remove leaked thinking blocks
        text = text.strip().strip("`")
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

    async def replan(
        self,
        original_request: str,
        failed_step_num: int,
        failed_tool: str,
        failed_error: str,
        context_store: dict,  # step_num → {"params": ..., "output": ...}
        tool_names: list[str],
    ) -> list[PlanStep]:
        """Ask the LLM to devise a Plan B after a step failure (#216).

        Uses the REPLAN_SYSTEM prompt which includes what succeeded so far,
        what failed, and why — so the LLM can generate a targeted recovery plan.

        Returns a (possibly empty) list of PlanStep objects.
        Maximum 1 replan per execution — the caller enforces the guard.
        """
        from bantz.core.intent import _stream_and_collect

        # Summarise completed steps for the prompt
        completed_lines: list[str] = []
        for step_num in sorted(context_store):
            entry = context_store[step_num]
            out = (entry.get("output") or "")[:300].replace("\n", " ")
            completed_lines.append(
                f"  Step {step_num}: output='{out}'" if out else f"  Step {step_num}: (no output)"
            )
        completed_summary = "\n".join(completed_lines) if completed_lines else "  (none)"

        prompt = REPLAN_SYSTEM.format(
            original_request=original_request,
            completed_summary=completed_summary,
            failed_step_num=failed_step_num,
            failed_tool=failed_tool,
            failed_error=failed_error[:300],
            tool_names=", ".join(tool_names),
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Devise Plan B now."},
        ]

        try:
            raw = await _stream_and_collect(
                messages, emit_thinking=True, source="replanner",
            )
            steps_data = self._parse_steps(raw)
        except Exception as exc:
            log.warning("Replanner failed: %s", exc)
            return []

        steps: list[PlanStep] = []
        for i, s in enumerate(steps_data, 1):
            steps.append(PlanStep(
                step=s.get("step", i),
                tool=s.get("tool", ""),
                params=s.get("params", {}),
                description=s.get("description", ""),
                depends_on=s.get("depends_on"),
            ))

        log.info("Replanner produced %d Plan-B step(s)", len(steps))
        return steps


# ── Singleton ────────────────────────────────────────────────────────────────

planner_agent = PlannerAgent()
