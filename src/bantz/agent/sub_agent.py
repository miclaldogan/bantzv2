"""
Bantz — Sub-Agent Base & Specialized Roles (#321)

Multi-Agent Hierarchy inspired by JARVIS.  The main Brain orchestrator
can delegate sub-tasks to specialized agents that have their own system
prompts, restricted tool access, and isolated message history.

Architecture
────────────
    Brain (Orchestrator)
      └── AgentManager.delegate(role, task, context)
              ├── ResearcherAgent   — web search, reading, summarisation
              ├── DeveloperAgent    — shell, filesystem, code tasks
              └── ReviewerAgent     — analysis, checking, validation

Design decisions:
  • Sub-agents do NOT share the Brain's conversation history.  The
    orchestrator passes a synthesised ``context`` dict — no token waste.
  • Sub-agents return a structured ``SubAgentResult`` (JSON-serialisable).
  • Each role has a curated ``allowed_tools`` set — sub-agents CANNOT
    access dangerous tools like ``manage_agents`` or destructive shell.
  • LLM call reuses the same Gemini-first-Ollama-fallback pattern.
  • Sub-agents are ephemeral — created per task, no persistent state.
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("bantz.sub_agent")


# ═══════════════════════════════════════════════════════════════════════════
# Result type
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SubAgentResult:
    """Structured response from a sub-agent execution.

    The orchestrator (Brain) reads ``summary`` for the user-facing answer
    and ``data`` for any structured payload (URLs, code blocks, etc.).
    """
    success: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    tools_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "data": self.data,
            "error": self.error,
            "tools_used": self.tools_used,
        }

    @classmethod
    def failure(cls, error: str) -> "SubAgentResult":
        return cls(success=False, summary="", error=error)


# ═══════════════════════════════════════════════════════════════════════════
# Base Sub-Agent
# ═══════════════════════════════════════════════════════════════════════════

class SubAgent(ABC):
    """Base class for all specialised sub-agents.

    Subclasses must define:
      • ``role``          — unique identifier (e.g. "researcher")
      • ``display_name``  — human-friendly label
      • ``system_prompt``  — LLM system instructions
      • ``allowed_tools`` — set of tool names this agent may invoke

    The ``run()`` method:
      1. Builds a minimal message array from the task + context
      2. Calls the LLM for a structured JSON plan
      3. Executes any tool calls in sequence
      4. Calls the LLM again with tool results to produce a final summary
      5. Returns ``SubAgentResult``
    """

    role: str
    display_name: str
    system_prompt: str
    allowed_tools: set[str]
    max_tool_calls: int = 5  # circuit breaker — prevent infinite loops

    @abstractmethod
    def _build_system(self, task: str, context: dict[str, Any]) -> str:
        """Build the full system prompt with task-specific context."""
        ...

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        """Execute the sub-agent's task.

        1. Ask LLM to plan which tools to use (JSON response).
        2. Execute tools sequentially.
        3. Ask LLM to synthesise a final answer from tool results.
        4. Return structured SubAgentResult.
        """
        context = context or {}
        system = self._build_system(task, context)
        tools_used: list[str] = []

        # ── Phase 1: Plan (ask LLM what tools to call) ──────────
        plan_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self._format_task_prompt(task, context)},
        ]

        try:
            plan_raw = await self._llm_chat(plan_messages)
        except Exception as exc:
            return SubAgentResult.failure(f"LLM error during planning: {exc}")

        tool_calls = self._parse_tool_calls(plan_raw)

        # ── Phase 2: Execute tools ──────────────────────────────
        tool_results: list[dict[str, Any]] = []
        if tool_calls:
            from bantz.tools import registry

            for i, call in enumerate(tool_calls[: self.max_tool_calls]):
                tool_name = call.get("tool", "")
                tool_args = call.get("args", {})

                if tool_name not in self.allowed_tools:
                    tool_results.append({
                        "tool": tool_name,
                        "success": False,
                        "output": f"Tool '{tool_name}' not permitted for {self.role} agent.",
                    })
                    continue

                tool = registry.get(tool_name)
                if not tool:
                    tool_results.append({
                        "tool": tool_name,
                        "success": False,
                        "output": f"Tool '{tool_name}' not found in registry.",
                    })
                    continue

                try:
                    tr = await tool.execute(**tool_args)
                    tool_results.append({
                        "tool": tool_name,
                        "success": tr.success,
                        "output": tr.output[:3000],  # cap output to save tokens
                    })
                    tools_used.append(tool_name)
                except Exception as exc:
                    tool_results.append({
                        "tool": tool_name,
                        "success": False,
                        "output": f"Execution error: {exc}",
                    })

        # ── Phase 3: Synthesise (LLM produces final answer) ─────
        synth_messages = list(plan_messages)
        synth_messages.append({"role": "assistant", "content": plan_raw})

        if tool_results:
            tool_summary = json.dumps(tool_results, ensure_ascii=False, indent=2)
            synth_messages.append({
                "role": "user",
                "content": (
                    f"Tool results:\n```json\n{tool_summary}\n```\n\n"
                    "Now synthesise a final answer based on these results. "
                    "Respond with a JSON object: "
                    '{"summary": "...", "data": {...optional structured data}}'
                ),
            })
        else:
            # No tools needed — the plan response IS the answer
            synth_messages.append({
                "role": "user",
                "content": (
                    "Now provide your final answer. "
                    "Respond with a JSON object: "
                    '{"summary": "...", "data": {...optional structured data}}'
                ),
            })

        try:
            final_raw = await self._llm_chat(synth_messages)
        except Exception as exc:
            # If synthesis fails but we have tool results, return those
            if tool_results:
                combined = "\n".join(
                    f"[{r['tool']}] {r['output'][:500]}"
                    for r in tool_results if r.get("success")
                )
                return SubAgentResult(
                    success=True,
                    summary=combined or "Tools executed but synthesis failed.",
                    tools_used=tools_used,
                )
            return SubAgentResult.failure(f"LLM error during synthesis: {exc}")

        return self._parse_final_result(final_raw, tools_used)

    # ── LLM Abstraction ──────────────────────────────────────────

    async def _llm_chat(self, messages: list[dict[str, str]]) -> str:
        """Gemini-first, Ollama-fallback — mirrors Brain's pattern."""
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                result = await gemini.chat(messages)
                if result and result.strip():
                    return result
        except Exception:
            pass

        from bantz.llm.ollama import ollama
        return await ollama.chat(messages)

    # ── Parsing Helpers ──────────────────────────────────────────

    def _format_task_prompt(self, task: str, context: dict[str, Any]) -> str:
        """Format the user prompt with task and any injected context."""
        parts = [f"Task: {task}"]
        if context:
            ctx_str = json.dumps(context, ensure_ascii=False, indent=2)
            parts.append(f"\nContext from the orchestrator:\n```json\n{ctx_str}\n```")
        return "\n".join(parts)

    def _parse_tool_calls(self, raw: str) -> list[dict[str, Any]]:
        """Extract tool call instructions from LLM response.

        Expected format (inside a JSON array):
        [{"tool": "web_search", "args": {"query": "..."}}]

        Also handles the response being wrapped in markdown code fences.
        """
        # Strip thinking blocks
        raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)

        # Try to find JSON array
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    return [
                        item for item in parsed
                        if isinstance(item, dict) and "tool" in item
                    ]
            except json.JSONDecodeError:
                pass

        # Try JSON object with "tools" key
        obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group())
                if isinstance(parsed, dict):
                    tools = parsed.get("tools") or parsed.get("tool_calls", [])
                    if isinstance(tools, list):
                        return [t for t in tools if isinstance(t, dict) and "tool" in t]
            except json.JSONDecodeError:
                pass

        # No tool calls — pure reasoning response
        return []

    def _parse_final_result(
        self, raw: str, tools_used: list[str],
    ) -> SubAgentResult:
        """Parse the LLM's final synthesis into a SubAgentResult."""
        # Strip thinking/code fences
        cleaned = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*", "", cleaned)

        # Try JSON parse
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict):
                    return SubAgentResult(
                        success=True,
                        summary=parsed.get("summary", raw.strip()),
                        data=parsed.get("data", {}),
                        tools_used=tools_used,
                    )
            except json.JSONDecodeError:
                pass

        # Fallback: use raw text as summary
        return SubAgentResult(
            success=True,
            summary=cleaned.strip() or raw.strip(),
            tools_used=tools_used,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Specialized Roles
# ═══════════════════════════════════════════════════════════════════════════

class ResearcherAgent(SubAgent):
    """Specialised in web search, reading, and information synthesis.

    Has access to search, URL reading, and summarisation tools.
    No shell, filesystem, or destructive capabilities.
    """

    role = "researcher"
    display_name = "Researcher"
    allowed_tools = {
        "web_search",
        "read_url",
        "summarizer",
        "news",
        "feed",
    }

    system_prompt = """\
You are a Research Agent — a specialised sub-agent within the Bantz system.
Your job is to find accurate, up-to-date information using web search and
URL reading tools.

RULES:
1. Always verify claims using multiple sources when possible.
2. Cite URLs in your findings.
3. Prefer recent sources over old ones.
4. If the search returns no useful results, say so honestly.
5. NEVER fabricate information — if you can't find it, say "not found".

TOOL USAGE:
To use tools, respond with a JSON array of tool calls:
[{"tool": "web_search", "args": {"query": "your search query"}}]

Available tools: web_search, read_url, summarizer, news, feed.

After receiving tool results, synthesise a clear, factual answer.
"""

    def _build_system(self, task: str, context: dict[str, Any]) -> str:
        parts = [self.system_prompt]
        if context.get("user_language"):
            parts.append(f"\nThe user's language is: {context['user_language']}")
        if context.get("prior_findings"):
            parts.append(f"\nPrior findings: {context['prior_findings']}")
        return "\n".join(parts)


class DeveloperAgent(SubAgent):
    """Specialised in code execution, file operations, and system commands.

    Has access to shell, filesystem, and document tools.
    Shell commands are sandboxed — no destructive system operations.
    """

    role = "developer"
    display_name = "Developer"
    allowed_tools = {
        "shell",
        "filesystem",
        "document",
        "system_info",
        "summarizer",
    }

    system_prompt = """\
You are a Developer Agent — a specialised sub-agent within the Bantz system.
Your job is to write code, run shell commands, and handle file operations.

RULES:
1. Write clean, idiomatic code with comments.
2. Test your work — run the code to verify it works before reporting.
3. Handle errors gracefully and report what went wrong.
4. NEVER run destructive commands (rm -rf, format, etc.) without explicit confirmation in the task.
5. Prefer small, focused scripts over monolithic ones.

TOOL USAGE:
To use tools, respond with a JSON array of tool calls:
[{"tool": "shell", "args": {"command": "ls -la"}}]

Available tools: shell, filesystem, document, system_info, summarizer.

After executing tools, report what you did and the outcome.
"""

    def _build_system(self, task: str, context: dict[str, Any]) -> str:
        parts = [self.system_prompt]
        if context.get("working_directory"):
            parts.append(f"\nWorking directory: {context['working_directory']}")
        if context.get("language"):
            parts.append(f"\nPreferred language: {context['language']}")
        if context.get("existing_code"):
            parts.append(f"\nExisting code context:\n{context['existing_code']}")
        return "\n".join(parts)


class ReviewerAgent(SubAgent):
    """Specialised in analysis, validation, and quality checking.

    Has read-only access to filesystem and web for verification.
    No execution or modification capabilities.
    """

    role = "reviewer"
    display_name = "Reviewer"
    allowed_tools = {
        "filesystem",
        "read_url",
        "web_search",
        "summarizer",
    }
    max_tool_calls = 3  # reviewers should be focused

    system_prompt = """\
You are a Reviewer Agent — a specialised sub-agent within the Bantz system.
Your job is to check work quality, validate information, and provide
constructive feedback.

RULES:
1. Be thorough but concise in your review.
2. Check for correctness, completeness, and potential issues.
3. Provide specific, actionable feedback — not vague complaints.
4. If reviewing code, look for bugs, security issues, and edge cases.
5. If reviewing research, check source reliability and factual accuracy.
6. Rate confidence as: HIGH, MEDIUM, or LOW with justification.

TOOL USAGE:
To verify facts or read files, respond with a JSON array of tool calls:
[{"tool": "web_search", "args": {"query": "verify claim X"}}]

Available tools: filesystem (read-only), read_url, web_search, summarizer.

After reviewing, provide a structured assessment with:
- verdict: "pass" | "fail" | "needs_revision"
- issues: list of specific problems found
- suggestions: list of improvements
"""

    def _build_system(self, task: str, context: dict[str, Any]) -> str:
        parts = [self.system_prompt]
        if context.get("artifact"):
            parts.append(f"\nContent to review:\n{context['artifact']}")
        if context.get("criteria"):
            parts.append(f"\nReview criteria: {context['criteria']}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Role Registry
# ═══════════════════════════════════════════════════════════════════════════

# All available sub-agent roles — keyed by their role identifier
AGENT_ROLES: dict[str, type[SubAgent]] = {
    "researcher": ResearcherAgent,
    "developer": DeveloperAgent,
    "reviewer": ReviewerAgent,
}

# Human-friendly aliases for flexible delegation
ROLE_ALIASES: dict[str, str] = {
    "research": "researcher",
    "search": "researcher",
    "lookup": "researcher",
    "find": "researcher",
    "dev": "developer",
    "coder": "developer",
    "code": "developer",
    "programmer": "developer",
    "sysadmin": "developer",
    "review": "reviewer",
    "check": "reviewer",
    "validate": "reviewer",
    "audit": "reviewer",
    "verify": "reviewer",
}


def resolve_role(name: str) -> str | None:
    """Resolve a role name (or alias) to a canonical role identifier.

    Returns None if the role is not recognised.
    """
    norm = name.strip().lower().replace("-", "_").replace(" ", "_")
    if norm in AGENT_ROLES:
        return norm
    return ROLE_ALIASES.get(norm)


def create_agent(role: str) -> SubAgent | None:
    """Create an ephemeral sub-agent instance for the given role.

    Returns None if the role is not recognised.
    """
    canonical = resolve_role(role)
    if canonical is None:
        return None
    cls = AGENT_ROLES[canonical]
    return cls()


def available_roles() -> list[dict[str, str]]:
    """Return list of available roles with display names for LLM routing."""
    return [
        {"role": role, "display_name": cls.display_name}
        for role, cls in AGENT_ROLES.items()
    ]
