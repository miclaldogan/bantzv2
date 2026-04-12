"""Tests for bantz.agent.sub_agent — Issue #321 SubAgent base + Roles.

Covers:
  1. SubAgentResult — creation, failure shorthand, to_dict serialisation
  2. Role resolution — canonical names, aliases, unknown roles
  3. Agent creation — factory, allowed_tools, system prompt building
  4. SubAgent.run() — 3-phase pipeline with mocked LLM and tools
  5. Parsing helpers — tool call extraction, final result parsing
  6. Tool access restriction — denied tools produce errors
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bantz.agent.sub_agent import (
    AGENT_ROLES,
    ROLE_ALIASES,
    DeveloperAgent,
    ResearcherAgent,
    ReviewerAgent,
    SubAgent,
    SubAgentResult,
    available_roles,
    create_agent,
    resolve_role,
)


# ═══════════════════════════════════════════════════════════════════════════
# SubAgentResult
# ═══════════════════════════════════════════════════════════════════════════

class TestSubAgentResult:
    """Result dataclass and factory methods."""

    def test_success_result(self):
        """Successful result with summary and data."""
        r = SubAgentResult(
            success=True,
            summary="GDP is $1T",
            data={"gdp": 1_000_000_000_000},
            tools_used=["web_search"],
        )
        assert r.success is True
        assert "GDP" in r.summary
        assert r.data["gdp"] == 1_000_000_000_000
        assert r.error == ""

    def test_failure_shorthand(self):
        """SubAgentResult.failure() convenience method."""
        r = SubAgentResult.failure("LLM timeout")
        assert r.success is False
        assert r.summary == ""
        assert r.error == "LLM timeout"
        assert r.tools_used == []

    def test_to_dict_roundtrip(self):
        """to_dict produces a JSON-serialisable dict."""
        r = SubAgentResult(
            success=True,
            summary="ok",
            data={"key": "val"},
            tools_used=["shell"],
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["summary"] == "ok"
        assert d["data"] == {"key": "val"}
        assert d["tools_used"] == ["shell"]
        # Ensure it's JSON-serialisable
        json.dumps(d)

    def test_default_fields(self):
        """Defaults are empty collections."""
        r = SubAgentResult(success=True, summary="done")
        assert r.data == {}
        assert r.tools_used == []
        assert r.error == ""


# ═══════════════════════════════════════════════════════════════════════════
# Role Resolution
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveRole:
    """resolve_role() maps names and aliases to canonical identifiers."""

    def test_canonical_names(self):
        """Direct role names resolve to themselves."""
        assert resolve_role("researcher") == "researcher"
        assert resolve_role("developer") == "developer"
        assert resolve_role("reviewer") == "reviewer"

    def test_aliases(self):
        """Common aliases resolve correctly."""
        assert resolve_role("dev") == "developer"
        assert resolve_role("search") == "researcher"
        assert resolve_role("code") == "developer"
        assert resolve_role("check") == "reviewer"
        assert resolve_role("verify") == "reviewer"
        assert resolve_role("audit") == "reviewer"

    def test_case_insensitive(self):
        """Resolution is case-insensitive."""
        assert resolve_role("RESEARCHER") == "researcher"
        assert resolve_role("Dev") == "developer"
        assert resolve_role("REVIEW") == "reviewer"

    def test_normalisation(self):
        """Leading/trailing whitespace is stripped."""
        assert resolve_role(" researcher ") == "researcher"
        assert resolve_role(" dev ") == "developer"
        assert resolve_role("  review  ") == "reviewer"

    def test_unknown_returns_none(self):
        """Unrecognised roles return None."""
        assert resolve_role("plumber") is None
        assert resolve_role("") is None
        assert resolve_role("xyz_unknown") is None


class TestCreateAgent:
    """create_agent() factory function."""

    def test_creates_researcher(self):
        agent = create_agent("researcher")
        assert isinstance(agent, ResearcherAgent)
        assert agent.role == "researcher"

    def test_creates_developer(self):
        agent = create_agent("developer")
        assert isinstance(agent, DeveloperAgent)

    def test_creates_reviewer(self):
        agent = create_agent("reviewer")
        assert isinstance(agent, ReviewerAgent)

    def test_alias_creates_agent(self):
        """create_agent works with aliases too."""
        agent = create_agent("dev")
        assert isinstance(agent, DeveloperAgent)

    def test_unknown_returns_none(self):
        assert create_agent("martian") is None

    def test_available_roles_list(self):
        """available_roles() returns role dicts."""
        roles = available_roles()
        assert len(roles) == 3
        role_names = {r["role"] for r in roles}
        assert role_names == {"researcher", "developer", "reviewer"}
        # Each has display_name
        for r in roles:
            assert "display_name" in r


# ═══════════════════════════════════════════════════════════════════════════
# Agent Properties
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentProperties:
    """Specialised agent configurations."""

    def test_researcher_tools(self):
        agent = ResearcherAgent()
        assert "web_search" in agent.allowed_tools
        assert "read_url" in agent.allowed_tools
        assert "shell" not in agent.allowed_tools

    def test_developer_tools(self):
        agent = DeveloperAgent()
        assert "shell" in agent.allowed_tools
        assert "filesystem" in agent.allowed_tools
        assert "web_search" not in agent.allowed_tools

    def test_reviewer_tools(self):
        agent = ReviewerAgent()
        assert "web_search" in agent.allowed_tools
        assert "shell" not in agent.allowed_tools
        assert agent.max_tool_calls == 3

    def test_reviewer_max_tool_calls_lower(self):
        """Reviewer has a stricter circuit breaker than default."""
        researcher = ResearcherAgent()
        reviewer = ReviewerAgent()
        assert reviewer.max_tool_calls < researcher.max_tool_calls


class TestBuildSystem:
    """_build_system() injects task-specific context into prompts."""

    def test_researcher_with_language_context(self):
        agent = ResearcherAgent()
        prompt = agent._build_system("Find GDP", {"user_language": "Turkish"})
        assert "Turkish" in prompt
        assert "Research Agent" in prompt

    def test_developer_with_workdir(self):
        agent = DeveloperAgent()
        prompt = agent._build_system("Write script", {"working_directory": "/tmp"})
        assert "/tmp" in prompt

    def test_reviewer_with_artifact(self):
        agent = ReviewerAgent()
        prompt = agent._build_system("Check code", {"artifact": "def foo(): pass"})
        assert "def foo(): pass" in prompt

    def test_base_prompt_always_present(self):
        """System prompt text is always included even without context."""
        agent = ResearcherAgent()
        prompt = agent._build_system("task", {})
        assert "Research Agent" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# Parsing Helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestParseToolCalls:
    """_parse_tool_calls() extracts structured tool calls from LLM text."""

    def test_json_array(self):
        """Standard JSON array of tool calls."""
        agent = ResearcherAgent()
        raw = '[{"tool": "web_search", "args": {"query": "hello"}}]'
        calls = agent._parse_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0]["tool"] == "web_search"

    def test_embedded_in_text(self):
        """Tool calls buried in prose text."""
        agent = ResearcherAgent()
        raw = (
            "I will search for that.\n"
            '[{"tool": "web_search", "args": {"query": "GDP Turkey"}}]\n'
            "Let me know if you need more."
        )
        calls = agent._parse_tool_calls(raw)
        assert len(calls) == 1

    def test_thinking_block_stripped(self):
        """<thinking> blocks are removed before parsing."""
        agent = ResearcherAgent()
        raw = (
            "<thinking>I should search</thinking>\n"
            '[{"tool": "web_search", "args": {"query": "test"}}]'
        )
        calls = agent._parse_tool_calls(raw)
        assert len(calls) == 1

    def test_no_tools_returns_empty(self):
        """Pure reasoning with no JSON → empty list."""
        agent = ResearcherAgent()
        raw = "I already know the answer. The GDP is $1T."
        calls = agent._parse_tool_calls(raw)
        assert calls == []

    def test_tools_key_in_object(self):
        """Tool calls inside a JSON object with 'tools' key."""
        agent = ResearcherAgent()
        raw = '{"tools": [{"tool": "news", "args": {"query": "latest"}}]}'
        calls = agent._parse_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0]["tool"] == "news"

    def test_multiple_calls(self):
        """Multiple tool calls in one array."""
        agent = ResearcherAgent()
        raw = json.dumps([
            {"tool": "web_search", "args": {"query": "a"}},
            {"tool": "read_url", "args": {"url": "http://x.com"}},
        ])
        calls = agent._parse_tool_calls(raw)
        assert len(calls) == 2


class TestParseFinalResult:
    """_parse_final_result() converts LLM synthesis text to SubAgentResult."""

    def test_valid_json(self):
        """Clean JSON response is parsed correctly."""
        agent = ResearcherAgent()
        raw = '{"summary": "GDP is X", "data": {"gdp": 123}}'
        result = agent._parse_final_result(raw, ["web_search"])
        assert result.success is True
        assert "GDP" in result.summary
        assert result.data["gdp"] == 123
        assert result.tools_used == ["web_search"]

    def test_json_in_code_fence(self):
        """JSON wrapped in markdown code fences."""
        agent = ResearcherAgent()
        raw = '```json\n{"summary": "result"}\n```'
        result = agent._parse_final_result(raw, [])
        assert result.success is True
        assert result.summary == "result"

    def test_fallback_to_raw_text(self):
        """Non-JSON text becomes the summary."""
        agent = ResearcherAgent()
        raw = "I couldn't find the GDP data."
        result = agent._parse_final_result(raw, [])
        assert result.success is True
        assert "couldn't find" in result.summary

    def test_thinking_block_stripped(self):
        """Thinking blocks are removed before parsing."""
        agent = ResearcherAgent()
        raw = '<thinking>hmm</thinking>{"summary": "clean result"}'
        result = agent._parse_final_result(raw, [])
        assert result.summary == "clean result"


# ═══════════════════════════════════════════════════════════════════════════
# SubAgent.run() — Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestSubAgentRun:
    """Integration tests for the 3-phase run() pipeline."""

    @pytest.mark.asyncio
    async def test_run_with_tools(self):
        """Full cycle: plan → execute tool → synthesise."""
        agent = ResearcherAgent()

        # Phase 1: LLM returns a tool call plan
        plan_response = '[{"tool": "web_search", "args": {"query": "Turkey GDP 2025"}}]'
        # Phase 3: LLM returns a synthesis
        synth_response = '{"summary": "Turkey GDP is $1T in 2025"}'

        # Mock LLM to return plan first, then synthesis
        call_count = 0

        async def mock_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return plan_response
            return synth_response

        # Mock the tool
        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=MagicMock(
            success=True,
            output="Turkey GDP: $1,000,000,000,000 (2025 estimate)",
        ))

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_tool

        with patch.object(agent, "_llm_chat", side_effect=mock_chat), \
             patch("bantz.tools.registry", mock_registry):
            result = await agent.run("Find Turkey GDP 2025")

        assert result.success is True
        assert "1T" in result.summary
        assert "web_search" in result.tools_used

    @pytest.mark.asyncio
    async def test_run_no_tools_needed(self):
        """LLM can answer directly without tools."""
        agent = ResearcherAgent()

        # Phase 1: no tool calls in the response
        plan_response = "I know this already — the answer is 42."
        # Phase 3: synthesis
        synth_response = '{"summary": "The answer is 42."}'

        call_count = 0

        async def mock_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return plan_response
            return synth_response

        with patch.object(agent, "_llm_chat", side_effect=mock_chat):
            result = await agent.run("What is the meaning of life?")

        assert result.success is True
        assert "42" in result.summary
        assert result.tools_used == []

    @pytest.mark.asyncio
    async def test_run_llm_plan_failure(self):
        """LLM failure during planning returns error result."""
        agent = ResearcherAgent()

        async def mock_chat_fail(messages):
            raise ConnectionError("Ollama unreachable")

        with patch.object(agent, "_llm_chat", side_effect=mock_chat_fail):
            result = await agent.run("search something")

        assert result.success is False
        assert "LLM error" in result.error

    @pytest.mark.asyncio
    async def test_run_tool_denied(self):
        """Tools outside allowed_tools are rejected."""
        agent = ResearcherAgent()  # doesn't have "shell"

        plan_response = '[{"tool": "shell", "args": {"command": "rm -rf /"}}]'
        synth_response = '{"summary": "Tool was denied"}'

        call_count = 0

        async def mock_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return plan_response
            return synth_response

        with patch.object(agent, "_llm_chat", side_effect=mock_chat):
            result = await agent.run("Delete everything")

        assert result.success is True  # synth still succeeds
        assert "shell" not in result.tools_used

    @pytest.mark.asyncio
    async def test_run_circuit_breaker(self):
        """Max tool calls limit is respected."""
        agent = ReviewerAgent()  # max_tool_calls = 3

        # Ask for 10 tool calls — only 3 should execute
        calls = [{"tool": "web_search", "args": {"query": f"q{i}"}} for i in range(10)]
        plan_response = json.dumps(calls)
        synth_response = '{"summary": "done"}'

        call_count = 0

        async def mock_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return plan_response
            return synth_response

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=MagicMock(
            success=True, output="result",
        ))

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_tool

        with patch.object(agent, "_llm_chat", side_effect=mock_chat), \
             patch("bantz.tools.registry", mock_registry):
            result = await agent.run("Run many searches")

        # Only 3 calls should have been made
        assert mock_tool.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_run_synthesis_fails_but_tools_succeed(self):
        """If synthesis LLM fails but tools worked, return tool results."""
        agent = ResearcherAgent()

        plan_response = '[{"tool": "web_search", "args": {"query": "test"}}]'

        call_count = 0

        async def mock_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return plan_response
            raise RuntimeError("Synthesis failed")

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=MagicMock(
            success=True, output="Search result: found it",
        ))

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_tool

        with patch.object(agent, "_llm_chat", side_effect=mock_chat), \
             patch("bantz.tools.registry", mock_registry):
            result = await agent.run("search test")

        # Should still succeed with combined tool output
        assert result.success is True
        assert "web_search" in result.tools_used


# ═══════════════════════════════════════════════════════════════════════════
# LLM Abstraction
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMChat:
    """_llm_chat() — Gemini-first, Ollama-fallback."""

    @pytest.mark.asyncio
    async def test_gemini_primary(self):
        """Uses Gemini when available."""
        agent = ResearcherAgent()
        mock_gemini = MagicMock()
        mock_gemini.is_enabled.return_value = True
        mock_gemini.chat = AsyncMock(return_value="gemini answer")

        with patch("bantz.agent.sub_agent.gemini", mock_gemini, create=True), \
             patch.dict("sys.modules", {"bantz.llm.gemini": MagicMock(gemini=mock_gemini)}):
            result = await agent._llm_chat([{"role": "user", "content": "hi"}])

        assert result == "gemini answer"

    @pytest.mark.asyncio
    async def test_ollama_fallback(self):
        """Falls back to Ollama if Gemini fails."""
        agent = ResearcherAgent()
        mock_ollama = MagicMock()
        mock_ollama.chat = AsyncMock(return_value="ollama answer")

        # Gemini import fails
        with patch("bantz.llm.gemini.gemini", side_effect=ImportError), \
             patch("bantz.llm.ollama.ollama", mock_ollama):
            result = await agent._llm_chat([{"role": "user", "content": "hi"}])

        assert result == "ollama answer"


# ═══════════════════════════════════════════════════════════════════════════
# Role Registry
# ═══════════════════════════════════════════════════════════════════════════

class TestRoleRegistry:
    """AGENT_ROLES and ROLE_ALIASES consistency."""

    def test_all_aliases_point_to_valid_roles(self):
        """Every alias maps to a real role in AGENT_ROLES."""
        for alias, canonical in ROLE_ALIASES.items():
            assert canonical in AGENT_ROLES, f"Alias '{alias}' → '{canonical}' not in AGENT_ROLES"

    def test_no_duplicate_aliases(self):
        """No alias collides with a canonical role name."""
        for alias in ROLE_ALIASES:
            # Aliases that ARE canonical names are allowed (e.g. "research" vs "researcher")
            # but an alias shouldn't shadow a different canonical
            canonical = ROLE_ALIASES[alias]
            if alias in AGENT_ROLES:
                assert AGENT_ROLES[alias] == AGENT_ROLES.get(canonical), \
                    f"Alias '{alias}' shadows a different canonical role"

    def test_three_roles_registered(self):
        assert len(AGENT_ROLES) == 3
        assert set(AGENT_ROLES.keys()) == {"researcher", "developer", "reviewer"}
