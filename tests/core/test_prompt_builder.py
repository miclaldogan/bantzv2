"""
Tests for ``bantz.core.prompt_builder`` (#227).

Covers:
  - CHAT_SYSTEM template has all required placeholders
  - COMMAND_SYSTEM is a non-empty string
  - build_chat_system() renders correctly from BantzContext
  - is_refusal() detects safety refusals
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

try:
    from bantz.core.context import BantzContext
except ImportError:

    @dataclass
    class BantzContext:  # type: ignore[no-redef]
        en_input: str = ""
        graph_context: str = ""
        vector_context: str = ""
        deep_memory: str = ""
        desktop_context: str = ""
        persona_state: str = ""
        formality_hint: str = ""
        style_hint: str = ""
        profile_hint: str = ""
        feedback_hint: str = ""


from bantz.core.prompt_builder import (
    CHAT_SYSTEM,
    COMMAND_SYSTEM,
    COMPUTER_USE_AUTHORIZATION,
    build_chat_system,
    is_refusal,
)


# ═══════════════════════════════════════════════════════════════════════════
# Template sanity
# ═══════════════════════════════════════════════════════════════════════════


class TestTemplates:
    def test_chat_system_has_all_placeholders(self):
        """All expected {name} placeholders must be present in CHAT_SYSTEM."""
        required = [
            "persona_state", "style_hint", "formality_hint",
            "time_hint", "profile_hint", "graph_hint",
            "vector_hint", "deep_memory", "desktop_hint",
        ]
        for name in required:
            assert f"{{{name}}}" in CHAT_SYSTEM, f"Missing placeholder: {name}"

    def test_chat_system_contains_character_description(self):
        assert "Bantz" in CHAT_SYSTEM
        assert "1920s" in CHAT_SYSTEM

    def test_command_system_non_empty(self):
        assert len(COMMAND_SYSTEM) > 50
        assert "bash" in COMMAND_SYSTEM.lower()

    def test_command_system_no_placeholders(self):
        """COMMAND_SYSTEM should have no {…} format placeholders."""
        import re
        placeholders = re.findall(r"\{[a-z_]+\}", COMMAND_SYSTEM)
        assert placeholders == [], f"Unexpected placeholders: {placeholders}"


# ═══════════════════════════════════════════════════════════════════════════
# Brevity rules (#274)
# ═══════════════════════════════════════════════════════════════════════════


class TestBrevityRules:
    """Verify that all system prompts enforce brevity constraints."""

    def test_chat_system_has_brevity_rule(self):
        """CHAT_SYSTEM must contain a brevity directive."""
        lower = CHAT_SYSTEM.lower()
        assert "brevity" in lower or "concise" in lower or "brief" in lower

    def test_chat_system_preserves_persona_through_tone(self):
        """Brevity rule allows persona via word choice, not length."""
        assert "vocabulary" in CHAT_SYSTEM or "word choice" in CHAT_SYSTEM \
            or "elegant" in CHAT_SYSTEM

    def test_chat_system_has_escape_hatch(self):
        """Users can still get long answers when they ask for depth."""
        lower = CHAT_SYSTEM.lower()
        assert "explain" in lower or "analyze" in lower or "demands depth" in lower

    def test_finalizer_system_has_dynamic_limit(self):
        """FINALIZER_SYSTEM uses a flexible sentence limit, not a fixed 5."""
        from bantz.core.finalizer import FINALIZER_SYSTEM
        assert "3–5" in FINALIZER_SYSTEM or "3-5" in FINALIZER_SYSTEM

    def test_bantz_chat_has_brevity(self):
        """BANTZ_CHAT persona template must include brevity instruction."""
        from bantz.personality.system_prompt import BANTZ_CHAT
        lower = BANTZ_CHAT.lower()
        assert "crisp" in lower or "brevity" in lower or "brief" in lower

    def test_bantz_chat_persona_through_word_choice(self):
        """BANTZ_CHAT must express persona through word choice, not length."""
        from bantz.personality.system_prompt import BANTZ_CHAT
        assert "word choice" in BANTZ_CHAT or "elegant" in BANTZ_CHAT

    def test_bantz_finalizer_has_dynamic_limit(self):
        """BANTZ_FINALIZER uses flexible sentence limit."""
        from bantz.personality.system_prompt import BANTZ_FINALIZER
        assert "3–5" in BANTZ_FINALIZER or "3-5" in BANTZ_FINALIZER


# ═══════════════════════════════════════════════════════════════════════════
# build_chat_system
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildChatSystem:
    def _make_ctx(self, **overrides) -> BantzContext:
        defaults = dict(
            graph_context="[graph] Alice knows Bob",
            vector_context="[vec 0.9] hello there",
            deep_memory="You love cats",
            desktop_context="Active: Firefox",
            persona_state="Curious mood",
            formality_hint="\n[Bonding level] warm",
            style_hint="Tone: casual, friendly. Address the user as 'boss'.",
            profile_hint="Name: Tester",
            feedback_hint="",
        )
        defaults.update(overrides)
        return BantzContext(**defaults)

    def test_renders_all_fields(self):
        ctx = self._make_ctx()
        tc = {"prompt_hint": "Tuesday morning"}
        result = build_chat_system(ctx, tc)
        assert "Tuesday morning" in result
        assert "Alice knows Bob" in result
        assert "hello there" in result
        assert "You love cats" in result
        assert "Firefox" in result
        assert "Curious mood" in result
        assert "warm" in result
        assert "casual" in result
        assert "Tester" in result

    def test_appends_feedback_hint(self):
        ctx = self._make_ctx(feedback_hint="\n[RLHF] user liked the last response")
        tc = {"prompt_hint": ""}
        result = build_chat_system(ctx, tc)
        assert "[RLHF] user liked the last response" in result

    def test_no_feedback_hint(self):
        ctx = self._make_ctx(feedback_hint="")
        tc = {"prompt_hint": ""}
        result = build_chat_system(ctx, tc)
        # Should still be valid without feedback
        assert "Bantz" in result

    def test_empty_fields_produce_valid_output(self):
        ctx = self._make_ctx(
            graph_context="", vector_context="", deep_memory="",
            desktop_context="", persona_state="", formality_hint="",
            style_hint="", profile_hint="",
        )
        tc = {"prompt_hint": ""}
        result = build_chat_system(ctx, tc)
        # Template renders without error; character preamble is intact
        assert "Bantz" in result
        assert "1920s" in result

    def test_missing_prompt_hint_key_uses_empty(self):
        ctx = self._make_ctx()
        tc = {}  # no prompt_hint key
        result = build_chat_system(ctx, tc)
        assert "Bantz" in result  # renders fine with empty time_hint


# ═══════════════════════════════════════════════════════════════════════════
# Computer-Use Authorization (#185)
# ═══════════════════════════════════════════════════════════════════════════


class TestComputerUseAuthorization:
    """System prompt gets the physical-interaction authorization block."""

    def _make_ctx(self, **overrides) -> BantzContext:
        defaults = dict(
            graph_context="", vector_context="", deep_memory="",
            desktop_context="", persona_state="", formality_hint="",
            style_hint="", profile_hint="", feedback_hint="",
        )
        defaults.update(overrides)
        return BantzContext(**defaults)

    def test_authorization_constant_has_key_directives(self):
        assert "AUTHORIZED" in COMPUTER_USE_AUTHORIZATION
        assert "visual_click" in COMPUTER_USE_AUTHORIZATION
        assert "NO REFUSALS" in COMPUTER_USE_AUTHORIZATION
        assert "VISUAL GROUNDING" in COMPUTER_USE_AUTHORIZATION

    def test_authorization_injected_when_input_control_enabled(self):
        from unittest.mock import patch
        ctx = self._make_ctx()
        tc = {"prompt_hint": ""}
        with patch("bantz.config.config") as mock_config:
            mock_config.input_control_enabled = True
            result = build_chat_system(ctx, tc)
        assert "AUTHORIZED" in result
        assert "visual_click" in result

    def test_authorization_absent_when_input_control_disabled(self):
        from unittest.mock import patch
        ctx = self._make_ctx()
        tc = {"prompt_hint": ""}
        with patch("bantz.config.config") as mock_config:
            mock_config.input_control_enabled = False
            result = build_chat_system(ctx, tc)
        assert "COMPUTER USE" not in result
        assert "AUTHORIZED" not in result

    def test_rule_6_no_longer_blocks_visual_grounding(self):
        """Rule 6 should NOT say 'use ONLY the Desktop Context'."""
        assert "use ONLY the Desktop Context" not in CHAT_SYSTEM


# ═══════════════════════════════════════════════════════════════════════════
# is_refusal
# ═══════════════════════════════════════════════════════════════════════════


class TestIsRefusal:
    @pytest.mark.parametrize("text", [
        "Sorry, I can't assist with that.",
        "I cannot provide that information.",
        "I'm unable to help with this request.",
        "That would be inappropriate.",
        "I'm not able to do that.",
        "Sorry, I cannot do that.",
    ])
    def test_detects_refusals(self, text):
        assert is_refusal(text) is True

    @pytest.mark.parametrize("text", [
        "Here's the weather for Istanbul.",
        "The capital of France is Paris.",
        "I'll check your schedule right away.",
        "",
        "sorry",  # bare 'sorry' is NOT a refusal (#282)
        "I'm sorry, let me reconsider.",  # CoT reasoning
    ])
    def test_non_refusals(self, text):
        assert is_refusal(text) is False

    def test_case_insensitive(self):
        assert is_refusal("SORRY, I CAN'T ASSIST WITH THAT") is True

    def test_whitespace_handling(self):
        assert is_refusal("  i cannot provide that  ") is True


# ═══════════════════════════════════════════════════════════════════════════
# Backward compatibility — brain re-exports
# ═══════════════════════════════════════════════════════════════════════════


class TestBrainReExports:
    """Verify that existing imports from brain.py still work."""

    def test_chat_system_available_from_brain(self):
        from bantz.core.brain import CHAT_SYSTEM as brain_cs
        assert brain_cs is CHAT_SYSTEM

    def test_command_system_available_from_brain(self):
        from bantz.core.brain import COMMAND_SYSTEM as brain_cmd
        assert brain_cmd is COMMAND_SYSTEM

    def test_is_refusal_available_from_brain(self):
        from bantz.core.brain import _is_refusal as brain_ir
        assert brain_ir is is_refusal


# ═══════════════════════════════════════════════════════════════════════════
# Tool description guardrails (#275)
# ═══════════════════════════════════════════════════════════════════════════


class TestToolDescriptionGuardrails:
    """Verify all tool descriptions contain required guardrail keywords."""

    def test_gmail_has_follow_up_rule(self):
        from bantz.tools.gmail import GmailTool
        desc = GmailTool.description.lower()
        assert "follow-up" in desc or "message id" in desc

    def test_gmail_forbids_repeat_search(self):
        from bantz.tools.gmail import GmailTool
        assert "NEVER repeat" in GmailTool.description or "never repeat" in GmailTool.description.lower()

    def test_shell_has_absolute_path_rule(self):
        from bantz.tools.shell import ShellTool
        desc = ShellTool.description
        assert "absolute path" in desc.lower()

    def test_shell_has_dynamic_home_dir(self):
        """Shell description must contain actual home directory, not a placeholder."""
        import os
        from bantz.tools.shell import ShellTool
        assert os.path.expanduser("~") in ShellTool.description

    def test_shell_has_wrong_right_example(self):
        from bantz.tools.shell import ShellTool
        assert "WRONG:" in ShellTool.description and "RIGHT:" in ShellTool.description

    def test_filesystem_has_path_rule(self):
        from bantz.tools.filesystem import FilesystemTool
        desc = FilesystemTool.description.lower()
        assert "absolute path" in desc or "never guess" in desc

    def test_web_search_has_specificity_rule(self):
        from bantz.tools.web_search import WebSearchTool
        desc = WebSearchTool.description.lower()
        assert "specific" in desc or "vague" in desc

    def test_calendar_has_no_invent_rule(self):
        from bantz.tools.calendar import CalendarTool
        desc = CalendarTool.description.lower()
        assert "never invent" in desc

    def test_web_reader_has_url_validation_rule(self):
        from bantz.tools.web_reader import WebReaderTool
        desc = WebReaderTool.description.lower()
        assert "never fabricate" in desc or "valid url" in desc


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic tool context injection (#275)
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildToolContext:
    """Verify brain._build_tool_context injects data only when relevant."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        brain = object.__new__(Brain)
        brain._last_messages = []
        brain._last_events = []
        return brain

    def test_empty_when_no_data(self):
        brain = self._make_brain()
        assert brain._build_tool_context("what is the weather") == ""

    def test_empty_when_unrelated_query(self):
        brain = self._make_brain()
        brain._last_messages = [{"id": "abc", "from": "x@y.com", "subject": "Test"}]
        # Weather query should NOT inject email context
        assert brain._build_tool_context("what is the weather") == ""

    def test_injects_email_context_when_relevant(self):
        brain = self._make_brain()
        brain._last_messages = [
            {"id": "abc123", "from": "erasmus@uni.eu", "subject": "Application Status"},
        ]
        ctx = brain._build_tool_context("read that email")
        assert "abc123" in ctx
        assert "erasmus@uni.eu" in ctx
        assert "Application Status" in ctx

    def test_injects_calendar_context_when_relevant(self):
        brain = self._make_brain()
        brain._last_events = [
            {"id": "evt1", "summary": "Team Meeting", "start": "2026-03-16T14:00"},
        ]
        ctx = brain._build_tool_context("what meetings do I have today")
        assert "evt1" in ctx
        assert "Team Meeting" in ctx

    def test_no_cross_contamination(self):
        """Email data should NOT appear for calendar queries and vice versa."""
        brain = self._make_brain()
        brain._last_messages = [{"id": "m1", "from": "a@b.com", "subject": "Hi"}]
        brain._last_events = [{"id": "e1", "summary": "Standup", "start": "10:00"}]
        # Calendar query — should have events but NOT emails
        ctx = brain._build_tool_context("what's on my calendar today")
        assert "e1" in ctx
        assert "m1" not in ctx

    def test_limits_to_5_entries(self):
        brain = self._make_brain()
        brain._last_messages = [
            {"id": f"m{i}", "from": f"u{i}@x.com", "subject": f"Subj {i}"}
            for i in range(10)
        ]
        ctx = brain._build_tool_context("check my email")
        # Should only have 5 entries (m0-m4), not m5-m9
        assert "m4" in ctx
        assert "m5" not in ctx


# ═══════════════════════════════════════════════════════════════════════════
# Anti-hallucination safeguards (#282)
# ═══════════════════════════════════════════════════════════════════════════


class TestAntiHallucination:
    """Verify CHAT_SYSTEM no longer encourages tool hallucination in chat mode."""

    def test_no_pretend_tool_use_instruction(self):
        """Rule 1 must NOT say 'NEVER say you lack access' (old instruction)."""
        assert "NEVER say you lack access" not in CHAT_SYSTEM

    def test_no_act_as_if_sending_telegram(self):
        """Rule 1 must NOT say 'Act as if you are sending a telegram'."""
        assert "Act as if you are sending a telegram" not in CHAT_SYSTEM

    def test_grand_telegraph_conditional(self):
        """Grand Telegraph Archives reference should be conditional on real tool data."""
        # The new rule 1 only mentions Grand Telegraph when tools ALREADY returned data
        assert "ALREADY been used" in CHAT_SYSTEM or "already been used" in CHAT_SYSTEM

    def test_anti_hallucination_rule_present(self):
        """Rule 9 should explicitly say ANTI-HALLUCINATION / CHAT MODE."""
        assert "ANTI-HALLUCINATION" in CHAT_SYSTEM
        assert "CHAT MODE" in CHAT_SYSTEM

    def test_no_fabricate_parenthetical_tool_calls(self):
        """Rule 9 should prohibit parenthetical pseudo-tool calls."""
        assert "(queries Grand Telegraph Archives)" in CHAT_SYSTEM  # mentioned as forbidden
        assert "(calls visual_click)" in CHAT_SYSTEM  # mentioned as forbidden

    def test_stop_on_data_request(self):
        """Chat mode should say 'STOP' for data requests it can't fulfill."""
        # Both rule 1 and rule 9 should tell the LLM to STOP
        assert CHAT_SYSTEM.count("STOP") >= 2


class TestIsRefusalNoFalsePositives:
    """Ensure _is_refusal doesn't trigger on legitimate CoT reasoning (#282)."""

    def test_bare_sorry_not_refusal(self):
        assert is_refusal("sorry") is False

    def test_sorry_in_cot_reasoning_not_refusal(self):
        assert is_refusal("I'm sorry, let me reconsider the routing.") is False

    def test_sorry_apology_not_refusal(self):
        """Butler saying 'sorry for the delay' should not trigger."""
        assert is_refusal("Sorry for the delay, ma'am.") is False

    def test_real_refusal_still_detected(self):
        assert is_refusal("Sorry, I can't assist with that.") is True
        assert is_refusal("I cannot provide that information.") is True

    def test_thinking_block_with_sorry_not_refusal(self):
        """CoT output with 'sorry' inside thinking tags should not trigger."""
        from bantz.core.intent import _is_refusal as intent_is_refusal
        raw = '<thinking>I\'m sorry, I need to re-read the request.</thinking>{"route":"tool","tool_name":"gmail","tool_args":{}}'
        assert intent_is_refusal(raw) is False


# ═══════════════════════════════════════════════════════════════════════════
# Fuzzy tool registry (#282 follow-up)
# ═══════════════════════════════════════════════════════════════════════════


class TestFuzzyToolRegistry:
    """ToolRegistry.get() handles casing, spaces, and hyphens."""

    def test_exact_match(self):
        from bantz.tools import ToolRegistry, BaseTool, ToolResult

        class FakeTool(BaseTool):
            name = "web_search"
            description = "test"
            async def execute(self, **kw):
                return ToolResult(success=True, output="ok")

        reg = ToolRegistry()
        reg.register(FakeTool())
        assert reg.get("web_search") is not None

    def test_capitalized_with_space(self):
        """'Web Search' should resolve to 'web_search'."""
        from bantz.tools import ToolRegistry, BaseTool, ToolResult

        class FakeTool(BaseTool):
            name = "web_search"
            description = "test"
            async def execute(self, **kw):
                return ToolResult(success=True, output="ok")

        reg = ToolRegistry()
        reg.register(FakeTool())
        assert reg.get("Web Search") is not None
        assert reg.get("Web Search").name == "web_search"

    def test_hyphenated(self):
        """'web-search' should resolve to 'web_search'."""
        from bantz.tools import ToolRegistry, BaseTool, ToolResult

        class FakeTool(BaseTool):
            name = "web_search"
            description = "test"
            async def execute(self, **kw):
                return ToolResult(success=True, output="ok")

        reg = ToolRegistry()
        reg.register(FakeTool())
        assert reg.get("web-search") is not None

    def test_visual_click_variants(self):
        """'Visual Click', 'visual-click', 'VISUAL_CLICK' should all match."""
        from bantz.tools import ToolRegistry, BaseTool, ToolResult

        class FakeTool(BaseTool):
            name = "visual_click"
            description = "test"
            async def execute(self, **kw):
                return ToolResult(success=True, output="ok")

        reg = ToolRegistry()
        reg.register(FakeTool())
        for variant in ("Visual Click", "visual-click", "VISUAL_CLICK", "Visual_Click"):
            assert reg.get(variant) is not None, f"Failed for: {variant}"

    def test_miss_returns_none(self):
        from bantz.tools import ToolRegistry
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None


class TestCotPromptVisualClick:
    """CoT prompt must properly route click requests to visual_click."""

    def test_visual_click_in_parameter_reference(self):
        from bantz.core.intent import COT_SYSTEM
        assert "visual_click" in COT_SYSTEM
        assert '"target"' in COT_SYSTEM

    def test_visual_click_in_routing_rules(self):
        from bantz.core.intent import COT_SYSTEM
        assert "visual_click: click a button" in COT_SYSTEM

    def test_click_routes_to_visual_click_not_accessibility(self):
        """The routing rules should NOT say 'accessibility: click UI element'."""
        from bantz.core.intent import COT_SYSTEM
        assert "accessibility: click" not in COT_SYSTEM

    def test_snake_case_instruction(self):
        """CoT prompt must instruct models to use snake_case tool names."""
        from bantz.core.intent import COT_SYSTEM
        assert "snake_case" in COT_SYSTEM
