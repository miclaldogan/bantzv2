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
    ])
    def test_detects_refusals(self, text):
        assert is_refusal(text) is True

    @pytest.mark.parametrize("text", [
        "Here's the weather for Istanbul.",
        "The capital of France is Paris.",
        "I'll check your schedule right away.",
        "",
    ])
    def test_non_refusals(self, text):
        assert is_refusal(text) is False

    def test_case_insensitive(self):
        assert is_refusal("SORRY, I CAN'T ASSIST WITH THAT") is True

    def test_whitespace_handling(self):
        assert is_refusal("  sorry  ") is True


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
