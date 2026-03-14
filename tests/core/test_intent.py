"""Tests for ``bantz.core.intent`` — CoT intent parser (#78, #212).

Covers:
  1. cot_route — basic routing (mocked Ollama)
  2. cot_route — recent_history injection for pronoun resolution (#212)
  3. _format_recent_history — formatting helper
  4. _is_refusal / _extract_json / _log_thinking — helpers
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# 1. cot_route — basic routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestCotRouteBasic:
    """cot_route returns routing plans from mocked LLM responses."""

    @pytest.mark.asyncio
    async def test_routes_weather_request(self):
        from bantz.core.intent import cot_route

        llm_response = json.dumps({
            "route": "tool",
            "tool_name": "weather",
            "tool_args": {"city": "Istanbul"},
            "risk_level": "safe",
            "confidence": 0.95,
        })

        with patch("bantz.core.intent.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=llm_response)
            plan, error = await cot_route("What is the weather in Istanbul?", [
                {"name": "weather", "description": "Check weather", "risk_level": "safe"},
            ])

        assert plan is not None
        assert error is None
        assert plan["tool_name"] == "weather"
        assert plan["tool_args"]["city"] == "Istanbul"

    @pytest.mark.asyncio
    async def test_chat_route_for_greeting(self):
        from bantz.core.intent import cot_route

        llm_response = json.dumps({
            "route": "chat",
            "tool_name": None,
            "tool_args": {},
            "risk_level": "safe",
            "confidence": 0.9,
        })

        with patch("bantz.core.intent.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=llm_response)
            plan, error = await cot_route("Hello there!", [])

        assert plan is not None
        assert error is None
        assert plan["route"] == "chat"

    @pytest.mark.asyncio
    async def test_returns_none_on_refusal(self):
        from bantz.core.intent import cot_route

        with patch("bantz.core.intent.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value="Sorry, I can't assist with that.")
            plan, error = await cot_route("do something bad", [])

        assert plan is None
        assert error is None  # refusal is not a routing error

    @pytest.mark.asyncio
    async def test_returns_none_on_low_confidence(self):
        from bantz.core.intent import cot_route

        llm_response = json.dumps({
            "route": "tool",
            "tool_name": "shell",
            "tool_args": {"command": "rm -rf /"},
            "risk_level": "destructive",
            "confidence": 0.1,
        })

        with patch("bantz.core.intent.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=llm_response)
            plan, error = await cot_route("maybe delete everything", [],
                                   confidence_threshold=0.4)

        assert plan is None
        assert error is None  # low confidence is not a routing error

    @pytest.mark.asyncio
    async def test_handles_thinking_block(self):
        """LLM response with <thinking> block is stripped before JSON parse."""
        from bantz.core.intent import cot_route

        llm_response = (
            '<thinking>The user wants weather info for London.</thinking>\n'
            + json.dumps({
                "route": "tool",
                "tool_name": "weather",
                "tool_args": {"city": "London"},
                "risk_level": "safe",
                "confidence": 0.9,
            })
        )

        with patch("bantz.core.intent.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=llm_response)
            plan, error = await cot_route("weather in London", [
                {"name": "weather", "description": "Check weather", "risk_level": "safe"},
            ])

        assert plan is not None
        assert error is None
        assert plan["tool_name"] == "weather"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. cot_route — recent_history for pronoun resolution (#212)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCotRouteWithHistory:
    """cot_route injects recent_history into prompt for coreference resolution."""

    @pytest.mark.asyncio
    async def test_recent_history_injected_into_system_prompt(self):
        """When recent_history is provided, it appears in the system message."""
        from bantz.core.intent import cot_route

        captured_messages: list = []

        async def capture_chat(messages):
            captured_messages.extend(messages)
            return json.dumps({
                "route": "tool",
                "tool_name": "gmail",
                "tool_args": {"action": "compose", "to": "john@example.com",
                              "intent": "Send the report"},
                "risk_level": "safe",
                "confidence": 0.9,
            })

        history = [
            {"role": "user", "content": "Who is John?"},
            {"role": "assistant", "content": "John is your colleague at john@example.com."},
            {"role": "user", "content": "Can you send him the report?"},
        ]

        with patch("bantz.core.intent.ollama") as mock_llm:
            mock_llm.chat = capture_chat
            plan, error = await cot_route(
                "Yes, send it to him",
                [{"name": "gmail", "description": "Send email", "risk_level": "safe"}],
                recent_history=history,
            )

        assert len(captured_messages) >= 1
        assert error is None
        system_content = captured_messages[0]["content"]
        assert "RECENT CONVERSATION" in system_content
        assert "john@example.com" in system_content

    @pytest.mark.asyncio
    async def test_no_history_no_injection(self):
        """When recent_history is None, no RECENT CONVERSATION block."""
        from bantz.core.intent import cot_route

        captured_messages: list = []

        async def capture_chat(messages):
            captured_messages.extend(messages)
            return json.dumps({
                "route": "chat",
                "tool_name": None,
                "tool_args": {},
                "risk_level": "safe",
                "confidence": 0.9,
            })

        with patch("bantz.core.intent.ollama") as mock_llm:
            mock_llm.chat = capture_chat
            plan, error = await cot_route("hello", [], recent_history=None)

        system_content = captured_messages[0]["content"]
        assert "RECENT CONVERSATION" not in system_content

    @pytest.mark.asyncio
    async def test_empty_history_no_injection(self):
        """Empty list → no RECENT CONVERSATION block."""
        from bantz.core.intent import cot_route

        captured_messages: list = []

        async def capture_chat(messages):
            captured_messages.extend(messages)
            return json.dumps({
                "route": "chat",
                "tool_name": None,
                "tool_args": {},
                "risk_level": "safe",
                "confidence": 0.9,
            })

        with patch("bantz.core.intent.ollama") as mock_llm:
            mock_llm.chat = capture_chat
            plan, error = await cot_route("hello", [], recent_history=[])

        system_content = captured_messages[0]["content"]
        assert "RECENT CONVERSATION" not in system_content


# ═══════════════════════════════════════════════════════════════════════════════
# 3. _format_recent_history
# ═══════════════════════════════════════════════════════════════════════════════


class TestFormatRecentHistory:
    """_format_recent_history formats conversation turns correctly."""

    def test_formats_turns(self):
        from bantz.core.intent import _format_recent_history

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = _format_recent_history(history)
        assert "user: Hello" in result
        assert "assistant: Hi there!" in result

    def test_truncates_long_content(self):
        from bantz.core.intent import _format_recent_history

        history = [{"role": "user", "content": "x" * 300}]
        result = _format_recent_history(history)
        # Content should be truncated to 200 chars
        assert len(result.split("user: ")[1]) <= 200

    def test_limits_to_six_turns(self):
        from bantz.core.intent import _format_recent_history

        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result = _format_recent_history(history)
        # Should only include last 6 turns
        assert "msg 4" in result
        assert "msg 9" in result
        assert "msg 3" not in result

    def test_empty_returns_empty(self):
        from bantz.core.intent import _format_recent_history
        assert _format_recent_history([]) == ""

    def test_none_returns_empty(self):
        from bantz.core.intent import _format_recent_history
        assert _format_recent_history(None) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Helpers — _is_refusal, _extract_json
# ═══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_is_refusal_positive(self):
        from bantz.core.intent import _is_refusal
        assert _is_refusal("Sorry, I can't assist with that.") is True

    def test_is_refusal_negative(self):
        from bantz.core.intent import _is_refusal
        assert _is_refusal('{"route": "chat"}') is False

    def test_extract_json_basic(self):
        from bantz.core.intent import _extract_json
        result = _extract_json('{"route": "chat"}')
        assert result["route"] == "chat"

    def test_extract_json_with_thinking(self):
        from bantz.core.intent import _extract_json
        text = '<thinking>think</thinking>\n{"route": "tool"}'
        result = _extract_json(text)
        assert result["route"] == "tool"

    def test_extract_json_with_markdown_fences(self):
        from bantz.core.intent import _extract_json
        text = '```json\n{"route": "tool"}\n```'
        result = _extract_json(text)
        assert result["route"] == "tool"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Backward compatibility — old signature still works
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackwardCompat:
    """cot_route without recent_history must still work (keyword-only default)."""

    @pytest.mark.asyncio
    async def test_old_signature_works(self):
        from bantz.core.intent import cot_route

        llm_response = json.dumps({
            "route": "chat",
            "tool_name": None,
            "tool_args": {},
            "risk_level": "safe",
            "confidence": 0.9,
        })

        with patch("bantz.core.intent.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=llm_response)
            # Old call signature — no recent_history
            plan, error = await cot_route("hello", [])

        assert plan is not None
        assert error is None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. People-Pleaser guard — malformed JSON returns error (#253)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPeoplePleaser:
    """cot_route returns error string instead of silent None on JSON failures."""

    @pytest.mark.asyncio
    async def test_malformed_json_both_attempts_returns_error(self):
        """When both LLM attempts return garbage, (None, error_string) is returned."""
        from bantz.core.intent import cot_route

        with patch("bantz.core.intent.ollama") as mock_llm:
            # Both attempts return non-JSON garbage
            mock_llm.chat = AsyncMock(return_value="Sure! I'll send that email right away!")
            plan, error = await cot_route("send email to john", [
                {"name": "gmail", "description": "Send email", "risk_level": "safe"},
            ])

        assert plan is None
        assert error is not None
        assert "failed" in error.lower() or "error" in error.lower()

    @pytest.mark.asyncio
    async def test_malformed_json_first_attempt_recovers(self):
        """When 1st attempt fails but 2nd returns valid JSON, success."""
        from bantz.core.intent import cot_route

        call_count = 0

        async def flaky_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Hmm, let me think about that..."  # garbage
            return json.dumps({
                "route": "tool",
                "tool_name": "gmail",
                "tool_args": {"action": "compose", "to": "john"},
                "risk_level": "safe",
                "confidence": 0.9,
            })

        with patch("bantz.core.intent.ollama") as mock_llm:
            mock_llm.chat = flaky_chat
            plan, error = await cot_route("send email to john", [
                {"name": "gmail", "description": "Send email", "risk_level": "safe"},
            ])

        assert plan is not None
        assert error is None
        assert plan["tool_name"] == "gmail"

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error(self):
        """Non-JSON exceptions (e.g. network) return error string."""
        from bantz.core.intent import cot_route

        with patch("bantz.core.intent.ollama") as mock_llm:
            mock_llm.chat = AsyncMock(side_effect=ConnectionError("Ollama is down"))
            plan, error = await cot_route("check weather", [])

        assert plan is None
        assert error is not None
        assert "Ollama is down" in error
