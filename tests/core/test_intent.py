"""Tests for ``bantz.core.intent`` — CoT intent parser (#78, #212, #273).

Covers:
  1. cot_route — basic routing (mocked Ollama streaming)
  2. cot_route — recent_history injection for pronoun resolution (#212)
  3. _format_recent_history — formatting helper
  4. _is_refusal / _extract_json / _log_thinking — helpers
  5. _stream_and_collect — streaming + thinking event emission (#273)
  6. _clean_thinking_text — tag stripping (#273)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest


# ── Helper: make an async iterator from a string ────────────────────────────

async def _aiter_tokens(text: str):
    """Yield a string token-by-token (simulates ollama.chat_stream)."""
    for char in text:
        yield char


def _mock_stream(text: str):
    """Return a callable that produces an async iterator of tokens."""
    async def _stream(messages):
        async for tok in _aiter_tokens(text):
            yield tok
    return _stream


# ═══════════════════════════════════════════════════════════════════════════════
# 1. cot_route — basic routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestCotRouteBasic:
    """cot_route returns routing plans from mocked LLM streaming responses."""

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

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(llm_response)
            mock_bus.emit = AsyncMock()
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

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(llm_response)
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route("Hello there!", [])

        assert plan is not None
        assert error is None
        assert plan["route"] == "chat"

    @pytest.mark.asyncio
    async def test_returns_none_on_refusal(self):
        from bantz.core.intent import cot_route

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream("Sorry, I can't assist with that.")
            mock_bus.emit = AsyncMock()
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

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(llm_response)
            mock_bus.emit = AsyncMock()
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

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(llm_response)
            mock_bus.emit = AsyncMock()
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

        async def capture_stream(messages):
            captured_messages.extend(messages)
            response = json.dumps({
                "route": "tool",
                "tool_name": "gmail",
                "tool_args": {"action": "compose", "to": "john@example.com",
                              "intent": "Send the report"},
                "risk_level": "safe",
                "confidence": 0.9,
            })
            for ch in response:
                yield ch

        history = [
            {"role": "user", "content": "Who is John?"},
            {"role": "assistant", "content": "John is your colleague at john@example.com."},
            {"role": "user", "content": "Can you send him the report?"},
        ]

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = capture_stream
            mock_bus.emit = AsyncMock()
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

        async def capture_stream(messages):
            captured_messages.extend(messages)
            response = json.dumps({
                "route": "chat",
                "tool_name": None,
                "tool_args": {},
                "risk_level": "safe",
                "confidence": 0.9,
            })
            for ch in response:
                yield ch

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = capture_stream
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route("hello", [], recent_history=None)

        system_content = captured_messages[0]["content"]
        # The static COT_SYSTEM mentions "RECENT CONVERSATION" in routing rules,
        # so we check for the specific dynamic block header instead.
        assert "RECENT CONVERSATION (use to resolve pronouns" not in system_content

    @pytest.mark.asyncio
    async def test_empty_history_no_injection(self):
        """Empty list → no RECENT CONVERSATION block."""
        from bantz.core.intent import cot_route

        captured_messages: list = []

        async def capture_stream(messages):
            captured_messages.extend(messages)
            response = json.dumps({
                "route": "chat",
                "tool_name": None,
                "tool_args": {},
                "risk_level": "safe",
                "confidence": 0.9,
            })
            for ch in response:
                yield ch

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = capture_stream
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route("hello", [], recent_history=[])

        system_content = captured_messages[0]["content"]
        assert "RECENT CONVERSATION (use to resolve pronouns" not in system_content

    @pytest.mark.asyncio
    async def test_tool_context_injected_when_provided(self):
        """tool_context string is appended to system prompt (#275)."""
        from bantz.core.intent import cot_route

        captured_messages: list = []

        async def capture_stream(messages):
            captured_messages.extend(messages)
            response = json.dumps({
                "route": "tool",
                "tool_name": "gmail",
                "tool_args": {"action": "read", "message_id": "abc123"},
                "risk_level": "safe",
                "confidence": 0.95,
            })
            for ch in response:
                yield ch

        tool_ctx = 'RECENT EMAIL RESULTS:\n  - ID: abc123 | From: test@x.com | Subject: "Test"'

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = capture_stream
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route(
                "read that email", [],
                tool_context=tool_ctx,
            )

        system_content = captured_messages[0]["content"]
        assert "abc123" in system_content
        assert "test@x.com" in system_content

    @pytest.mark.asyncio
    async def test_tool_context_empty_not_injected(self):
        """Empty tool_context does not add extra content (#275)."""
        from bantz.core.intent import cot_route

        captured_messages: list = []

        async def capture_stream(messages):
            captured_messages.extend(messages)
            response = json.dumps({
                "route": "chat", "tool_name": None, "tool_args": {},
                "risk_level": "safe", "confidence": 0.9,
            })
            for ch in response:
                yield ch

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = capture_stream
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route("hello", [], tool_context="")

        system_content = captured_messages[0]["content"]
        assert "RECENT EMAIL" not in system_content


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

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(llm_response)
            mock_bus.emit = AsyncMock()
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

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            # Both attempts return non-JSON garbage
            mock_llm.chat_stream = _mock_stream("Sure! I'll send that email right away!")
            mock_bus.emit = AsyncMock()
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

        async def flaky_stream(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                text = "Hmm, let me think about that..."  # garbage
            else:
                text = json.dumps({
                    "route": "tool",
                    "tool_name": "gmail",
                    "tool_args": {"action": "compose", "to": "john"},
                    "risk_level": "safe",
                    "confidence": 0.9,
                })
            for ch in text:
                yield ch

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = flaky_stream
            mock_bus.emit = AsyncMock()
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

        async def error_stream(messages):
            raise ConnectionError("Ollama is down")
            yield  # make it an async generator  # noqa: unreachable

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = error_stream
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route("check weather", [])

        assert plan is None
        assert error is not None
        assert "Ollama is down" in error


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Planner route support (#272)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlannerRoute:
    """cot_route can return route='planner' for multi-step requests (#272)."""

    @pytest.mark.asyncio
    async def test_planner_route_returned(self):
        from bantz.core.intent import cot_route

        llm_response = json.dumps({
            "route": "planner",
            "tool_name": None,
            "tool_args": {},
            "risk_level": "safe",
            "confidence": 0.9,
            "reasoning": "User wants weather then email — requires two tools.",
        })

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(llm_response)
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route(
                "Check the weather in Istanbul then email it to John", [
                    {"name": "weather", "description": "Check weather", "risk_level": "safe"},
                    {"name": "gmail", "description": "Send email", "risk_level": "safe"},
                ],
            )

        assert plan is not None
        assert error is None
        assert plan["route"] == "planner"
        assert plan["reasoning"] == "User wants weather then email — requires two tools."

    @pytest.mark.asyncio
    async def test_planner_route_with_reasoning_field(self):
        """Reasoning field is preserved in parsed output."""
        from bantz.core.intent import cot_route

        llm_response = json.dumps({
            "route": "tool",
            "tool_name": "weather",
            "tool_args": {"city": "London"},
            "risk_level": "safe",
            "confidence": 0.95,
            "reasoning": "Single tool needed — just weather lookup.",
        })

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(llm_response)
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route("weather in London", [
                {"name": "weather", "description": "Check weather", "risk_level": "safe"},
            ])

        assert plan is not None
        assert "reasoning" in plan


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Streaming thinking events (#273)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamThinking:
    """_stream_and_collect emits thinking_token/thinking_done events (#273)."""

    @pytest.mark.asyncio
    async def test_thinking_events_emitted(self):
        """Tokens inside <thinking> are emitted as thinking_token events."""
        from bantz.core.intent import _stream_and_collect

        response = "<thinking>I should use the weather tool.</thinking>\n" + json.dumps({"route": "tool"})

        emitted: list = []

        async def capture_emit(name, **kwargs):
            emitted.append((name, kwargs))

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(response)
            mock_bus.emit = capture_emit

            result = await _stream_and_collect(
                [{"role": "user", "content": "test"}],
                emit_thinking=True, source="test",
            )

        event_names = [e[0] for e in emitted]
        assert "thinking_start" in event_names
        assert "thinking_done" in event_names
        # At least one thinking_token should have been emitted
        token_events = [e for e in emitted if e[0] == "thinking_token"]
        assert len(token_events) > 0

    @pytest.mark.asyncio
    async def test_no_thinking_events_when_disabled(self):
        """With emit_thinking=False, no events are emitted."""
        from bantz.core.intent import _stream_and_collect

        response = "<thinking>secret</thinking>\n{}"

        emitted: list = []

        async def capture_emit(name, **kwargs):
            emitted.append((name, kwargs))

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(response)
            mock_bus.emit = capture_emit

            await _stream_and_collect(
                [{"role": "user", "content": "test"}],
                emit_thinking=False, source="test",
            )

        assert len(emitted) == 0

    @pytest.mark.asyncio
    async def test_full_response_returned(self):
        """_stream_and_collect returns the raw full response string."""
        from bantz.core.intent import _stream_and_collect

        response = "<thinking>think</thinking>\n{\"route\": \"chat\"}"

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(response)
            mock_bus.emit = AsyncMock()

            result = await _stream_and_collect(
                [{"role": "user", "content": "test"}],
                emit_thinking=True, source="test",
            )

        assert "<thinking>" in result
        assert '{"route": "chat"}' in result


class TestCleanThinkingText:
    """_clean_thinking_text strips literal XML tags (#273 Critique 2)."""

    def test_strips_open_tag(self):
        from bantz.core.intent import _clean_thinking_text
        assert _clean_thinking_text("<thinking>hello") == "hello"

    def test_strips_close_tag(self):
        from bantz.core.intent import _clean_thinking_text
        assert _clean_thinking_text("world</thinking>") == "world"

    def test_strips_both_tags(self):
        from bantz.core.intent import _clean_thinking_text
        assert _clean_thinking_text("<thinking>content</thinking>") == "content"

    def test_empty_string(self):
        from bantz.core.intent import _clean_thinking_text
        assert _clean_thinking_text("") == ""

    def test_no_tags(self):
        from bantz.core.intent import _clean_thinking_text
        assert _clean_thinking_text("just text") == "just text"


class TestStripThinkingUnclosed:
    """strip_thinking handles unclosed tags (#273 Critique 2)."""

    def test_unclosed_thinking_stripped(self):
        from bantz.core.intent import strip_thinking
        result = strip_thinking("<thinking>This never closes")
        assert result.strip() == ""

    def test_normal_thinking_stripped(self):
        from bantz.core.intent import strip_thinking
        result = strip_thinking("<thinking>inner</thinking> after")
        assert "inner" not in result
        assert "after" in result

    def test_double_open_stripped(self):
        from bantz.core.intent import strip_thinking
        result = strip_thinking("<thinking>first<thinking>second</thinking> after")
        assert "after" in result


# ═══════════════════════════════════════════════════════════════════════════════
# _is_refusal — thinking-aware refusal detection (#282)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsRefusalThinkingStrip:
    """_is_refusal must strip <thinking> blocks before checking patterns (#282)."""

    def test_sorry_in_thinking_not_refusal(self):
        """'sorry' inside thinking reasoning must NOT trigger refusal."""
        from bantz.core.intent import _is_refusal
        raw = (
            '<thinking>I\'m sorry, I need to re-read the request carefully. '
            'The user wants to read emails.</thinking>'
            '{"route":"tool","tool_name":"gmail","tool_args":{"action":"search","query":"erasmus"}}'
        )
        assert _is_refusal(raw) is False

    def test_bare_sorry_not_refusal(self):
        from bantz.core.intent import _is_refusal
        assert _is_refusal("sorry") is False
        assert _is_refusal("I'm sorry, let me try again.") is False

    def test_real_refusal_detected(self):
        from bantz.core.intent import _is_refusal
        assert _is_refusal("Sorry, I can't assist with that request.") is True
        assert _is_refusal("I cannot provide that information.") is True
        assert _is_refusal("That would be inappropriate.") is True

    def test_refusal_outside_thinking_detected(self):
        """Refusal in the JSON/response area (after thinking) must still trigger."""
        from bantz.core.intent import _is_refusal
        raw = (
            '<thinking>The user asked something harmful.</thinking>'
            'Sorry, I cannot provide that information.'
        )
        assert _is_refusal(raw) is True

    def test_cannot_alone_not_refusal(self):
        """Bare 'i cannot' should NOT trigger — too broad.
        Only specific refusal phrases like 'i cannot provide' or 'i cannot help'."""
        from bantz.core.intent import _is_refusal
        assert _is_refusal("I cannot believe it's raining!") is False


# ═══════════════════════════════════════════════════════════════════════════════
# cot_route — sloppy route normalisation (#282)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCotRouteNormalisation:
    """cot_route should handle models that put tool names in the route field."""

    @pytest.mark.asyncio
    async def test_route_is_tool_name_normalised(self):
        """If model returns route='gmail' instead of 'tool', it should be normalised."""
        from bantz.core.intent import cot_route

        llm_response = json.dumps({
            "route": "gmail",
            "tool_name": "email",
            "tool_args": {"action": "search", "query": "erasmus"},
            "risk_level": "safe",
            "confidence": 0.95,
        })

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(llm_response)
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route("read my emails about erasmus", [
                {"name": "gmail", "description": "Email", "risk_level": "safe"},
            ])

        assert plan is not None
        # The route is "gmail" — brain.py normalises this; cot_route just returns the plan
        assert plan["route"] == "gmail"  # cot_route doesn't normalise, brain.py does
        assert plan["tool_name"] == "email"

    @pytest.mark.asyncio
    async def test_thinking_with_sorry_still_routes(self):
        """Model that says 'sorry' in thinking should still return a valid plan."""
        from bantz.core.intent import cot_route

        llm_response = (
            '<thinking>I\'m sorry, I initially thought this was chat. '
            'But the user clearly wants to check email.</thinking>'
            + json.dumps({
                "route": "tool",
                "tool_name": "gmail",
                "tool_args": {"action": "search", "query": "erasmus"},
                "risk_level": "safe",
                "confidence": 0.92,
            })
        )

        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(llm_response)
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route("read my last emails about erasmus", [
                {"name": "gmail", "description": "Email", "risk_level": "safe"},
            ])

        assert error is None
        assert plan is not None
        assert plan["tool_name"] == "gmail"
