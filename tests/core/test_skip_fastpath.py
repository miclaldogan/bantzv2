"""cot_route skip_fastpath (audit C1 prerequisite, issue #502).

The pre-route regexes match on the *input text*, which does not change
after a tool failure — so the C1 recovery loop's re-decide calls must
bypass the whole family or they re-select the same failed tool forever.
These tests pin both directions: default False = fast-paths fire exactly
as before; True = every pre-route (including ``investigate:``) is skipped
and the LLM is consulted.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from bantz.core.intent import _fastpath_route, cot_route

# A canned LLM verdict, so tests can detect "the LLM path was reached"
# without a live model.
_LLM_VERDICT = json.dumps({
    "route": "chat",
    "tool_name": None,
    "tool_args": {},
    "risk_level": "safe",
    "confidence": 0.9,
})


async def _aiter_tokens(text: str):
    for char in text:
        yield char


def _mock_stream(text: str):
    async def _stream(messages, **kwargs):
        async for tok in _aiter_tokens(text):
            yield tok
    return _stream


# Inputs that MUST hit a fast-path by default, with the route/tool the
# pre-route family is contractually expected to produce.
FASTPATH_CASES = [
    ("investigate: high swap usage — 92%", "chat", None),
    ("remind me in 10 minutes to stretch", "tool", "reminder"),
    ("do you remember my sister's name?", "chat", None),
    ("i want to listen to AC/DC", "tool", "vision_execute"),
    ("let's watch lofi videos on youtube", "tool", "vision_execute"),
    ("workspace 3", "tool", "desktop"),
    ("next workspace", "tool", "desktop"),
    ("click the send button", "tool", "visual_click"),
]


class TestFastpathExtraction:
    """_fastpath_route preserves each family's pre-#502 verdict."""

    @pytest.mark.parametrize("text,route,tool", FASTPATH_CASES)
    def test_families_still_match(self, text, route, tool):
        result = _fastpath_route(text)
        assert result is not None, f"expected a fast-path match for {text!r}"
        plan, err = result
        assert err is None
        assert plan["route"] == route
        assert plan.get("tool_name") == tool

    def test_no_match_falls_through(self):
        assert _fastpath_route("what's the weather in Elazig?") is None


class TestSkipFastpath:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("text,route,tool", FASTPATH_CASES)
    async def test_default_uses_fastpath_without_llm(self, text, route, tool):
        """skip_fastpath omitted → fast-path verdict, zero LLM calls."""
        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = AsyncMock(
                side_effect=AssertionError("LLM must not be called on a fast-path"),
            )
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route(text, [])
        assert error is None
        assert plan["route"] == route
        assert plan.get("tool_name") == tool

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text,_route,_tool", FASTPATH_CASES)
    async def test_skip_reaches_llm_for_every_family(self, text, _route, _tool):
        """skip_fastpath=True → every pre-route family is bypassed and the
        LLM is consulted — including the investigate: pre-route, which does
        not look like a pattern fast-path but is one (#502 gotcha)."""
        with patch("bantz.core.intent.ollama") as mock_llm, \
             patch("bantz.core.intent.bus") as mock_bus:
            mock_llm.chat_stream = _mock_stream(_LLM_VERDICT)
            mock_bus.emit = AsyncMock()
            plan, error = await cot_route(text, [], skip_fastpath=True)
        # The canned verdict (not the fast-path plan) must come back:
        # fast-path plans carry a "pre-route:" reasoning marker.
        assert plan is not None
        assert "pre-route" not in (plan.get("reasoning") or "")
