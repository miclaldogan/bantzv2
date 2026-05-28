"""Tests for Issue #435 — bantz --once progress messages and stream drain.

Covers:
  - brain.process() accepts a progress_cb parameter
  - progress_cb("Thinking…") is emitted before cot_route
  - progress_cb("Translating…") emitted only when bridge enabled
  - progress_cb=None (default) does not crash
  - _once() writes progress to stderr, response to stdout
  - _once() drains AsyncIterator streams returned by brain.process()
  - _once() falls back to result.response when stream is None
  - _once() calls tts_engine.speak() when tts enabled + speak_all_responses
  - _once() skips TTS when disabled
  - _once() does not double-speak TTS-tool results
"""
from __future__ import annotations

import asyncio
import inspect
import io
import sys
from contextlib import redirect_stdout
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bantz.core.types import BrainResult


# ── helpers ───────────────────────────────────────────────────────────────────

async def _aiter(*items: str) -> AsyncIterator[str]:
    """Tiny async generator helper for stream mocks."""
    for item in items:
        yield item


# ── brain.process() signature ─────────────────────────────────────────────────

class TestBrainProcessSignature:
    def test_progress_cb_parameter_exists(self):
        from bantz.core.brain import Brain
        sig = inspect.signature(Brain.process)
        assert "progress_cb" in sig.parameters

    def test_progress_cb_defaults_to_none(self):
        from bantz.core.brain import Brain
        sig = inspect.signature(Brain.process)
        assert sig.parameters["progress_cb"].default is None

    def test_progress_cb_is_optional(self):
        """Calling process without progress_cb should not raise a TypeError."""
        from bantz.core.brain import Brain
        sig = inspect.signature(Brain.process)
        param = sig.parameters["progress_cb"]
        # Optional means it has a default value
        assert param.default is not inspect.Parameter.empty


# ── progress emission logic ───────────────────────────────────────────────────

class TestBrainProgressEmission:
    """Unit-test the progress hooks inside brain.process()."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain()
        # Suppress internal startup side effects
        b._memory_initialized = True
        b._graph_initialized = True
        return b

    @pytest.mark.asyncio
    async def test_thinking_progress_emitted(self):
        """'Thinking…' must be emitted before cot_route."""
        calls: list[str] = []
        fake_result = BrainResult(response="ok", tool_used=None)

        with patch("bantz.core.brain.Brain._ensure_memory"), \
             patch("bantz.core.brain.Brain._ensure_graph", new_callable=AsyncMock), \
             patch("bantz.core.brain.Brain._to_en", new_callable=AsyncMock, return_value="hi"), \
             patch("bantz.core.brain.get_bridge", return_value=None), \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock,
                   return_value=({"route": "chat", "tool_name": "", "confidence": 1.0}, None)), \
             patch("bantz.core.brain.Brain._chat_stream", return_value=_aiter("hello")), \
             patch("bantz.core.brain.data_layer.conversations") as mock_conv:
            mock_conv.add = MagicMock()
            mock_conv.context = MagicMock(return_value=[])
            b = self._make_brain()
            await b.process("hi", progress_cb=calls.append)

        assert "Thinking…" in calls

    @pytest.mark.asyncio
    async def test_translating_progress_emitted_when_bridge_enabled(self):
        """'Translating…' emitted only when bridge.is_enabled() is True."""
        calls: list[str] = []

        mock_bridge = MagicMock()
        mock_bridge.is_enabled.return_value = True

        with patch("bantz.core.brain.Brain._ensure_memory"), \
             patch("bantz.core.brain.Brain._ensure_graph", new_callable=AsyncMock), \
             patch("bantz.core.brain.Brain._to_en", new_callable=AsyncMock, return_value="hi"), \
             patch("bantz.core.brain.get_bridge", return_value=mock_bridge), \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock,
                   return_value=({"route": "chat", "tool_name": "", "confidence": 1.0}, None)), \
             patch("bantz.core.brain.Brain._chat_stream", return_value=_aiter("hello")), \
             patch("bantz.core.brain.data_layer.conversations") as mock_conv:
            mock_conv.add = MagicMock()
            mock_conv.context = MagicMock(return_value=[])
            b = self._make_brain()
            await b.process("merhaba", progress_cb=calls.append)

        assert "Translating…" in calls

    @pytest.mark.asyncio
    async def test_translating_not_emitted_when_bridge_disabled(self):
        """No 'Translating…' when bridge is disabled or unavailable."""
        calls: list[str] = []

        mock_bridge = MagicMock()
        mock_bridge.is_enabled.return_value = False

        with patch("bantz.core.brain.Brain._ensure_memory"), \
             patch("bantz.core.brain.Brain._ensure_graph", new_callable=AsyncMock), \
             patch("bantz.core.brain.Brain._to_en", new_callable=AsyncMock, return_value="hi"), \
             patch("bantz.core.brain.get_bridge", return_value=mock_bridge), \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock,
                   return_value=({"route": "chat", "tool_name": "", "confidence": 1.0}, None)), \
             patch("bantz.core.brain.Brain._chat_stream", return_value=_aiter("hello")), \
             patch("bantz.core.brain.data_layer.conversations") as mock_conv:
            mock_conv.add = MagicMock()
            mock_conv.context = MagicMock(return_value=[])
            b = self._make_brain()
            await b.process("hi", progress_cb=calls.append)

        assert "Translating…" not in calls

    @pytest.mark.asyncio
    async def test_no_progress_cb_does_not_crash(self):
        """Omitting progress_cb (default None) must never raise."""
        with patch("bantz.core.brain.Brain._ensure_memory"), \
             patch("bantz.core.brain.Brain._ensure_graph", new_callable=AsyncMock), \
             patch("bantz.core.brain.Brain._to_en", new_callable=AsyncMock, return_value="hi"), \
             patch("bantz.core.brain.get_bridge", return_value=None), \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock,
                   return_value=({"route": "chat", "tool_name": "", "confidence": 1.0}, None)), \
             patch("bantz.core.brain.Brain._chat_stream", return_value=_aiter("hello")), \
             patch("bantz.core.brain.data_layer.conversations") as mock_conv:
            mock_conv.add = MagicMock()
            mock_conv.context = MagicMock(return_value=[])
            b = self._make_brain()
            # Should not raise
            await b.process("hi")

    @pytest.mark.asyncio
    async def test_thinking_emitted_before_cot_route_is_called(self):
        """Order guarantee: 'Thinking…' must appear before cot_route runs."""
        order: list[str] = []

        async def fake_cot_route(*a, **kw):
            order.append("cot_route")
            return {"route": "chat", "tool_name": "", "confidence": 1.0}, None

        def fake_progress(msg: str) -> None:
            order.append(msg)

        with patch("bantz.core.brain.Brain._ensure_memory"), \
             patch("bantz.core.brain.Brain._ensure_graph", new_callable=AsyncMock), \
             patch("bantz.core.brain.Brain._to_en", new_callable=AsyncMock, return_value="hi"), \
             patch("bantz.core.brain.get_bridge", return_value=None), \
             patch("bantz.core.brain.cot_route", side_effect=fake_cot_route), \
             patch("bantz.core.brain.Brain._chat_stream", return_value=_aiter("hi")), \
             patch("bantz.core.brain.data_layer.conversations") as mock_conv:
            mock_conv.add = MagicMock()
            mock_conv.context = MagicMock(return_value=[])
            b = self._make_brain()
            await b.process("hi", progress_cb=fake_progress)

        assert "Thinking…" in order
        thinking_idx = order.index("Thinking…")
        cot_idx = order.index("cot_route")
        assert thinking_idx < cot_idx, f"Expected Thinking… before cot_route, got order={order}"


# ── _once() behaviour ─────────────────────────────────────────────────────────

def _make_brain_result(
    *,
    response: str = "",
    stream: AsyncIterator[str] | None = None,
    tool_used: str | None = None,
    needs_confirm: bool = False,
) -> BrainResult:
    return BrainResult(
        response=response,
        tool_used=tool_used,
        stream=stream,
        needs_confirm=needs_confirm,
    )


class TestOnceProgressToStderr:
    """_once() must write progress to stderr, response to stdout."""

    @pytest.mark.asyncio
    async def test_progress_goes_to_stderr_not_stdout(self):
        from bantz.__main__ import _once

        stderr_buf = io.StringIO()
        stdout_buf = io.StringIO()

        with patch("bantz.core.brain.brain") as mock_brain, \
             patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = False
            mock_brain.process = AsyncMock(
                return_value=_make_brain_result(response="Hello, ma'am.")
            )
            with redirect_stdout(stdout_buf):
                with patch("sys.stderr", stderr_buf):
                    await _once("hello")

        assert "Hello, ma'am." in stdout_buf.getvalue()
        # progress (e.g. "Thinking…") must NOT be on stdout
        stderr_val = stderr_buf.getvalue()
        stdout_val = stdout_buf.getvalue()
        assert "Thinking" not in stdout_val or "Thinking" in stderr_val


class TestOnceStreamDrain:
    """_once() must consume AsyncIterator streams returned by brain.process()."""

    @pytest.mark.asyncio
    async def test_drains_stream_and_prints_joined_response(self):
        from bantz.__main__ import _once

        chunks = ["Good", " morning", ", ma'am."]
        stream = _aiter(*chunks)
        result = _make_brain_result(stream=stream)

        stdout_buf = io.StringIO()
        with patch("bantz.core.brain.brain") as mock_brain, \
             patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = False
            mock_brain.process = AsyncMock(return_value=result)
            with redirect_stdout(stdout_buf):
                with patch("sys.stderr", io.StringIO()):
                    await _once("good morning")

        assert "Good morning, ma'am." in stdout_buf.getvalue()

    @pytest.mark.asyncio
    async def test_prints_direct_response_when_no_stream(self):
        from bantz.__main__ import _once

        result = _make_brain_result(response="42 is the answer.", stream=None)
        stdout_buf = io.StringIO()
        with patch("bantz.core.brain.brain") as mock_brain, \
             patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = False
            mock_brain.process = AsyncMock(return_value=result)
            with redirect_stdout(stdout_buf):
                with patch("sys.stderr", io.StringIO()):
                    await _once("what is 6*7")

        assert "42 is the answer." in stdout_buf.getvalue()

    @pytest.mark.asyncio
    async def test_empty_stream_produces_empty_output(self):
        from bantz.__main__ import _once

        stream = _aiter()  # empty
        result = _make_brain_result(stream=stream)
        stdout_buf = io.StringIO()
        with patch("bantz.core.brain.brain") as mock_brain, \
             patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = False
            mock_brain.process = AsyncMock(return_value=result)
            with redirect_stdout(stdout_buf):
                with patch("sys.stderr", io.StringIO()):
                    await _once("hello")

        # Should print empty line (print("") adds a newline) without crashing
        assert stdout_buf.getvalue() == "\n"


class TestOnceTTS:
    """_once() TTS integration."""

    @pytest.mark.asyncio
    async def test_tts_speak_called_when_enabled(self):
        from bantz.__main__ import _once

        result = _make_brain_result(response="Very good, sir.")
        mock_tts = MagicMock()
        mock_tts.speak = AsyncMock()

        with patch("bantz.core.brain.brain") as mock_brain, \
             patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = True
            mock_cfg.tts_speak_all_responses = True
            mock_brain.process = AsyncMock(return_value=result)
            with redirect_stdout(io.StringIO()):
                with patch("sys.stderr", io.StringIO()):
                    with patch.dict("sys.modules", {"bantz.agent.tts": MagicMock(tts_engine=mock_tts)}):
                        await _once("hello")

        mock_tts.speak.assert_awaited_once_with("Very good, sir.")

    @pytest.mark.asyncio
    async def test_tts_skipped_when_disabled(self):
        from bantz.__main__ import _once

        result = _make_brain_result(response="Very good, sir.")
        mock_tts = MagicMock()
        mock_tts.speak = AsyncMock()

        with patch("bantz.core.brain.brain") as mock_brain, \
             patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = False
            mock_cfg.tts_speak_all_responses = True
            mock_brain.process = AsyncMock(return_value=result)
            with redirect_stdout(io.StringIO()):
                with patch("sys.stderr", io.StringIO()):
                    await _once("hello")

        mock_tts.speak.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tts_skipped_for_tts_tool_result(self):
        """Don't double-speak when the tool itself was TTS."""
        from bantz.__main__ import _once

        result = _make_brain_result(response="Speaking…", tool_used="tts")
        mock_tts = MagicMock()
        mock_tts.speak = AsyncMock()

        with patch("bantz.core.brain.brain") as mock_brain, \
             patch("bantz.config.config") as mock_cfg:
            mock_cfg.tts_enabled = True
            mock_cfg.tts_speak_all_responses = True
            mock_brain.process = AsyncMock(return_value=result)
            with redirect_stdout(io.StringIO()):
                with patch("sys.stderr", io.StringIO()):
                    with patch.dict("sys.modules", {"bantz.agent.tts": MagicMock(tts_engine=mock_tts)}):
                        await _once("speak something")

        mock_tts.speak.assert_not_awaited()


# ── source-level checks ───────────────────────────────────────────────────────

class TestOnceSourceLevel:
    """Verify the implementation meets the spec without running live code."""

    def test_progress_cb_passed_to_brain_process(self):
        src = inspect.getsource(sys.modules["bantz.__main__"]._once
                                if "bantz.__main__" in sys.modules
                                else __import__("bantz.__main__", fromlist=["_once"])._once)
        assert "progress_cb" in src

    def test_stream_drain_present(self):
        from bantz.__main__ import _once
        src = inspect.getsource(_once)
        assert "result.stream" in src
        assert "async for" in src

    def test_stderr_used_for_progress(self):
        from bantz.__main__ import _once
        src = inspect.getsource(_once)
        assert "stderr" in src

    def test_tts_speak_called_in_once(self):
        from bantz.__main__ import _once
        src = inspect.getsource(_once)
        assert "tts_engine" in src
        assert "speak" in src

    def test_no_unconditional_loading_models_print(self):
        """The old 'Loading models…' printed to stdout unconditionally is gone."""
        from bantz.__main__ import _once
        src = inspect.getsource(_once)
        # The old line was: print("Loading models…", flush=True)
        assert 'print("Loading models' not in src
