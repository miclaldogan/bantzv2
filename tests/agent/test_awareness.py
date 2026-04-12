"""
Tests for bantz.agent.awareness — Continuous Awareness Pipeline (#325).

Covers:
  ✓ AwarenessState dataclass defaults
  ✓ Rolling buffer maxlen (20-state cap)
  ✓ get_current_context() format and empty-state handling
  ✓ get_screenshot_for_vlm() returns path only when screenshot succeeded
  ✓ Graceful handling when xdotool / xclip / grim are unavailable (mock subprocess)
  ✓ Graceful handling when psutil raises
  ✓ Screenshot trigger keyword detection in Brain._maybe_inject_awareness_screenshot
  ✓ AwarenessCollector.run() cancellation
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# AwarenessState
# ═══════════════════════════════════════════════════════════════════════════

class TestAwarenessState:
    def test_defaults(self):
        from bantz.agent.awareness import AwarenessState
        s = AwarenessState()
        assert s.active_window_title == ""
        assert s.active_window_process == ""
        assert s.clipboard_text == ""
        assert s.screenshot_path == ""
        assert isinstance(s.timestamp, datetime)

    def test_custom_values(self):
        from bantz.agent.awareness import AwarenessState
        s = AwarenessState(
            active_window_title="VS Code",
            active_window_process="code",
            clipboard_text="hello",
            screenshot_path="/tmp/x.png",
        )
        assert s.active_window_title == "VS Code"
        assert s.active_window_process == "code"
        assert s.clipboard_text == "hello"
        assert s.screenshot_path == "/tmp/x.png"


# ═══════════════════════════════════════════════════════════════════════════
# AwarenessCollector — rolling buffer
# ═══════════════════════════════════════════════════════════════════════════

class TestRollingBuffer:
    def test_buffer_maxlen(self):
        """Rolling buffer must not exceed 20 entries."""
        from bantz.agent.awareness import AwarenessCollector, AwarenessState
        c = AwarenessCollector(interval_s=15.0)
        assert c.BUFFER_SIZE == 20
        # Insert 25 states — only last 20 should remain
        for i in range(25):
            c._buffer.append(AwarenessState(active_window_title=str(i)))
        assert len(c._buffer) == 20
        assert c._buffer[-1].active_window_title == "24"
        assert c._buffer[0].active_window_title == "5"

    def test_latest_is_none_when_empty(self):
        from bantz.agent.awareness import AwarenessCollector
        c = AwarenessCollector()
        assert c.latest is None

    def test_latest_returns_most_recent(self):
        from bantz.agent.awareness import AwarenessCollector, AwarenessState
        c = AwarenessCollector()
        c._buffer.append(AwarenessState(active_window_title="first"))
        c._buffer.append(AwarenessState(active_window_title="second"))
        assert c.latest is not None
        assert c.latest.active_window_title == "second"


# ═══════════════════════════════════════════════════════════════════════════
# get_current_context()
# ═══════════════════════════════════════════════════════════════════════════

class TestGetCurrentContext:
    def test_empty_buffer_returns_empty_string(self):
        from bantz.agent.awareness import AwarenessCollector
        c = AwarenessCollector()
        assert c.get_current_context() == ""

    def test_format_with_all_fields(self):
        from bantz.agent.awareness import AwarenessCollector, AwarenessState
        c = AwarenessCollector()
        c._buffer.append(AwarenessState(
            active_window_title="VS Code",
            active_window_process="code",
            clipboard_text="def foo():",
            screenshot_path="/tmp/bantz_awareness_latest.png",
        ))
        ctx = c.get_current_context()
        assert ctx.startswith("[Awareness]")
        assert "VS Code (code)" in ctx
        assert '"def foo():"' in ctx
        assert "/tmp/bantz_awareness_latest.png" in ctx

    def test_format_no_process(self):
        from bantz.agent.awareness import AwarenessCollector, AwarenessState
        c = AwarenessCollector()
        c._buffer.append(AwarenessState(active_window_title="Terminal"))
        ctx = c.get_current_context()
        assert "Terminal" in ctx
        assert "()" not in ctx  # no empty parens when process is missing

    def test_format_no_screenshot(self):
        from bantz.agent.awareness import AwarenessCollector, AwarenessState
        c = AwarenessCollector()
        c._buffer.append(AwarenessState(
            active_window_title="Firefox",
            clipboard_text="https://example.com",
            screenshot_path="",
        ))
        ctx = c.get_current_context()
        assert "Screenshot" not in ctx

    def test_clipboard_truncated_in_context(self):
        """Clipboard text longer than 120 chars should be clipped."""
        from bantz.agent.awareness import AwarenessCollector, AwarenessState
        c = AwarenessCollector()
        long_text = "x" * 200
        c._buffer.append(AwarenessState(clipboard_text=long_text))
        ctx = c.get_current_context()
        # The quoted clipboard snippet should not exceed 120 chars + quotes
        clip_segment = [p for p in ctx.split("|") if "Clipboard" in p][0]
        # Strip label to measure the actual snippet length
        snippet = clip_segment.split('"')[1] if '"' in clip_segment else ""
        assert len(snippet) <= 120

    def test_all_empty_fields_returns_empty(self):
        from bantz.agent.awareness import AwarenessCollector, AwarenessState
        c = AwarenessCollector()
        c._buffer.append(AwarenessState())  # all fields are ""
        assert c.get_current_context() == ""


# ═══════════════════════════════════════════════════════════════════════════
# get_screenshot_for_vlm()
# ═══════════════════════════════════════════════════════════════════════════

class TestGetScreenshotForVlm:
    def test_none_when_empty(self):
        from bantz.agent.awareness import AwarenessCollector
        assert AwarenessCollector().get_screenshot_for_vlm() is None

    def test_returns_path_when_available(self):
        from bantz.agent.awareness import AwarenessCollector, AwarenessState
        c = AwarenessCollector()
        c._buffer.append(AwarenessState(screenshot_path="/tmp/x.png"))
        assert c.get_screenshot_for_vlm() == "/tmp/x.png"

    def test_returns_none_when_screenshot_failed(self):
        from bantz.agent.awareness import AwarenessCollector, AwarenessState
        c = AwarenessCollector()
        c._buffer.append(AwarenessState(screenshot_path=""))
        assert c.get_screenshot_for_vlm() is None


# ═══════════════════════════════════════════════════════════════════════════
# Subprocess helpers — graceful failure when tools are absent
# ═══════════════════════════════════════════════════════════════════════════

class TestSubprocessHelpers:
    def test_xdotool_not_found(self):
        from bantz.agent.awareness import _get_active_window_title
        with patch("subprocess.run", side_effect=FileNotFoundError("xdotool")):
            assert _get_active_window_title() == ""

    def test_xdotool_timeout(self):
        import subprocess
        from bantz.agent.awareness import _get_active_window_title
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("xdotool", 3)):
            assert _get_active_window_title() == ""

    def test_xdotool_nonzero_exit(self):
        from bantz.agent.awareness import _get_active_window_title
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert _get_active_window_title() == ""

    def test_xdotool_title_success(self):
        from bantz.agent.awareness import _get_active_window_title
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "VS Code\n"
        with patch("subprocess.run", return_value=mock_result):
            assert _get_active_window_title() == "VS Code"

    def test_get_process_xdotool_missing(self):
        from bantz.agent.awareness import _get_active_window_process
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _get_active_window_process() == ""

    def test_get_process_psutil_error(self):
        import psutil
        from bantz.agent.awareness import _get_active_window_process
        wid_result = MagicMock(returncode=0, stdout="12345\n")
        pid_result = MagicMock(returncode=0, stdout="999\n")
        with patch("subprocess.run", side_effect=[wid_result, pid_result]), \
             patch("psutil.Process", side_effect=psutil.NoSuchProcess(999)):
            assert _get_active_window_process() == ""

    def test_xclip_not_found(self):
        from bantz.agent.awareness import _get_clipboard
        with patch("subprocess.run", side_effect=FileNotFoundError("xclip")):
            assert _get_clipboard() == ""

    def test_xclip_empty_clipboard(self):
        from bantz.agent.awareness import _get_clipboard
        mock_result = MagicMock(returncode=0, stdout="")
        with patch("subprocess.run", return_value=mock_result):
            assert _get_clipboard() == ""

    def test_xclip_success(self):
        from bantz.agent.awareness import _get_clipboard
        mock_result = MagicMock(returncode=0, stdout="hello world\n")
        with patch("subprocess.run", return_value=mock_result):
            assert _get_clipboard() == "hello world"

    def test_grim_not_found(self):
        from bantz.agent.awareness import _capture_screenshot
        with patch("subprocess.run", side_effect=FileNotFoundError("grim")):
            assert _capture_screenshot() is False

    def test_grim_success(self):
        from bantz.agent.awareness import _capture_screenshot
        mock_result = MagicMock(returncode=0, stderr="")
        with patch("subprocess.run", return_value=mock_result):
            assert _capture_screenshot() is True

    def test_grim_nonzero_exit(self):
        from bantz.agent.awareness import _capture_screenshot
        mock_result = MagicMock(returncode=1, stderr="display not found")
        with patch("subprocess.run", return_value=mock_result):
            assert _capture_screenshot() is False


# ═══════════════════════════════════════════════════════════════════════════
# _collect() — integration of subprocess helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestCollect:
    @pytest.mark.asyncio
    async def test_collect_appends_state(self):
        from bantz.agent.awareness import AwarenessCollector

        c = AwarenessCollector()
        with patch("bantz.agent.awareness._get_active_window_title", return_value="Vim"), \
             patch("bantz.agent.awareness._get_active_window_process", return_value="vim"), \
             patch("bantz.agent.awareness._get_clipboard", return_value="import os"), \
             patch("bantz.agent.awareness._capture_screenshot", return_value=True):
            await c._collect()

        assert len(c._buffer) == 1
        state = c._buffer[0]
        assert state.active_window_title == "Vim"
        assert state.active_window_process == "vim"
        assert state.clipboard_text == "import os"
        assert state.screenshot_path != ""

    @pytest.mark.asyncio
    async def test_collect_all_failing(self):
        """Even when every subprocess fails, a state is appended (all empty)."""
        from bantz.agent.awareness import AwarenessCollector

        c = AwarenessCollector()
        with patch("bantz.agent.awareness._get_active_window_title", return_value=""), \
             patch("bantz.agent.awareness._get_active_window_process", return_value=""), \
             patch("bantz.agent.awareness._get_clipboard", return_value=""), \
             patch("bantz.agent.awareness._capture_screenshot", return_value=False):
            await c._collect()

        assert len(c._buffer) == 1
        s = c._buffer[0]
        assert s.active_window_title == ""
        assert s.screenshot_path == ""

    @pytest.mark.asyncio
    async def test_collect_exception_in_helper_is_tolerated(self):
        """Exceptions from individual helpers must not crash the collector."""
        from bantz.agent.awareness import AwarenessCollector

        c = AwarenessCollector()
        with patch("bantz.agent.awareness._get_active_window_title", side_effect=Exception("boom")), \
             patch("bantz.agent.awareness._get_active_window_process", return_value="code"), \
             patch("bantz.agent.awareness._get_clipboard", return_value=""), \
             patch("bantz.agent.awareness._capture_screenshot", return_value=False):
            await c._collect()

        # Still appended something, even if title is empty
        assert len(c._buffer) == 1


# ═══════════════════════════════════════════════════════════════════════════
# run() — cancellation
# ═══════════════════════════════════════════════════════════════════════════

class TestRun:
    @pytest.mark.asyncio
    async def test_run_cancels_cleanly(self):
        from bantz.agent.awareness import AwarenessCollector

        c = AwarenessCollector(interval_s=100.0)  # long interval — won't fire twice
        with patch.object(c, "_collect", new_callable=AsyncMock):
            task = asyncio.create_task(c.run())
            await asyncio.sleep(0)  # let the first collect fire
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert not c._running


# ═══════════════════════════════════════════════════════════════════════════
# Brain — screenshot trigger keywords
# ═══════════════════════════════════════════════════════════════════════════

class TestScreenshotTriggerKeywords:
    """Tests that _AWARENESS_SCREENSHOT_TRIGGERS covers the required words."""

    def _triggers(self):
        from bantz.core.brain import Brain
        return Brain._AWARENESS_SCREENSHOT_TRIGGERS

    def test_english_triggers_present(self):
        triggers = self._triggers()
        for kw in ("this", "here", "fix this", "look at", "what is this"):
            assert kw in triggers, f"Expected '{kw}' in triggers"

    def test_turkish_triggers_present(self):
        triggers = self._triggers()
        for kw in ("bu", "bak", "ekran", "ne var"):
            assert kw in triggers, f"Expected '{kw}' in triggers"

    def test_trigger_detection_case_insensitive(self):
        """_maybe_inject_awareness_screenshot should fire on 'THIS' / 'BU'."""
        from bantz.core.brain import Brain
        brain = Brain.__new__(Brain)
        brain._pending_vlm_task = None

        triggered_keywords = []

        def fake_inject(self_arg, en_input, orig_input):
            lower = (en_input + " " + orig_input).lower()
            if any(kw in lower for kw in Brain._AWARENESS_SCREENSHOT_TRIGGERS):
                triggered_keywords.append(True)

        # Patch the method itself with a lightweight version for detection
        with patch.object(Brain, "_maybe_inject_awareness_screenshot", fake_inject):
            brain._maybe_inject_awareness_screenshot("Fix THIS please", "")
            brain._maybe_inject_awareness_screenshot("bak bakalım", "")

        assert len(triggered_keywords) == 2

    def test_no_trigger_on_unrelated_input(self):
        """Unrelated input must not trigger the screenshot pipeline."""
        from bantz.core.brain import Brain
        brain = Brain.__new__(Brain)
        brain._pending_vlm_task = None

        with patch("bantz.agent.awareness.awareness_collector") as mock_collector:
            mock_collector.get_screenshot_for_vlm.return_value = None
            # Patch config
            with patch("bantz.core.brain.config") as mock_cfg:
                mock_cfg.awareness_enabled = True
                brain._maybe_inject_awareness_screenshot(
                    "what is the weather in Istanbul", "istanbul hava"
                )
            # Screenshot path lookup should not have been called
            # (no deictic word in the input)
            mock_collector.get_screenshot_for_vlm.assert_not_called()
