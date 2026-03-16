"""Tests for ``bantz.core.routing_engine`` — quick_route, dispatch_internal,
generate_command, execute_plan, and workflow handlers (#228, #272)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════
# 1. quick_route — hardware/UI controls only (#272)
# ═══════════════════════════════════════════════════════════════════════════

class TestQuickRoute:
    """quick_route now ONLY matches hardware/UI toggles.

    All semantic routing (shell, system metrics, briefing, maintenance,
    reflection, ambient, …) was purged in #272 — the LLM decides via
    cot_route instead.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from bantz.core.routing_engine import quick_route
        self.qr = quick_route

    # ── TTS stop ──────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "shut up",
        "be quiet",
        "stop talking",
    ])
    def test_tts_stop(self, text):
        r = self.qr(text, text)
        assert r is not None
        assert r["tool"] == "_tts_stop"

    # ── Wake word on/off ──────────────────────────────────────────────
    @pytest.mark.parametrize("text,expected", [
        ("start listening", "_wake_word_on"),
        ("wake word on", "_wake_word_on"),
        ("stop listening", "_wake_word_off"),
        ("wake word off", "_wake_word_off"),
    ])
    def test_wake_word_toggle(self, text, expected):
        r = self.qr(text, text)
        assert r is not None
        assert r["tool"] == expected

    # ── Audio ducking ─────────────────────────────────────────────────
    @pytest.mark.parametrize("text,expected", [
        ("enable ducking", "_audio_duck_on"),
        ("ducking off", "_audio_duck_off"),
    ])
    def test_audio_ducking(self, text, expected):
        r = self.qr(text, text)
        assert r is not None
        assert r["tool"] == expected

    # ── Clear memory ──────────────────────────────────────────────────
    def test_clear_memory(self):
        r = self.qr("clear memory", "clear memory")
        assert r is not None
        assert r["tool"] == "_clear_memory"

    # ── Previously regex-matched queries now go through LLM (#272) ────
    @pytest.mark.parametrize("text", [
        # Shell commands — no longer regex-matched
        "ls -la",
        "df -h",
        "cat /etc/hosts",
        "pwd",
        # System metrics
        "system status",
        "cpu usage",
        "memory usage",
        "check disk space",
        "how much RAM is free",
        # Status queries
        "ambient status",
        "proactive status",
        "health status",
        # Briefing
        "good morning",
        "morning briefing",
        "what's today",
        # Maintenance / Reflection
        "run maintenance",
        "run maintenance dry",
        "run reflection",
        "show reflections",
    ])
    def test_removed_regex_returns_none(self, text):
        """Queries that were previously regex-matched must now return None."""
        assert self.qr(text, text) is None

    # ── No match → None ──────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "hello there",
        "what is the weather",
        "tell me a joke",
    ])
    def test_no_match(self, text):
        assert self.qr(text, text) is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. dispatch_internal — internal tool execution
# ═══════════════════════════════════════════════════════════════════════════

class TestDispatchInternal:
    """Verify dispatch_internal returns BrainResult for known tools, None otherwise."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from bantz.core.routing_engine import dispatch_internal
        self.dispatch = dispatch_internal

    def test_tts_stop_speaking(self):
        with patch("bantz.core.routing_engine.data_layer") as dl:
            dl.conversations = MagicMock()
            with patch("bantz.agent.tts.tts_engine") as tts:
                tts.is_speaking = True
                result = _run(self.dispatch(
                    "_tts_stop", {}, "stop", "stop", {},
                ))
        assert result is not None
        assert result.tool_used == "tts"
        assert "Stopped" in result.response
        tts.stop.assert_called_once()

    def test_tts_stop_not_speaking(self):
        with patch("bantz.core.routing_engine.data_layer") as dl:
            dl.conversations = MagicMock()
            with patch("bantz.agent.tts.tts_engine") as tts:
                tts.is_speaking = False
                result = _run(self.dispatch(
                    "_tts_stop", {}, "stop", "stop", {},
                ))
        assert "not speaking" in result.response

    def test_maintenance_dispatch(self):
        with patch("bantz.core.routing_engine.data_layer") as dl, \
             patch("bantz.core.routing_engine.handle_maintenance",
                   new_callable=AsyncMock,
                   return_value="🔧 Done") as mock_m:
            dl.conversations = MagicMock()
            result = _run(self.dispatch(
                "_maintenance", {"dry_run": False},
                "run maintenance", "run maintenance", {},
            ))
        assert result is not None
        assert result.tool_used == "maintenance"
        mock_m.assert_awaited_once_with(False)

    def test_unknown_tool_returns_none(self):
        with patch("bantz.core.routing_engine.data_layer"):
            result = _run(self.dispatch(
                "some_external_tool", {}, "", "", {},
            ))
        assert result is None

    def test_list_reflections_dispatch(self):
        with patch("bantz.core.routing_engine.data_layer") as dl, \
             patch("bantz.core.routing_engine.handle_list_reflections",
                   return_value="🤔 Recent reflections:") as mock_r:
            dl.conversations = MagicMock()
            result = _run(self.dispatch(
                "_list_reflections", {},
                "show reflections", "show reflections", {},
            ))
        assert result is not None
        assert "reflections" in result.response.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 3. generate_command — LLM bash generation
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateCommand:
    def test_strips_backticks(self):
        with patch("bantz.core.routing_engine.ollama") as mock_ollama:
            mock_ollama.chat = AsyncMock(return_value="```ls -la```")
            from bantz.core.routing_engine import generate_command
            result = _run(generate_command("list files", "list files"))
        assert result == "ls -la"

    def test_plain_command(self):
        with patch("bantz.core.routing_engine.ollama") as mock_ollama:
            mock_ollama.chat = AsyncMock(return_value="df -h\n")
            from bantz.core.routing_engine import generate_command
            result = _run(generate_command("disk space", "disk space"))
        assert result == "df -h"


# ═══════════════════════════════════════════════════════════════════════════
# 4. execute_plan — Plan-and-Solve
# ═══════════════════════════════════════════════════════════════════════════

class TestExecutePlan:
    def test_returns_none_for_simple_request(self):
        with patch("bantz.core.routing_engine.registry") as mock_reg, \
             patch("bantz.agent.planner.planner_agent") as mock_planner:
            mock_reg.names.return_value = ["shell", "weather"]
            mock_planner.decompose = AsyncMock(return_value=[])  # no steps
            from bantz.core.routing_engine import execute_plan
            result = _run(execute_plan("hello", "hello", {}))
        assert result is None

    def test_returns_brain_result_for_complex(self):
        steps = [
            {"tool": "weather", "args": {}},
            {"tool": "shell", "args": {"command": "echo done"}},
        ]
        exec_result = MagicMock()
        exec_result.summary.return_value = "All done."

        with patch("bantz.core.routing_engine.registry") as mock_reg, \
             patch("bantz.agent.planner.planner_agent") as mock_planner, \
             patch("bantz.agent.executor.plan_executor") as mock_exec, \
             patch("bantz.core.routing_engine.data_layer") as dl, \
             patch("bantz.core.routing_engine.ollama") as mock_ollama:
            mock_reg.names.return_value = ["shell", "weather"]
            mock_planner.decompose = AsyncMock(return_value=steps)
            mock_planner.format_itinerary.return_value = "Plan:\n1. weather\n2. shell"
            mock_exec.run = AsyncMock(return_value=exec_result)
            dl.conversations = MagicMock()
            mock_ollama.chat = AsyncMock(return_value="ok")

            from bantz.core.routing_engine import execute_plan
            result = _run(execute_plan("complex task", "complex task", {}))

        assert result is not None
        assert result.tool_used == "planner"
        assert "All done" in result.response

    def test_passes_recent_history_to_decompose(self):
        """execute_plan forwards recent_history to planner_agent.decompose (#212)."""
        with patch("bantz.core.routing_engine.registry") as mock_reg, \
             patch("bantz.agent.planner.planner_agent") as mock_planner, \
             patch("bantz.core.routing_engine.data_layer") as dl:
            mock_reg.names.return_value = ["shell"]
            mock_planner.decompose = AsyncMock(return_value=[])
            dl.conversations = MagicMock()

            history = [
                {"role": "user", "content": "Who is John?"},
                {"role": "assistant", "content": "John is your colleague."},
            ]
            from bantz.core.routing_engine import execute_plan
            _run(execute_plan("send him the file", "send him the file", {},
                              recent_history=history))

        # Verify decompose was called with recent_history
        mock_planner.decompose.assert_awaited_once()
        call_kwargs = mock_planner.decompose.call_args
        assert call_kwargs.kwargs.get("recent_history") is history

    def test_execute_plan_without_history_still_works(self):
        """execute_plan without recent_history defaults to None (#212 backward compat)."""
        with patch("bantz.core.routing_engine.registry") as mock_reg, \
             patch("bantz.agent.planner.planner_agent") as mock_planner, \
             patch("bantz.core.routing_engine.data_layer") as dl:
            mock_reg.names.return_value = ["shell"]
            mock_planner.decompose = AsyncMock(return_value=[])
            dl.conversations = MagicMock()

            from bantz.core.routing_engine import execute_plan
            result = _run(execute_plan("hello", "hello", {}))

        assert result is None
        call_kwargs = mock_planner.decompose.call_args
        assert call_kwargs.kwargs.get("recent_history") is None


# ═══════════════════════════════════════════════════════════════════════════
# 5. Workflow handlers
# ═══════════════════════════════════════════════════════════════════════════

class TestWorkflowHandlers:
    def test_handle_maintenance_success(self):
        report = MagicMock()
        report.summary.return_value = "Maintenance complete."
        with patch("bantz.agent.workflows.maintenance.run_maintenance",
                   new_callable=AsyncMock, return_value=report):
            from bantz.core.routing_engine import handle_maintenance
            result = _run(handle_maintenance())
        assert "Maintenance complete" in result

    def test_handle_maintenance_error(self):
        with patch("bantz.agent.workflows.maintenance.run_maintenance",
                   new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            from bantz.core.routing_engine import handle_maintenance
            result = _run(handle_maintenance())
        assert "❌" in result
        assert "boom" in result

    def test_handle_list_reflections_empty(self):
        with patch("bantz.agent.workflows.reflection.list_reflections",
                   return_value=[]):
            from bantz.core.routing_engine import handle_list_reflections
            result = handle_list_reflections()
        assert "No reflections" in result

    def test_handle_list_reflections_items(self):
        items = [{"date": "2025-07-14", "summary": "Good day", "sessions": 3}]
        with patch("bantz.agent.workflows.reflection.list_reflections",
                   return_value=items):
            from bantz.core.routing_engine import handle_list_reflections
            result = handle_list_reflections()
        assert "2025-07-14" in result
        assert "3 sessions" in result

    def test_handle_run_reflection_success(self):
        ref_result = MagicMock()
        ref_result.summary_line.return_value = "Reflected."
        with patch("bantz.agent.workflows.reflection.run_reflection",
                   new_callable=AsyncMock, return_value=ref_result):
            from bantz.core.routing_engine import handle_run_reflection
            result = _run(handle_run_reflection())
        assert "Reflected" in result

    def test_handle_run_reflection_error(self):
        with patch("bantz.agent.workflows.reflection.run_reflection",
                   new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            from bantz.core.routing_engine import handle_run_reflection
            result = _run(handle_run_reflection())
        assert "❌" in result
        assert "fail" in result
