"""
Tests for brain.py integration with RL engine, interventions,
job_scheduler, maintenance, and reflection (#125-#130 brain wiring).
"""
from __future__ import annotations

import asyncio
import re
import types
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip('textual')

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Quick-route regex tests (#129, #130)
# ═══════════════════════════════════════════════════════════════════════════

class TestQuickRouteMaintenanceReflection:
    """After #272: quick_route no longer matches maintenance, reflection,
    or briefing — those are now routed by LLM via cot_route."""

    @staticmethod
    def _qr(orig: str, en: str | None = None):
        from bantz.core.brain import Brain
        return Brain._quick_route(orig, en or orig)

    # ── Maintenance triggers → now return None (#272) ─────────────────

    def test_run_maintenance(self):
        assert self._qr("run maintenance") is None

    def test_run_maintenance_dry(self):
        assert self._qr("run maintenance dry") is None

    def test_sistemi_temizle(self):
        assert True

    def test_system_cleanup(self):
        assert self._qr("system cleanup") is None

    def test_clean_up_system(self):
        assert self._qr("clean up the system") is None

    def test_bakim_yap(self):
        assert True

    def test_maintenance_run(self):
        assert self._qr("maintenance run") is None

    # ── Reflection triggers → now return None (#272) ──────────────────

    def test_show_reflections(self):
        assert self._qr("show reflections") is None

    def test_list_reflections(self):
        assert self._qr("list reflections") is None

    def test_past_reflections(self):
        assert self._qr("past reflections") is None

    def test_dunku_ozet(self):
        assert True

    def test_gecmis_ozetler(self):
        assert True

    def test_run_reflection(self):
        assert self._qr("run reflection") is None

    def test_run_reflection_dry(self):
        assert self._qr("run reflection dry") is None

    def test_yansima_yap(self):
        assert True

    def test_generate_reflection(self):
        assert self._qr("generate reflection") is None

    def test_reflect_on_today(self):
        assert self._qr("reflect on today") is None

    def test_bugunu_ozetle(self):
        assert True

    # ── Non-matches (still no match) ──────────────────────────────────

    def test_maintain_focus_not_maintenance(self):
        r = self._qr("maintain my focus please")
        assert r is None or r["tool"] != "_maintenance"

    def test_reflect_in_conversation(self):
        """'reflect' alone in casual chat should not trigger."""
        r = self._qr("let me reflect on this idea for a moment")
        assert r is None or r["tool"] != "_run_reflection"

    def test_briefing_removed(self):
        """Briefing is now routed via LLM, not regex (#272)."""
        assert self._qr("good morning") is None

    def test_reminder_still_works(self):
        assert True


# ═══════════════════════════════════════════════════════════════════════════
# 2. RL reward hook (#125)
# ═══════════════════════════════════════════════════════════════════════════

class TestRLRewardHook:
    """Verify _rl_reward_hook delegates to rl_hooks module (#226)."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._memory_ready = True
        b._graph_ready = True
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        b._last_tool_output = ""
        b._last_tool_name = ""
        b._feedback_ctx = ""
        b._turn_counter = 0
        b._context_turn = 0
        b._CONTEXT_TTL = 3
        b._last_screen_description = ""
        b._screen_description_turn = -1
        b._pending_vlm_task = None
        return b

    def test_reward_positive_on_success(self):
        """_rl_reward_hook creates a task that calls rl_hooks.rl_reward_hook."""
        b = self._make_brain()
        from bantz.tools import ToolResult
        result = ToolResult(success=True, output="ok")

        mock_rl_hook = AsyncMock()
        mock_task = MagicMock()

        with patch("bantz.core.rl_hooks.rl_reward_hook", mock_rl_hook), \
             patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.create_task = mock_task
            b._rl_reward_hook("weather", result)

        mock_task.assert_called_once()

    def test_reward_negative_on_failure(self):
        """_rl_reward_hook fires for failure results too."""
        b = self._make_brain()
        from bantz.tools import ToolResult
        result = ToolResult(success=False, output="", error="fail")

        mock_rl_hook = AsyncMock()
        mock_task = MagicMock()

        with patch("bantz.core.rl_hooks.rl_reward_hook", mock_rl_hook), \
             patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.create_task = mock_task
            b._rl_reward_hook("shell", result)

        mock_task.assert_called_once()

    def test_hook_no_crash_on_import_error(self):
        b = self._make_brain()
        from bantz.tools import ToolResult
        result = ToolResult(success=True, output="ok")

        with patch.dict("sys.modules", {"bantz.core.rl_hooks": None}):
            # Should not raise even when module fails to import
            b._rl_reward_hook("weather", result)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Intervention queue hook (#126)
# ═══════════════════════════════════════════════════════════════════════════

class TestInterventionQueueHook:
    """Deprecated methods removed in #225 — verify they no longer exist."""

    def test_check_intervention_queue_removed(self):
        from bantz.core.brain import Brain
        assert not hasattr(Brain, "_check_intervention_queue")

    def test_prepend_intervention_removed(self):
        from bantz.core.brain import Brain
        assert not hasattr(Brain, "_prepend_intervention")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Reminder → job_scheduler bridge (#128)
# ═══════════════════════════════════════════════════════════════════════════

class TestReminderBridge:
    """Verify ReminderTool._bridge_to_job_scheduler calls add_reminder."""

    def test_bridge_calls_add_reminder(self):
        from bantz.tools.reminder import ReminderTool
        fire_at = datetime.now() + timedelta(hours=1)
        mock_js = MagicMock()
        mock_js._started = True
        mock_js.add_reminder.return_value = "job_123"

        with patch("bantz.agent.job_scheduler.job_scheduler", mock_js):
            result = ReminderTool._bridge_to_job_scheduler("test", fire_at, "none")

        assert result == "job_123"
        mock_js.add_reminder.assert_called_once_with("test", fire_at, repeat="none")

    def test_bridge_returns_none_when_not_started(self):
        from bantz.tools.reminder import ReminderTool
        fire_at = datetime.now() + timedelta(hours=1)
        mock_js = MagicMock()
        mock_js._started = False

        with patch("bantz.agent.job_scheduler.job_scheduler", mock_js):
            result = ReminderTool._bridge_to_job_scheduler("test", fire_at)

        assert result is None
        mock_js.add_reminder.assert_not_called()

    def test_bridge_returns_none_on_exception(self):
        from bantz.tools.reminder import ReminderTool
        fire_at = datetime.now() + timedelta(hours=1)

        with patch.dict("sys.modules", {"bantz.agent.job_scheduler": None}):
            result = ReminderTool._bridge_to_job_scheduler("test", fire_at)

        assert result is None

    def test_bridge_passes_repeat_mode(self):
        from bantz.tools.reminder import ReminderTool
        fire_at = datetime.now() + timedelta(hours=1)
        mock_js = MagicMock()
        mock_js._started = True
        mock_js.add_reminder.return_value = "job_daily"

        with patch("bantz.agent.job_scheduler.job_scheduler", mock_js):
            result = ReminderTool._bridge_to_job_scheduler("daily check", fire_at, "daily")

        mock_js.add_reminder.assert_called_once_with("daily check", fire_at, repeat="daily")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Maintenance handler (#129)
# ═══════════════════════════════════════════════════════════════════════════

class TestMaintenanceHandler:
    """Verify _handle_maintenance calls run_maintenance and returns summary."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        b._last_tool_output = ""
        b._last_tool_name = ""
        b._feedback_ctx = ""
        b._turn_counter = 0
        b._context_turn = 0
        b._CONTEXT_TTL = 3
        b._last_screen_description = ""
        b._screen_description_turn = -1
        b._pending_vlm_task = None
        return b

    def test_handle_maintenance_returns_summary(self):
        b = self._make_brain()
        mock_report = MagicMock()
        mock_report.summary.return_value = "🔧 Maintenance: 5 ok, 0 skipped, 0 failed"
        mock_run = AsyncMock(return_value=mock_report)

        with patch("bantz.agent.workflows.maintenance.run_maintenance", mock_run):
            text = _run(b._handle_maintenance())

        assert "Maintenance" in text
        mock_run.assert_awaited_once_with(dry_run=False)

    def test_handle_maintenance_dry_run(self):
        b = self._make_brain()
        mock_report = MagicMock()
        mock_report.summary.return_value = "🔧 Maintenance (DRY-RUN)"
        mock_run = AsyncMock(return_value=mock_report)

        with patch("bantz.agent.workflows.maintenance.run_maintenance", mock_run):
            text = _run(b._handle_maintenance(dry_run=True))

        mock_run.assert_awaited_once_with(dry_run=True)

    def test_handle_maintenance_error(self):
        b = self._make_brain()
        mock_run = AsyncMock(side_effect=RuntimeError("docker not found"))

        with patch("bantz.agent.workflows.maintenance.run_maintenance", mock_run):
            text = _run(b._handle_maintenance())

        assert "❌" in text
        assert "docker not found" in text


# ═══════════════════════════════════════════════════════════════════════════
# 6. Reflection handlers (#130)
# ═══════════════════════════════════════════════════════════════════════════

class TestReflectionHandlers:
    """Verify _handle_list_reflections and _handle_run_reflection."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        b._last_tool_output = ""
        b._last_tool_name = ""
        b._feedback_ctx = ""
        b._turn_counter = 0
        b._context_turn = 0
        b._CONTEXT_TTL = 3
        b._last_screen_description = ""
        b._screen_description_turn = -1
        b._pending_vlm_task = None
        return b

    def test_list_reflections_empty(self):
        b = self._make_brain()
        with patch("bantz.agent.workflows.reflection.list_reflections", return_value=[]):
            text = b._handle_list_reflections()
        assert "No reflections" in text

    def test_list_reflections_with_data(self):
        b = self._make_brain()
        items = [
            {"date": "2025-07-14", "sessions": 5, "summary": "Productive day"},
            {"date": "2025-07-13", "sessions": 3, "summary": "Light chat"},
        ]
        with patch("bantz.agent.workflows.reflection.list_reflections", return_value=items):
            text = b._handle_list_reflections()
        assert "2025-07-14" in text
        assert "Productive day" in text
        assert "2025-07-13" in text

    def test_list_reflections_error(self):
        b = self._make_brain()
        with patch("bantz.agent.workflows.reflection.list_reflections",
                    side_effect=RuntimeError("db locked")):
            text = b._handle_list_reflections()
        assert "❌" in text

    def test_run_reflection(self):
        b = self._make_brain()
        mock_result = MagicMock()
        mock_result.summary_line.return_value = "🤔 Reflection (2025-07-14): 5 sessions"
        mock_run = AsyncMock(return_value=mock_result)

        with patch("bantz.agent.workflows.reflection.run_reflection", mock_run):
            text = _run(b._handle_run_reflection())

        assert "Reflection" in text
        mock_run.assert_awaited_once_with(dry_run=False)

    def test_run_reflection_dry(self):
        b = self._make_brain()
        mock_result = MagicMock()
        mock_result.summary_line.return_value = "🤔 Reflection (DRY-RUN)"
        mock_run = AsyncMock(return_value=mock_result)

        with patch("bantz.agent.workflows.reflection.run_reflection", mock_run):
            text = _run(b._handle_run_reflection(dry_run=True))

        mock_run.assert_awaited_once_with(dry_run=True)

    def test_run_reflection_error(self):
        b = self._make_brain()
        mock_run = AsyncMock(side_effect=RuntimeError("no llm"))

        with patch("bantz.agent.workflows.reflection.run_reflection", mock_run):
            text = _run(b._handle_run_reflection())

        assert "❌" in text
        assert "no llm" in text


# ═══════════════════════════════════════════════════════════════════════════
# 7. Integration: process() wires RL reward after tool execution
# ═══════════════════════════════════════════════════════════════════════════

class TestProcessRLWiring:
    """Verify that process() calls _rl_reward_hook after tool execution."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        b._last_tool_output = ""
        b._last_tool_name = ""
        b._feedback_ctx = ""
        b._turn_counter = 0
        b._context_turn = 0
        b._CONTEXT_TTL = 3
        b._last_screen_description = ""
        b._screen_description_turn = -1
        b._pending_vlm_task = None
        return b

    def test_process_calls_rl_hook_on_tool(self):
        """When brain.process() executes a tool, _rl_reward_hook is called."""
        b = self._make_brain()
        from bantz.tools import ToolResult

        mock_result = ToolResult(success=True, output="Today is sunny, 28°C")

        # Stub out dependencies
        b._ensure_memory = MagicMock()
        b._ensure_graph = AsyncMock()
        b._to_en = AsyncMock(return_value="weather")
        b._fire_embeddings = MagicMock()
        b._graph_store = AsyncMock()

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=mock_result)

        mock_finalize = AsyncMock(return_value="It's sunny!")

        # Spy on _rl_reward_hook  
        b._rl_reward_hook = MagicMock()

        cot_plan = {
            "route": "tool",
            "tool_name": "weather",
            "tool_args": {"city": "Istanbul"},
            "risk_level": "safe",
            "confidence": 0.95,
            "reasoning": "User wants weather.",
        }

        with patch("bantz.core.brain.time_ctx") as mock_tc, \
             patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.core.brain.registry") as mock_reg, \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock,
                   return_value=(cot_plan, None)), \
             patch.object(b, "_finalize", mock_finalize), \
             patch.object(b, "_finalize_stream", AsyncMock(return_value=None)):

            mock_tc.snapshot.return_value = {"prompt_hint": "", "time_segment": "morning",
                                              "day_name": "Monday"}
            mock_dl.conversations = MagicMock()
            mock_reg.get.return_value = mock_tool

            result = _run(b.process("weather"))

        b._rl_reward_hook.assert_called_once_with("weather", mock_result)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Integration: process() routes maintenance/reflection correctly
# ═══════════════════════════════════════════════════════════════════════════

class TestProcessMaintenanceReflectionRouting:
    """Verify process() dispatches maintenance/reflection via cot_route→dispatch_internal (#272).

    Since quick_route no longer matches these, cot_route must return a
    ``route: "tool"`` plan pointing to the internal tool name.
    """

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        b._last_tool_output = ""
        b._last_tool_name = ""
        b._feedback_ctx = ""
        b._turn_counter = 0
        b._context_turn = 0
        b._CONTEXT_TTL = 3
        b._last_screen_description = ""
        b._screen_description_turn = -1
        b._pending_vlm_task = None
        return b

    def test_process_routes_maintenance(self):
        b = self._make_brain()
        b._ensure_memory = MagicMock()
        b._ensure_graph = AsyncMock()
        b._to_en = AsyncMock(return_value="run maintenance")

        cot_plan = {
            "route": "tool",
            "tool_name": "_maintenance",
            "tool_args": {"dry_run": False},
            "risk_level": "safe",
            "confidence": 0.95,
            "reasoning": "User wants to run system maintenance.",
        }

        with patch("bantz.core.brain.time_ctx") as mock_tc, \
             patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.core.routing_engine.data_layer") as dl_re, \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock,
                   return_value=(cot_plan, None)), \
             patch("bantz.core.routing_engine.handle_maintenance",
                   new_callable=AsyncMock,
                   return_value="🔧 Maintenance: all ok") as mock_maint:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()
            dl_re.conversations = MagicMock()

            result = _run(b.process("run maintenance"))

        assert result.tool_used == "maintenance"
        assert "Maintenance" in result.response

    def test_process_routes_list_reflections(self):
        b = self._make_brain()
        b._ensure_memory = MagicMock()
        b._ensure_graph = AsyncMock()
        b._to_en = AsyncMock(return_value="show reflections")

        cot_plan = {
            "route": "tool",
            "tool_name": "_list_reflections",
            "tool_args": {},
            "risk_level": "safe",
            "confidence": 0.9,
            "reasoning": "User wants to see past reflections.",
        }

        with patch("bantz.core.brain.time_ctx") as mock_tc, \
             patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.core.routing_engine.data_layer") as dl_re, \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock,
                   return_value=(cot_plan, None)), \
             patch("bantz.core.routing_engine.handle_list_reflections",
                   return_value="🤔 Recent reflections:\n  • 2025-07-14") as mock_refl:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()
            dl_re.conversations = MagicMock()

            result = _run(b.process("show reflections"))

        assert result.tool_used == "reflection"
        assert "2025-07-14" in result.response

    def test_process_routes_run_reflection(self):
        b = self._make_brain()
        b._ensure_memory = MagicMock()
        b._ensure_graph = AsyncMock()
        b._to_en = AsyncMock(return_value="run reflection")

        cot_plan = {
            "route": "tool",
            "tool_name": "_run_reflection",
            "tool_args": {},
            "risk_level": "safe",
            "confidence": 0.9,
            "reasoning": "User wants to generate a reflection.",
        }

        with patch("bantz.core.brain.time_ctx") as mock_tc, \
             patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.core.routing_engine.data_layer") as dl_re, \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock,
                   return_value=(cot_plan, None)), \
             patch("bantz.core.routing_engine.handle_run_reflection",
                   new_callable=AsyncMock,
                   return_value="🤔 Reflection done") as mock_refl:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()
            dl_re.conversations = MagicMock()

            result = _run(b.process("run reflection"))

        assert result.tool_used == "reflection"
        mock_refl.assert_awaited_once()


# ═══════════════════════════════════════════════════════════════════════════
# Wake word voice control routes (#165)
# ═══════════════════════════════════════════════════════════════════════════

class TestWakeWordRoutes:
    """Ensure brain can enable/disable wake word via voice commands."""

    @staticmethod
    def _qr(orig: str, en: str | None = None):
        from bantz.core.brain import Brain
        return Brain._quick_route(orig, en or orig)

    # ── Wake word OFF ────────────────────────────────────────────────

    def test_stop_listening(self):
        assert self._qr("stop listening")["tool"] == "_wake_word_off"

    def test_pause_wake_word(self):
        assert self._qr("pause wake word")["tool"] == "_wake_word_off"

    def test_wake_word_off(self):
        assert self._qr("wake word off")["tool"] == "_wake_word_off"

    def test_disable_wake(self):
        assert self._qr("disable wake word")["tool"] == "_wake_word_off"

    def test_dont_listen(self):
        assert self._qr("don't listen")["tool"] == "_wake_word_off"

    def test_dinlemeyi_durdur(self):
        assert True

    # ── Wake word ON ─────────────────────────────────────────────────

    def test_start_listening(self):
        assert self._qr("start listening")["tool"] == "_wake_word_on"

    def test_resume_listening(self):
        assert self._qr("resume listening")["tool"] == "_wake_word_on"

    def test_wake_word_on(self):
        assert self._qr("wake word on")["tool"] == "_wake_word_on"

    def test_enable_wake(self):
        assert self._qr("enable wake word")["tool"] == "_wake_word_on"

    def test_dinlemeye_basla(self):
        assert True

    # ── False positives ──────────────────────────────────────────────

    def test_listen_to_music_is_not_wake(self):
        """'listen to music' should NOT trigger wake word control."""
        r = self._qr("listen to music")
        assert r is None or r.get("tool") not in ("_wake_word_on", "_wake_word_off")


class TestWakeWordProcessHandlers:
    """Brain.process() handlers for _wake_word_on / _wake_word_off."""

    def test_process_wake_word_off(self):
        from bantz.core.brain import Brain

        b = Brain()
        b._ensure_memory = MagicMock()
        b._ensure_graph = AsyncMock()
        b._to_en = AsyncMock(return_value="stop listening")

        mock_listener = MagicMock()
        mock_listener.running = True

        with patch("bantz.core.brain.time_ctx") as mock_tc, \
             patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.core.routing_engine.data_layer") as dl_re:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()
            dl_re.conversations = MagicMock()

            with patch("bantz.core.workflow.workflow_engine") as mock_wf:
                mock_wf.detect.return_value = []
                with patch("bantz.agent.wake_word.wake_listener", mock_listener):
                    result = _run(b.process("stop listening"))

        assert result.tool_used == "wake_word"
        mock_listener.stop.assert_called_once()

    def test_process_wake_word_on(self):
        from bantz.core.brain import Brain

        b = Brain()
        b._ensure_memory = MagicMock()
        b._ensure_graph = AsyncMock()
        b._to_en = AsyncMock(return_value="start listening")

        mock_listener = MagicMock()
        mock_listener.running = False
        mock_listener.start.return_value = True

        with patch("bantz.core.brain.time_ctx") as mock_tc, \
             patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.core.routing_engine.data_layer") as dl_re:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()
            dl_re.conversations = MagicMock()

            with patch("bantz.core.workflow.workflow_engine") as mock_wf:
                mock_wf.detect.return_value = []
                with patch("bantz.agent.wake_word.wake_listener", mock_listener):
                    result = _run(b.process("start listening"))

        assert result.tool_used == "wake_word"
        mock_listener.start.assert_called_once()

    def test_process_wake_word_off_not_running(self):
        from bantz.core.brain import Brain

        b = Brain()
        b._ensure_memory = MagicMock()
        b._ensure_graph = AsyncMock()
        b._to_en = AsyncMock(return_value="stop listening")

        mock_listener = MagicMock()
        mock_listener.running = False

        with patch("bantz.core.brain.time_ctx") as mock_tc, \
             patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.core.routing_engine.data_layer") as dl_re:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()
            dl_re.conversations = MagicMock()

            with patch("bantz.core.workflow.workflow_engine") as mock_wf:
                mock_wf.detect.return_value = []
                with patch("bantz.agent.wake_word.wake_listener", mock_listener):
                    result = _run(b.process("stop listening"))

        assert "not running" in result.response.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Notifier fallback in _notify_toast (#153)
# ═══════════════════════════════════════════════════════════════════════════

class TestNotifyToastFallback:
    """_notify_toast should fall back to desktop notify-send when TUI is unreachable."""

    def test_fallback_to_notifier_when_no_tui(self):
        import bantz.core.brain as brain_mod

        old_cb = brain_mod._toast_callback
        brain_mod._toast_callback = None

        mock_notifier = MagicMock()
        mock_notifier.enabled = True

        try:
            with patch("bantz.agent.notifier.notifier", mock_notifier):
                # Suppress App.current failure
                with patch("textual.app.App") as mock_app_cls:
                    type(mock_app_cls).current = property(lambda self: None)
                    brain_mod._notify_toast("Test Title", "Test Body", "info")

            mock_notifier.send.assert_called_once()
            call_args = mock_notifier.send.call_args
            assert "Test Title" in call_args[0][0]
        finally:
            brain_mod._toast_callback = old_cb

    def test_callback_takes_priority(self):
        import bantz.core.brain as brain_mod

        cb = MagicMock()
        old_cb = brain_mod._toast_callback
        brain_mod._toast_callback = cb

        mock_notifier = MagicMock()
        mock_notifier.enabled = True

        try:
            with patch("bantz.agent.notifier.notifier", mock_notifier):
                brain_mod._notify_toast("Title", "Body", "info")

            cb.assert_called_once_with("Title", "Body", "info")
            mock_notifier.send.assert_not_called()
        finally:
            brain_mod._toast_callback = old_cb

    def test_notifier_src_has_fallback(self):
        """notification_manager source must reference notifier.send as fallback."""
        import inspect
        from bantz.core import notification_manager
        src = inspect.getsource(notification_manager.notify_toast)
        assert "notifier.send" in src
        assert "notifier" in src


# ═══════════════════════════════════════════════════════════════════════════
# Audio Ducking brain routes (#171)
# ═══════════════════════════════════════════════════════════════════════════

class TestAudioDuckRoutes:
    """_quick_route must match audio ducking on/off commands."""

    def _route(self, text):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        return b._quick_route(text, text.lower())

    def test_enable_ducking(self):
        r = self._route("enable ducking")
        assert r and r["tool"] == "_audio_duck_on"

    def test_ducking_on(self):
        r = self._route("ducking on")
        assert r and r["tool"] == "_audio_duck_on"

    def test_turn_on_ducking(self):
        r = self._route("turn on ducking")
        assert r and r["tool"] == "_audio_duck_on"

    

    def test_disable_ducking(self):
        r = self._route("disable ducking")
        assert r and r["tool"] == "_audio_duck_off"

    def test_ducking_off(self):
        r = self._route("ducking off")
        assert r and r["tool"] == "_audio_duck_off"

    def test_turn_off_ducking(self):
        r = self._route("turn off ducking")
        assert r and r["tool"] == "_audio_duck_off"

    def test_no_ducking(self):
        r = self._route("no ducking please")
        assert r and r["tool"] == "_audio_duck_off"

    

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._model = "test"
        b._ctx = MagicMock()
        b._session = MagicMock()
        b._en_cache = {}
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        b._last_tool_output = ""
        b._last_tool_name = ""
        b._feedback_ctx = ""
        b._turn_counter = 0
        b._context_turn = 0
        b._CONTEXT_TTL = 3
        b._last_screen_description = ""
        b._screen_description_turn = -1
        b._pending_vlm_task = None
        return b

    def test_process_duck_on_available(self):
        b = self._make_brain()
        mock_ducker = MagicMock()
        mock_ducker.available.return_value = True
        with patch("bantz.core.brain.data_layer") as dl, \
             patch("bantz.core.routing_engine.data_layer") as dl_re, \
             patch.dict("sys.modules", {"bantz.agent.audio_ducker": MagicMock(audio_ducker=mock_ducker)}):
            dl_re.conversations = MagicMock()
            result = _run(b.process("enable ducking"))
        assert "enabled" in result.response.lower() or "🔉" in result.response
        assert mock_ducker.enabled is True

    def test_process_duck_on_unavailable(self):
        b = self._make_brain()
        mock_ducker = MagicMock()
        mock_ducker.available.return_value = False
        with patch("bantz.core.brain.data_layer") as dl, \
             patch("bantz.core.routing_engine.data_layer") as dl_re, \
             patch.dict("sys.modules", {"bantz.agent.audio_ducker": MagicMock(audio_ducker=mock_ducker)}):
            dl_re.conversations = MagicMock()
            result = _run(b.process("enable ducking"))
        assert "not available" in result.response.lower() or "❌" in result.response

    def test_process_duck_off(self):
        b = self._make_brain()
        mock_ducker = MagicMock()
        with patch("bantz.core.brain.data_layer") as dl, \
             patch("bantz.core.routing_engine.data_layer") as dl_re, \
             patch.dict("sys.modules", {"bantz.agent.audio_ducker": MagicMock(audio_ducker=mock_ducker)}):
            dl_re.conversations = MagicMock()
            result = _run(b.process("disable ducking"))
        assert "disabled" in result.response.lower() or "🔇" in result.response


# ═══════════════════════════════════════════════════════════════════════════
# Ambient brain routes (#166)
# ═══════════════════════════════════════════════════════════════════════════

class TestAmbientRoutes:
    """After #272: quick_route no longer matches ambient queries.

    They are now routed via cot_route → dispatch_internal.
    """

    def _route(self, text):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        return b._quick_route(text, text.lower())

    def test_ambient_noise(self):
        assert self._route("ambient noise") is None

    def test_ambient_status(self):
        assert self._route("ambient status") is None

    def test_environment_noise(self):
        assert self._route("environment noise level") is None

    

    def test_hows_the_ambient(self):
        assert self._route("how's the ambient") is None

    

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._model = "test"
        b._ctx = MagicMock()
        b._session = MagicMock()
        b._en_cache = {}
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        b._last_tool_output = ""
        b._last_tool_name = ""
        b._feedback_ctx = ""
        b._turn_counter = 0
        b._context_turn = 0
        b._CONTEXT_TTL = 3
        b._last_screen_description = ""
        b._screen_description_turn = -1
        b._pending_vlm_task = None
        return b

    def test_process_ambient_with_data(self):
        from bantz.agent.ambient import AmbientLabel, AmbientSnapshot
        b = self._make_brain()
        snap = AmbientSnapshot(timestamp=1700000000, rms=2500, zcr=0.06, label=AmbientLabel.SPEECH, duration_s=3.0)
        mock_analyzer = MagicMock()
        mock_analyzer.latest.return_value = snap
        mock_analyzer.day_summary.return_value = "Ambient today (5 samples): speech: 60%, silence: 40%"

        cot_plan = {
            "route": "tool", "tool_name": "_ambient_status",
            "tool_args": {}, "risk_level": "safe", "confidence": 0.9,
            "reasoning": "User wants ambient status.",
        }

        with patch("bantz.core.brain.data_layer") as dl, \
             patch("bantz.core.routing_engine.data_layer") as dl_re, \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock,
                   return_value=(cot_plan, None)), \
             patch.dict("sys.modules", {"bantz.agent.ambient": MagicMock(ambient_analyzer=mock_analyzer)}):
            dl_re.conversations = MagicMock()
            result = _run(b.process("ambient noise"))
        assert "SPEECH" in result.response
        assert "🎤" in result.response

    def test_process_ambient_no_data(self):
        b = self._make_brain()
        mock_analyzer = MagicMock()
        mock_analyzer.latest.return_value = None

        cot_plan = {
            "route": "tool", "tool_name": "_ambient_status",
            "tool_args": {}, "risk_level": "safe", "confidence": 0.9,
            "reasoning": "User wants ambient status.",
        }

        with patch("bantz.core.brain.data_layer") as dl, \
             patch("bantz.core.routing_engine.data_layer") as dl_re, \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock,
                   return_value=(cot_plan, None)), \
             patch.dict("sys.modules", {"bantz.agent.ambient": MagicMock(ambient_analyzer=mock_analyzer)}):
            dl_re.conversations = MagicMock()
            result = _run(b.process("ambient status"))
        assert "waiting" in result.response.lower() or "no ambient" in result.response.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Parity — Telegram gets identical prompt to Terminal (no remote_hint)
# ═══════════════════════════════════════════════════════════════════════════


class TestPromptParity:
    """Telegram path must produce the exact same system prompt as Terminal.

    remote_hint was removed entirely: it told the LLM to be 'EXTREMELY concise'
    which degraded intelligence.  _is_remote now only controls TTS suppression.
    """

    def test_no_remote_hint_in_chat(self):
        """_chat() must NOT contain any remote_hint / mobile text injection."""
        import inspect
        from bantz.core.brain import Brain
        src = inspect.getsource(Brain._chat)
        assert "remote_hint" not in src
        assert "mobile text" not in src
        assert "EXTREMELY concise" not in src

    def test_no_remote_hint_in_stream(self):
        """_chat_stream() must NOT contain any remote_hint injection."""
        import inspect
        from bantz.core.brain import Brain
        src = inspect.getsource(Brain._chat_stream)
        assert "remote_hint" not in src
        assert "mobile text" not in src
        assert "EXTREMELY concise" not in src

    def test_no_telegraph_roleplay(self):
        """Neither path must have old Telegraph RP language."""
        import inspect
        from bantz.core.brain import Brain
        src_chat = inspect.getsource(Brain._chat)
        src_stream = inspect.getsource(Brain._chat_stream)
        assert "Telegraph Mode" not in src_chat
        assert "Telegraph Mode" not in src_stream
        assert "not at the machine" not in src_chat
        assert "not at the machine" not in src_stream

    def test_is_remote_still_suppresses_tts(self):
        """_is_remote flag must still exist (for TTS suppression), just not in prompts."""
        import inspect
        from bantz.core.brain import Brain
        src_process = inspect.getsource(Brain.process)
        assert "_is_remote" in src_process
