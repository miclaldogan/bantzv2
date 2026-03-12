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
    """Ensure _quick_route returns correct pseudo-tool for maintenance & reflection."""

    @staticmethod
    def _qr(orig: str, en: str | None = None):
        from bantz.core.brain import Brain
        return Brain._quick_route(orig, en or orig)

    # ── Maintenance triggers ──────────────────────────────────────────

    def test_run_maintenance(self):
        assert self._qr("run maintenance")["tool"] == "_maintenance"

    def test_run_maintenance_dry(self):
        r = self._qr("run maintenance dry")
        assert r["tool"] == "_maintenance"
        assert r["args"]["dry_run"] is True

    def test_sistemi_temizle(self):
        assert self._qr("sistemi temizle")["tool"] == "_maintenance"

    def test_system_cleanup(self):
        assert self._qr("system cleanup")["tool"] == "_maintenance"

    def test_clean_up_system(self):
        assert self._qr("clean up the system")["tool"] == "_maintenance"

    def test_bakim_yap(self):
        assert self._qr("bakım yap")["tool"] == "_maintenance"

    def test_maintenance_run(self):
        assert self._qr("maintenance run")["tool"] == "_maintenance"

    # ── Reflection triggers: list ─────────────────────────────────────

    def test_show_reflections(self):
        assert self._qr("show reflections")["tool"] == "_list_reflections"

    def test_list_reflections(self):
        assert self._qr("list reflections")["tool"] == "_list_reflections"

    def test_past_reflections(self):
        assert self._qr("past reflections")["tool"] == "_list_reflections"

    def test_dunku_ozet(self):
        assert self._qr("dünkü özet")["tool"] == "_list_reflections"

    def test_gecmis_ozetler(self):
        assert self._qr("geçmiş özetler")["tool"] == "_list_reflections"

    # ── Reflection triggers: run ──────────────────────────────────────

    def test_run_reflection(self):
        assert self._qr("run reflection")["tool"] == "_run_reflection"

    def test_run_reflection_dry(self):
        r = self._qr("run reflection dry")
        assert r["tool"] == "_run_reflection"
        assert r["args"]["dry_run"] is True

    def test_yansima_yap(self):
        assert self._qr("yansıma yap")["tool"] == "_run_reflection"

    def test_generate_reflection(self):
        assert self._qr("generate reflection")["tool"] == "_run_reflection"

    def test_reflect_on_today(self):
        assert self._qr("reflect on today")["tool"] == "_run_reflection"

    def test_bugunu_ozetle(self):
        assert self._qr("bugünü özetle")["tool"] == "_run_reflection"

    # ── Non-matches (should NOT trigger maintenance/reflection) ───────

    def test_maintain_focus_not_maintenance(self):
        r = self._qr("maintain my focus please")
        assert r is None or r["tool"] != "_maintenance"

    def test_reflect_in_conversation(self):
        """'reflect' alone in casual chat should not trigger."""
        r = self._qr("let me reflect on this idea for a moment")
        assert r is None or r["tool"] != "_run_reflection"

    def test_briefing_still_works(self):
        assert self._qr("good morning")["tool"] == "_briefing"

    def test_reminder_still_works(self):
        assert self._qr("remind me to call dentist at 3pm")["tool"] == "reminder"


# ═══════════════════════════════════════════════════════════════════════════
# 2. RL reward hook (#125)
# ═══════════════════════════════════════════════════════════════════════════

class TestRLRewardHook:
    """Verify _rl_reward_hook calls rl_engine.reward on tool success/failure."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        return b

    def test_reward_positive_on_success(self):
        b = self._make_brain()
        from bantz.tools import ToolResult
        result = ToolResult(success=True, output="ok")

        mock_engine = MagicMock()
        mock_engine.initialized = True
        mock_encode = MagicMock(return_value=MagicMock(key="morning|monday|home|weather"))

        with patch("bantz.agent.rl_engine.rl_engine", mock_engine), \
             patch("bantz.agent.rl_engine.encode_state", mock_encode):
            b._rl_reward_hook("weather", result)

        mock_engine.reward.assert_called_once()
        call_args = mock_engine.reward.call_args
        assert call_args[0][0] == 1.0  # positive reward

    def test_reward_negative_on_failure(self):
        b = self._make_brain()
        from bantz.tools import ToolResult
        result = ToolResult(success=False, output="", error="fail")

        mock_engine = MagicMock()
        mock_engine.initialized = True
        mock_encode = MagicMock(return_value=MagicMock(key="morning|monday|home|shell"))

        with patch("bantz.agent.rl_engine.rl_engine", mock_engine), \
             patch("bantz.agent.rl_engine.encode_state", mock_encode):
            b._rl_reward_hook("shell", result)

        mock_engine.reward.assert_called_once()
        call_args = mock_engine.reward.call_args
        assert call_args[0][0] == -0.5  # negative reward

    def test_hook_noop_when_uninitialized(self):
        b = self._make_brain()
        from bantz.tools import ToolResult
        result = ToolResult(success=True, output="ok")

        mock_engine = MagicMock()
        mock_engine.initialized = False

        with patch("bantz.agent.rl_engine.rl_engine", mock_engine):
            b._rl_reward_hook("shell", result)

        mock_engine.reward.assert_not_called()

    def test_hook_no_crash_on_import_error(self):
        b = self._make_brain()
        from bantz.tools import ToolResult
        result = ToolResult(success=True, output="ok")

        with patch.dict("sys.modules", {"bantz.agent.rl_engine": None}):
            # Should not raise even when module fails to import
            b._rl_reward_hook("weather", result)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Intervention queue hook (#126)
# ═══════════════════════════════════════════════════════════════════════════

class TestInterventionQueueHook:
    """Verify _check_intervention_queue pops next intervention."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        return b

    def test_pop_returns_formatted_text(self):
        b = self._make_brain()
        mock_iv = MagicMock()
        mock_iv.source = "rl_engine"
        mock_iv.title = "Start focus mode"
        mock_iv.reason = "You usually focus at this time"

        mock_queue = MagicMock()
        mock_queue.pop.return_value = mock_iv

        with patch("bantz.agent.interventions.intervention_queue", mock_queue):
            text = _run(b._check_intervention_queue())

        assert text is not None
        assert "rl_engine" in text
        assert "Start focus mode" in text

    def test_pop_returns_none_when_empty(self):
        b = self._make_brain()
        mock_queue = MagicMock()
        mock_queue.pop.return_value = None

        with patch("bantz.agent.interventions.intervention_queue", mock_queue):
            text = _run(b._check_intervention_queue())

        assert text is None

    def test_pop_no_crash_on_error(self):
        b = self._make_brain()
        with patch.dict("sys.modules", {"bantz.agent.interventions": None}):
            text = _run(b._check_intervention_queue())
        assert text is None

    def test_prepend_intervention_with_text(self):
        b = self._make_brain()
        b._pending_intervention = "💡 [observer] High CPU\n   CPU at 95%"
        result = b._prepend_intervention("It's sunny outside.")
        assert result.startswith("💡 [observer]")
        assert "It's sunny outside." in result
        # Consumed after use
        assert b._pending_intervention is None

    def test_prepend_intervention_none(self):
        b = self._make_brain()
        b._pending_intervention = None
        result = b._prepend_intervention("Hello")
        assert result == "Hello"

    def test_prepend_intervention_no_attr(self):
        b = self._make_brain()
        # No _pending_intervention set at all
        result = b._prepend_intervention("Hello")
        assert result == "Hello"


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

        with patch("bantz.core.brain.time_ctx") as mock_tc, \
             patch("bantz.core.brain.data_layer") as mock_dl, \
             patch("bantz.core.brain.registry") as mock_reg, \
             patch("bantz.core.brain.cot_route", new_callable=AsyncMock) as mock_cot, \
             patch.object(b, "_finalize", mock_finalize), \
             patch.object(b, "_finalize_stream", AsyncMock(return_value=None)):

            mock_tc.snapshot.return_value = {"prompt_hint": "", "time_segment": "morning",
                                              "day_name": "Monday"}
            mock_dl.conversations = MagicMock()
            mock_reg.get.return_value = mock_tool

            # quick_route returns weather tool
            from bantz.core.brain import Brain
            with patch.object(Brain, "_quick_route", return_value={
                "tool": "weather", "args": {"city": "Istanbul"}
            }):
                result = _run(b.process("weather"))

        b._rl_reward_hook.assert_called_once_with("weather", mock_result)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Integration: process() routes maintenance/reflection correctly
# ═══════════════════════════════════════════════════════════════════════════

class TestProcessMaintenanceReflectionRouting:
    """Verify process() actually dispatches to maintenance/reflection handlers."""

    def _make_brain(self):
        from bantz.core.brain import Brain
        b = Brain.__new__(Brain)
        b._bridge = False
        b._memory_ready = True
        b._graph_ready = True
        b._last_messages = []
        b._last_events = []
        b._last_draft = None
        return b

    def test_process_routes_maintenance(self):
        b = self._make_brain()
        b._ensure_memory = MagicMock()
        b._ensure_graph = AsyncMock()
        b._to_en = AsyncMock(return_value="run maintenance")
        b._handle_maintenance = AsyncMock(return_value="🔧 Maintenance: all ok")

        with patch("bantz.core.brain.time_ctx") as mock_tc, \
             patch("bantz.core.brain.data_layer") as mock_dl:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()

            # workflow_engine.detect returns [] (no workflow)
            with patch("bantz.core.workflow.workflow_engine") as mock_wf:
                mock_wf.detect.return_value = []
                result = _run(b.process("run maintenance"))

        assert result.tool_used == "maintenance"
        assert "Maintenance" in result.response

    def test_process_routes_list_reflections(self):
        b = self._make_brain()
        b._ensure_memory = MagicMock()
        b._ensure_graph = AsyncMock()
        b._to_en = AsyncMock(return_value="show reflections")
        b._handle_list_reflections = MagicMock(return_value="🤔 Recent reflections:\n  • 2025-07-14")

        with patch("bantz.core.brain.time_ctx") as mock_tc, \
             patch("bantz.core.brain.data_layer") as mock_dl:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()

            with patch("bantz.core.workflow.workflow_engine") as mock_wf:
                mock_wf.detect.return_value = []
                result = _run(b.process("show reflections"))

        assert result.tool_used == "reflection"
        assert "2025-07-14" in result.response

    def test_process_routes_run_reflection(self):
        b = self._make_brain()
        b._ensure_memory = MagicMock()
        b._ensure_graph = AsyncMock()
        b._to_en = AsyncMock(return_value="run reflection")
        b._handle_run_reflection = AsyncMock(return_value="🤔 Reflection done")

        with patch("bantz.core.brain.time_ctx") as mock_tc, \
             patch("bantz.core.brain.data_layer") as mock_dl:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()

            with patch("bantz.core.workflow.workflow_engine") as mock_wf:
                mock_wf.detect.return_value = []
                result = _run(b.process("run reflection"))

        assert result.tool_used == "reflection"
        b._handle_run_reflection.assert_awaited_once()


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
        assert self._qr("dinlemeyi durdur")["tool"] == "_wake_word_off"

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
        assert self._qr("dinlemeye başla")["tool"] == "_wake_word_on"

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
             patch("bantz.core.brain.data_layer") as mock_dl:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()

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
             patch("bantz.core.brain.data_layer") as mock_dl:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()

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
             patch("bantz.core.brain.data_layer") as mock_dl:
            mock_tc.snapshot.return_value = {"prompt_hint": ""}
            mock_dl.conversations = MagicMock()

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
        """Source code must reference notifier.send as fallback."""
        import inspect
        import bantz.core.brain as brain_mod
        src = inspect.getsource(brain_mod._notify_toast)
        assert "notifier.send" in src
        assert "notify-send" in src or "notifier" in src
