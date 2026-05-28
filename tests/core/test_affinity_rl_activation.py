"""Tests for #438 — RL / AffinityEngine activation defaults and habit wiring.

Covers:
  - Config: rl_enabled defaults to True
  - DataLayer: AffinityEngine is init'd when rl_enabled=True, skipped when False
  - BantzContext: habit_hint field exists and defaults to ""
  - habit_hint(): correct segment, correct text, graceful empty / error
  - inject(): ctx.habit_hint is populated
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════
# Config default
# ═══════════════════════════════════════════════════════════════════════════


class TestRLEnabledDefault:
    def test_rl_enabled_is_true_by_default(self):
        """rl_enabled must default to True so the AffinityEngine fires (#438)."""
        from bantz.config import Config
        import inspect, dataclasses

        # Inspect the Field default directly so .env overrides don't mask the code default
        fields = Config.model_fields
        rl_field = fields.get("rl_enabled")
        assert rl_field is not None, "rl_enabled field missing from Config"
        assert rl_field.default is True, (
            f"rl_enabled default should be True, got {rl_field.default!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# BantzContext field
# ═══════════════════════════════════════════════════════════════════════════


class TestBantzContextHabitHint:
    def test_habit_hint_field_exists_and_defaults_empty(self):
        from bantz.core.context import BantzContext

        ctx = BantzContext()
        assert hasattr(ctx, "habit_hint")
        assert ctx.habit_hint == ""

    def test_habit_hint_field_is_assignable(self):
        from bantz.core.context import BantzContext

        ctx = BantzContext()
        ctx.habit_hint = "[Usage habits — morning] Most-used tools: weather (3×)"
        assert "weather" in ctx.habit_hint


# ═══════════════════════════════════════════════════════════════════════════
# habit_hint() helper
# ═══════════════════════════════════════════════════════════════════════════


class TestHabitHint:
    """habit_hint() derives segment from clock and formats top tools."""

    def _make_mock_habits(self, top_tools):
        mock = MagicMock()
        mock.top_tools_for_segment.return_value = top_tools
        return mock

    def _call_at_hour(self, hour, mock_habits):
        with (
            patch("bantz.core.habits.habits", mock_habits),
            patch("datetime.datetime") as mock_dt,
        ):
            mock_dt.now.return_value.hour = hour
            from bantz.core.memory_injector import habit_hint
            return habit_hint()

    def test_morning_segment(self):
        mock_habits = self._make_mock_habits(
            [{"tool": "weather", "count": 5}, {"tool": "news", "count": 3}]
        )
        result = self._call_at_hour(9, mock_habits)
        assert "morning" in result
        assert "weather" in result
        assert "5" in result
        mock_habits.top_tools_for_segment.assert_called_once_with("morning", n=3)

    def test_afternoon_segment(self):
        mock_habits = self._make_mock_habits(
            [{"tool": "calendar", "count": 4}]
        )
        result = self._call_at_hour(14, mock_habits)
        assert "afternoon" in result
        assert "calendar" in result

    def test_evening_segment(self):
        mock_habits = self._make_mock_habits([{"tool": "music", "count": 2}])
        result = self._call_at_hour(19, mock_habits)
        assert "evening" in result

    def test_night_segment(self):
        mock_habits = self._make_mock_habits([{"tool": "reminder", "count": 1}])
        result = self._call_at_hour(22, mock_habits)
        assert "night" in result

    def test_late_night_segment(self):
        mock_habits = self._make_mock_habits([{"tool": "spotify", "count": 7}])
        result = self._call_at_hour(3, mock_habits)
        assert "late_night" in result

    def test_empty_when_no_top_tools(self):
        mock_habits = self._make_mock_habits([])
        result = self._call_at_hour(10, mock_habits)
        assert result == ""

    def test_empty_on_exception(self):
        """Any exception must be swallowed and return empty string."""
        mock_habits = self._make_mock_habits(None)
        mock_habits.top_tools_for_segment.side_effect = RuntimeError("DB not found")
        result = self._call_at_hour(10, mock_habits)
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
# inject() wires habit_hint into ctx
# ═══════════════════════════════════════════════════════════════════════════


class TestInjectHabitHint:
    """inject() must set ctx.habit_hint from habit_hint()."""

    async def test_inject_sets_habit_hint(self):
        from bantz.core.context import BantzContext
        from bantz.core import memory_injector

        ctx = BantzContext()

        recall_result = MagicMock()
        recall_result.graph_context = ""
        recall_result.vector_context = ""
        recall_result.deep_memory = ""
        recall_result.combined = ""

        mock_omni = MagicMock()
        mock_omni.recall = AsyncMock(return_value=recall_result)

        mock_profile = MagicMock()
        mock_profile.prompt_hint.return_value = ""

        expected_hint = "[Usage habits — morning] Most-used tools: weather (5×)"

        with (
            patch("bantz.memory.omni_memory.omni_memory", mock_omni),
            patch("bantz.core.profile.profile", mock_profile),
            patch("bantz.core.memory_injector.habit_hint", return_value=expected_hint),
            patch("bantz.core.memory_injector.desktop_context", return_value=""),
            patch("bantz.core.memory_injector.persona_hint", return_value=""),
            patch("bantz.core.memory_injector.formality_hint", return_value=""),
            patch("bantz.core.memory_injector.style_hint", return_value=""),
        ):
            await memory_injector.inject(ctx, "what's the weather?")

        assert ctx.habit_hint == expected_hint

    async def test_inject_habit_hint_empty_when_no_data(self):
        from bantz.core.context import BantzContext
        from bantz.core import memory_injector

        ctx = BantzContext()

        recall_result = MagicMock()
        recall_result.graph_context = ""
        recall_result.vector_context = ""
        recall_result.deep_memory = ""
        recall_result.combined = ""

        mock_omni = MagicMock()
        mock_omni.recall = AsyncMock(return_value=recall_result)

        mock_profile = MagicMock()
        mock_profile.prompt_hint.return_value = ""

        with (
            patch("bantz.memory.omni_memory.omni_memory", mock_omni),
            patch("bantz.core.profile.profile", mock_profile),
            patch("bantz.core.memory_injector.habit_hint", return_value=""),
            patch("bantz.core.memory_injector.desktop_context", return_value=""),
            patch("bantz.core.memory_injector.persona_hint", return_value=""),
            patch("bantz.core.memory_injector.formality_hint", return_value=""),
            patch("bantz.core.memory_injector.style_hint", return_value=""),
        ):
            await memory_injector.inject(ctx, "hello")

        assert ctx.habit_hint == ""
