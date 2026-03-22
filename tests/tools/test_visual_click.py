"""Tests for Issue #185 - VisualClickTool (The Butler's Eyes).

Covers:
  1. Successful visual click (navigate found + click executed)
  2. Navigation failure (element not found) - Butler lore response
  3. Action mapping (click, double_click, right_click, hover)
  4. Input control disabled gate
  5. Missing target parameter
  6. Unknown action rejection
  7. Input action failure (exception during click)
  8. Quick route regex matching
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — mock NavResult
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MockNavResult:
    """Lightweight NavResult stand-in for tests."""
    found: bool
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    method: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    app_name: str = ""
    element_label: str = ""
    role: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        if self.width and self.height:
            return (self.x + self.width // 2, self.y + self.height // 2)
        return (self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "x": self.x, "y": self.y,
            "method": self.method,
            "confidence": self.confidence,
        }


def _nav_found(x=100, y=200, method="vlm", confidence=0.9):
    return MockNavResult(
        found=True, x=x, y=y, width=0, height=0,
        method=method, confidence=confidence, latency_ms=42.0,
    )


def _nav_not_found():
    return MockNavResult(found=False, method="none", latency_ms=1200.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Successful visual click
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualClickSuccess:
    """Happy path: element found + action executed."""

    @pytest.mark.asyncio
    async def test_click_success(self):
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()
        mock_click = AsyncMock()

        with patch("bantz.tools.visual_click.config") as mock_config, \
             patch("bantz.vision.navigator.navigator") as mock_nav:
            mock_config.input_control_enabled = True
            mock_nav.navigate_to = AsyncMock(return_value=_nav_found())

            import bantz.tools.input_control as ic
            original_click = getattr(ic, 'click', None)
            ic.click = mock_click
            try:
                result = await tool.execute(target="Send button", action="click")
            finally:
                if original_click:
                    ic.click = original_click

        assert result.success is True
        assert "clicked" in result.output.lower()
        assert "Send button" in result.output
        assert "vlm" in result.output.lower()
        mock_click.assert_called_once_with(100, 200)

    @pytest.mark.asyncio
    async def test_hover_uses_move_to(self):
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()
        mock_move = AsyncMock()

        with patch("bantz.tools.visual_click.config") as mock_config, \
             patch("bantz.vision.navigator.navigator") as mock_nav:
            mock_config.input_control_enabled = True
            mock_nav.navigate_to = AsyncMock(return_value=_nav_found(x=300, y=400))

            import bantz.tools.input_control as ic
            original = getattr(ic, 'move_to', None)
            ic.move_to = mock_move
            try:
                result = await tool.execute(target="Settings icon", action="hover")
            finally:
                if original:
                    ic.move_to = original

        assert result.success is True
        assert "hovered over" in result.output.lower()
        mock_move.assert_called_once_with(300, 400)

    @pytest.mark.asyncio
    async def test_double_click(self):
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()
        mock_dbl = AsyncMock()

        with patch("bantz.tools.visual_click.config") as mock_config, \
             patch("bantz.vision.navigator.navigator") as mock_nav:
            mock_config.input_control_enabled = True
            mock_nav.navigate_to = AsyncMock(return_value=_nav_found())

            import bantz.tools.input_control as ic
            original = getattr(ic, 'double_click', None)
            ic.double_click = mock_dbl
            try:
                result = await tool.execute(target="file.txt", action="double_click")
            finally:
                if original:
                    ic.double_click = original

        assert result.success is True
        assert "double-clicked" in result.output.lower()

    @pytest.mark.asyncio
    async def test_right_click(self):
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()
        mock_rc = AsyncMock()

        with patch("bantz.tools.visual_click.config") as mock_config, \
             patch("bantz.vision.navigator.navigator") as mock_nav:
            mock_config.input_control_enabled = True
            mock_nav.navigate_to = AsyncMock(return_value=_nav_found())

            import bantz.tools.input_control as ic
            original = getattr(ic, 'right_click', None)
            ic.right_click = mock_rc
            try:
                result = await tool.execute(target="desktop", action="right_click")
            finally:
                if original:
                    ic.right_click = original

        assert result.success is True
        assert "right-clicked" in result.output.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Navigation failure — Butler lore
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualClickFailure:
    """Element not found — graceful Butler response."""

    @pytest.mark.asyncio
    async def test_not_found_returns_butler_lore(self):
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()

        with patch("bantz.tools.visual_click.config") as mock_config, \
             patch("bantz.vision.navigator.navigator") as mock_nav:
            mock_config.input_control_enabled = True
            mock_nav.navigate_to = AsyncMock(return_value=_nav_not_found())

            result = await tool.execute(target="Invisible Button", action="click")

        assert result.success is False
        assert "could not find" in result.error.lower()
        assert "Invisible Button" in result.error

    @pytest.mark.asyncio
    async def test_not_found_includes_nav_data(self):
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()

        with patch("bantz.tools.visual_click.config") as mock_config, \
             patch("bantz.vision.navigator.navigator") as mock_nav:
            mock_config.input_control_enabled = True
            mock_nav.navigate_to = AsyncMock(return_value=_nav_not_found())

            result = await tool.execute(target="Ghost", action="click")

        assert result.data["found"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Gate checks
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualClickGates:
    """Parameter validation and config gates."""

    @pytest.mark.asyncio
    async def test_input_control_disabled(self):
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()

        with patch("bantz.tools.visual_click.config") as mock_config:
            mock_config.input_control_enabled = False
            result = await tool.execute(target="Button", action="click")

        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_target(self):
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()

        with patch("bantz.tools.visual_click.config") as mock_config:
            mock_config.input_control_enabled = True
            result = await tool.execute(target="", action="click")

        assert result.success is False
        assert "no target" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()

        with patch("bantz.tools.visual_click.config") as mock_config:
            mock_config.input_control_enabled = True
            result = await tool.execute(target="Button", action="smash")

        assert result.success is False
        assert "unknown action" in result.error.lower()
        assert "smash" in result.error


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Input action failure
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualClickActionError:
    """Input control raises an exception after navigation succeeds."""

    @pytest.mark.asyncio
    async def test_action_exception_caught(self):
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()

        with patch("bantz.tools.visual_click.config") as mock_config, \
             patch("bantz.vision.navigator.navigator") as mock_nav:
            mock_config.input_control_enabled = True
            mock_nav.navigate_to = AsyncMock(return_value=_nav_found())

            import bantz.tools.input_control as ic
            original = getattr(ic, 'click', None)
            ic.click = AsyncMock(side_effect=OSError("Display not available"))
            try:
                result = await tool.execute(target="OK", action="click")
            finally:
                if original:
                    ic.click = original

        assert result.success is False
        assert "hand slipped" in result.output.lower()
        assert "Display not available" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# 4b. Navigator pipeline crash
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualClickNavigatorError:
    """Navigator itself blows up (VLM unreachable, screenshot fail, etc.)."""

    @pytest.mark.asyncio
    async def test_navigator_exception_caught(self):
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()

        with patch("bantz.tools.visual_click.config") as mock_config, \
             patch("bantz.vision.navigator.navigator") as mock_nav:
            mock_config.input_control_enabled = True
            mock_nav.navigate_to = AsyncMock(
                side_effect=ConnectionError("VLM server unreachable"),
            )

            result = await tool.execute(target="Send", action="click")

        assert result.success is False
        assert "vision system" in result.output.lower()
        assert "VLM server unreachable" in result.output
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_navigator_import_error_caught(self):
        """Even if the navigator module fails to import, tool reports cleanly."""
        from bantz.tools.visual_click import VisualClickTool

        tool = VisualClickTool()

        with patch("bantz.tools.visual_click.config") as mock_config, \
             patch.dict("sys.modules", {"bantz.vision.navigator": None}):
            mock_config.input_control_enabled = True
            result = await tool.execute(target="OK", action="click")

        assert result.success is False
        assert "error" in result.error.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Tool registration
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualClickRegistration:
    """Tool must be registered in the global registry."""

    def test_tool_is_registered(self):
        from bantz.tools.visual_click import registry
        tool = registry.get("visual_click")
        assert tool is not None
        assert tool.name == "visual_click"
        assert tool.risk_level == "moderate"

    def test_schema_has_description(self):
        from bantz.tools.visual_click import registry
        tool = registry.get("visual_click")
        schema = tool.schema()
        assert "click" in schema["description"].lower() or "locate" in schema["description"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Quick route regex matching (#185)
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualClickNoQuickRoute:
    """visual_click must NOT be quick-routed — LLM picks it via tool-calling."""

    def _route(self, text: str) -> dict | None:
        from bantz.core.routing_engine import quick_route
        with patch("bantz.core.routing_engine.config") as mock_config:
            mock_config.input_control_enabled = True
            return quick_route(text, text)

    def test_click_not_quick_routed(self):
        r = self._route("click the send button")
        assert r is None or r["tool"] != "visual_click"

    def test_hover_not_quick_routed(self):
        r = self._route("hover the settings icon")
        assert r is None or r["tool"] != "visual_click"

    def test_double_click_not_quick_routed(self):
        r = self._route("double click the file.txt")
        assert r is None or r["tool"] != "visual_click"

    def test_right_click_not_quick_routed(self):
        r = self._route("right click on desktop")
        assert r is None or r["tool"] != "visual_click"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Anti-prompt & few-shot guards (#185)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAntiPromptGuards:
    """Shell tool warns against UI use; visual_click has a few-shot example."""

    def test_shell_description_warns_against_clicking(self):
        from bantz.tools.shell import ShellTool
        desc = ShellTool.description.lower()
        assert "do not use this tool" in desc and ("click" in desc or "gui" in desc)

    def test_shell_description_points_to_visual_click(self):
        from bantz.tools.shell import ShellTool
        assert "visual_click" in ShellTool.description

    def test_visual_click_has_few_shot_example(self):
        from bantz.tools.visual_click import VisualClickTool
        desc = VisualClickTool.description.lower()
        assert "example" in desc
        assert "click the terminal" in desc

    def test_authorization_has_few_shot_example(self):
        from bantz.core.prompt_builder import COMPUTER_USE_AUTHORIZATION
        assert "EXAMPLE" in COMPUTER_USE_AUTHORIZATION
        assert "visual_click" in COMPUTER_USE_AUTHORIZATION
        assert "WRONG action" in COMPUTER_USE_AUTHORIZATION
