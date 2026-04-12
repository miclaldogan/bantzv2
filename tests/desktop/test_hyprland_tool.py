"""Tests for bantz.tools.hyprland_tool — Hyprland control tool (#365)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from bantz.tools.hyprland_tool import HyprlandTool
from bantz.tools import ToolResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tool() -> HyprlandTool:
    return HyprlandTool()


# ── Registration ──────────────────────────────────────────────────────────────

class TestRegistration:
    def test_name(self, tool: HyprlandTool) -> None:
        assert tool.name == "hyprland"

    def test_risk_level(self, tool: HyprlandTool) -> None:
        assert tool.risk_level == "moderate"

    def test_description_nonempty(self, tool: HyprlandTool) -> None:
        assert len(tool.description) > 20

    def test_registered_in_registry(self) -> None:
        from bantz.tools import registry
        assert registry.get("hyprland") is not None


# ── Unknown action ────────────────────────────────────────────────────────────

class TestUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action(self, tool: HyprlandTool) -> None:
        result = await tool.execute(action="bogus")
        assert not result.success
        assert "Unknown action" in result.output

    @pytest.mark.asyncio
    async def test_default_action_is_status(self, tool: HyprlandTool) -> None:
        with patch("bantz.desktop.hyprctl.HyprctlClient.status_summary",
                    return_value={"running": False}):
            result = await tool.execute()
            assert result.success
            assert "not running" in result.output


# ── Status action ─────────────────────────────────────────────────────────────

class TestStatusAction:
    @pytest.mark.asyncio
    async def test_not_running(self, tool: HyprlandTool) -> None:
        with patch("bantz.desktop.hyprctl.HyprctlClient.status_summary",
                    return_value={"running": False}):
            result = await tool.execute(action="status")
            assert result.success
            assert "not running" in result.output

    @pytest.mark.asyncio
    async def test_running_with_details(self, tool: HyprlandTool) -> None:
        with patch("bantz.desktop.hyprctl.HyprctlClient.status_summary",
                    return_value={
                        "running": True,
                        "monitors": 2,
                        "workspaces": 3,
                        "active_window": "Firefox",
                        "active_class": "firefox",
                    }):
            result = await tool.execute(action="status")
            assert result.success
            assert "Monitors: 2" in result.output
            assert "Firefox" in result.output


# ── Windows action ────────────────────────────────────────────────────────────

class TestWindowsAction:
    @pytest.mark.asyncio
    async def test_no_windows(self, tool: HyprlandTool) -> None:
        with patch("bantz.desktop.hyprctl.HyprctlClient.get_clients", return_value=[]):
            result = await tool.execute(action="windows")
            assert result.success
            assert "No windows" in result.output

    @pytest.mark.asyncio
    async def test_lists_windows(self, tool: HyprlandTool) -> None:
        from bantz.desktop.hyprctl import WindowInfo
        mock_clients = [
            WindowInfo(
                address="0x1", title="Bantz Chat", class_name="bantz-chat",
                workspace_id=1, width=1536, height=1440, focused=True,
            ),
            WindowInfo(
                address="0x2", title="Firefox", class_name="firefox",
                workspace_id=2, width=1024, height=1440,
            ),
        ]
        with patch("bantz.desktop.hyprctl.HyprctlClient.get_clients", return_value=mock_clients):
            result = await tool.execute(action="windows")
            assert result.success
            assert "Bantz Chat" in result.output
            assert "Firefox" in result.output
            assert "◄" in result.output  # focused marker


# ── Focus action ──────────────────────────────────────────────────────────────

class TestFocusAction:
    @pytest.mark.asyncio
    async def test_focus_no_target(self, tool: HyprlandTool) -> None:
        result = await tool.execute(action="focus")
        assert not result.success

    @pytest.mark.asyncio
    async def test_focus_not_found(self, tool: HyprlandTool) -> None:
        with patch("bantz.desktop.hyprctl.HyprctlClient.get_clients", return_value=[]):
            result = await tool.execute(action="focus", title="nonexistent")
            assert not result.success
            assert "No window" in result.output

    @pytest.mark.asyncio
    async def test_focus_success(self, tool: HyprlandTool) -> None:
        from bantz.desktop.hyprctl import WindowInfo, HyprctlResult
        mock_clients = [
            WindowInfo(address="0x1", title="Firefox", class_name="firefox"),
        ]
        with patch("bantz.desktop.hyprctl.HyprctlClient.get_clients", return_value=mock_clients):
            with patch("bantz.desktop.hyprctl.HyprctlClient.focus_window",
                        return_value=HyprctlResult(success=True)):
                result = await tool.execute(action="focus", title="firefox")
                assert result.success
                assert "Focused" in result.output


# ── Move action ───────────────────────────────────────────────────────────────

class TestMoveAction:
    @pytest.mark.asyncio
    async def test_move_no_params(self, tool: HyprlandTool) -> None:
        result = await tool.execute(action="move")
        assert not result.success

    @pytest.mark.asyncio
    async def test_move_success(self, tool: HyprlandTool) -> None:
        from bantz.desktop.hyprctl import WindowInfo, HyprctlResult
        mock_clients = [
            WindowInfo(address="0x1", title="Firefox", class_name="firefox"),
        ]
        with patch("bantz.desktop.hyprctl.HyprctlClient.get_clients", return_value=mock_clients):
            with patch("bantz.desktop.hyprctl.HyprctlClient.move_window_to_workspace",
                        return_value=HyprctlResult(success=True)):
                result = await tool.execute(action="move", target="firefox", workspace=3)
                assert result.success
                assert "workspace 3" in result.output


# ── Layout action ─────────────────────────────────────────────────────────────

class TestLayoutAction:
    @pytest.mark.asyncio
    async def test_layout_success(self, tool: HyprlandTool) -> None:
        with patch("bantz.desktop.hyprctl.HyprctlClient.apply_bantz_layout", return_value=True):
            result = await tool.execute(action="layout")
            assert result.success
            assert "60/40" in result.output

    @pytest.mark.asyncio
    async def test_layout_failure(self, tool: HyprlandTool) -> None:
        with patch("bantz.desktop.hyprctl.HyprctlClient.apply_bantz_layout", return_value=False):
            result = await tool.execute(action="layout")
            assert not result.success


# ── Exec action ───────────────────────────────────────────────────────────────

class TestExecAction:
    @pytest.mark.asyncio
    async def test_exec_no_command(self, tool: HyprlandTool) -> None:
        result = await tool.execute(action="exec")
        assert not result.success

    @pytest.mark.asyncio
    async def test_exec_success(self, tool: HyprlandTool) -> None:
        from bantz.desktop.hyprctl import HyprctlResult
        with patch("bantz.desktop.hyprctl.HyprctlClient.exec_once",
                    return_value=HyprctlResult(success=True)):
            result = await tool.execute(action="exec", command="kitty")
            assert result.success
            assert "kitty" in result.output


# ── Widgets action ────────────────────────────────────────────────────────────

class TestWidgetsAction:
    @pytest.mark.asyncio
    async def test_default_widget(self, tool: HyprlandTool) -> None:
        with patch("bantz.desktop.widgets.WidgetDataProvider.get", return_value='{"text": "active"}'):
            result = await tool.execute(action="widgets")
            assert result.success

    @pytest.mark.asyncio
    async def test_specific_widget(self, tool: HyprlandTool) -> None:
        with patch("bantz.desktop.widgets.WidgetDataProvider.get", return_value="42"):
            result = await tool.execute(action="widgets", widget="cpu")
            assert result.success
            assert "42" in result.output


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_exception_returns_error(self, tool: HyprlandTool) -> None:
        with patch("bantz.desktop.hyprctl.HyprctlClient.status_summary",
                    side_effect=RuntimeError("connection lost")):
            result = await tool.execute(action="status")
            assert not result.success
            assert "connection lost" in result.error
