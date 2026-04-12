"""Tests for bantz.desktop.hyprctl — Hyprland IPC client (#365)."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bantz.desktop.hyprctl import (
    HyprctlClient,
    HyprctlResult,
    MonitorInfo,
    WindowInfo,
    WorkspaceInfo,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client() -> HyprctlClient:
    c = HyprctlClient()
    c._available = None  # reset cache
    return c


# ── Test data ─────────────────────────────────────────────────────────────────

MOCK_MONITORS = [
    {
        "name": "DP-1",
        "width": 2560,
        "height": 1440,
        "x": 0,
        "y": 0,
        "scale": 1.0,
        "focused": True,
        "description": "Dell S2722QC",
    }
]

MOCK_WORKSPACES = [
    {"id": 1, "name": "1", "monitor": "DP-1", "windows": 2, "lastwindowtitle": "Bantz"},
    {"id": 2, "name": "2", "monitor": "DP-1", "windows": 1, "lastwindowtitle": "Firefox"},
]

MOCK_CLIENTS = [
    {
        "address": "0x1234",
        "title": "Bantz Chat",
        "class": "bantz-chat",
        "workspace": {"id": 1, "name": "1"},
        "monitor": "DP-1",
        "at": [0, 0],
        "size": [1536, 1440],
        "floating": False,
        "focusHistoryID": 0,
        "pid": 12345,
    },
    {
        "address": "0x5678",
        "title": "Firefox",
        "class": "firefox",
        "workspace": {"id": 2, "name": "2"},
        "monitor": "DP-1",
        "at": [1536, 0],
        "size": [1024, 1440],
        "floating": False,
        "focusHistoryID": 1,
        "pid": 67890,
    },
]

MOCK_ACTIVE_WINDOW = {
    "address": "0x1234",
    "title": "Bantz Chat",
    "class": "bantz-chat",
    "workspace": {"id": 1, "name": "1"},
    "monitor": "DP-1",
    "at": [0, 0],
    "size": [1536, 1440],
    "floating": False,
    "pid": 12345,
}


# ── Availability ──────────────────────────────────────────────────────────────

class TestAvailability:
    @pytest.mark.asyncio
    async def test_not_available_when_no_binary(self, client: HyprctlClient) -> None:
        with patch("shutil.which", return_value=None):
            assert not await client.is_available()

    @pytest.mark.asyncio
    async def test_available_when_binary_and_running(self, client: HyprctlClient) -> None:
        with patch("shutil.which", return_value="/usr/bin/hyprctl"):
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b'{"version":"0.40"}', b""))
            mock_proc.returncode = 0
            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                assert await client.is_available()

    @pytest.mark.asyncio
    async def test_not_available_when_binary_but_not_running(self, client: HyprctlClient) -> None:
        with patch("shutil.which", return_value="/usr/bin/hyprctl"):
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
            mock_proc.returncode = 1
            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                assert not await client.is_available()

    @pytest.mark.asyncio
    async def test_caches_result(self, client: HyprctlClient) -> None:
        client._available = True
        assert await client.is_available()

    def test_reset_cache(self, client: HyprctlClient) -> None:
        client._available = True
        client.reset_cache()
        assert client._available is None


# ── Monitor parsing ───────────────────────────────────────────────────────────

class TestMonitors:
    @pytest.mark.asyncio
    async def test_parse_monitors(self, client: HyprctlClient) -> None:
        with patch.object(client, "_run_json", return_value=MOCK_MONITORS):
            monitors = await client.get_monitors()
            assert len(monitors) == 1
            assert monitors[0].name == "DP-1"
            assert monitors[0].width == 2560
            assert monitors[0].height == 1440
            assert monitors[0].focused is True

    @pytest.mark.asyncio
    async def test_empty_when_error(self, client: HyprctlClient) -> None:
        with patch.object(client, "_run_json", return_value=[]):
            monitors = await client.get_monitors()
            assert monitors == []

    @pytest.mark.asyncio
    async def test_handles_dict_response(self, client: HyprctlClient) -> None:
        with patch.object(client, "_run_json", return_value={}):
            monitors = await client.get_monitors()
            assert monitors == []


# ── Workspace parsing ─────────────────────────────────────────────────────────

class TestWorkspaces:
    @pytest.mark.asyncio
    async def test_parse_workspaces(self, client: HyprctlClient) -> None:
        with patch.object(client, "_run_json", return_value=MOCK_WORKSPACES):
            ws = await client.get_workspaces()
            assert len(ws) == 2
            assert ws[0].id == 1
            assert ws[0].name == "1"
            assert ws[0].windows == 2
            assert ws[1].last_window_title == "Firefox"


# ── Client/window parsing ────────────────────────────────────────────────────

class TestClients:
    @pytest.mark.asyncio
    async def test_parse_clients(self, client: HyprctlClient) -> None:
        with patch.object(client, "_run_json", return_value=MOCK_CLIENTS):
            clients = await client.get_clients()
            assert len(clients) == 2
            bantz = clients[0]
            assert bantz.title == "Bantz Chat"
            assert bantz.class_name == "bantz-chat"
            assert bantz.workspace_id == 1
            assert bantz.width == 1536
            assert bantz.focused is True
            assert bantz.pid == 12345

    @pytest.mark.asyncio
    async def test_firefox_client(self, client: HyprctlClient) -> None:
        with patch.object(client, "_run_json", return_value=MOCK_CLIENTS):
            clients = await client.get_clients()
            ff = clients[1]
            assert ff.class_name == "firefox"
            assert ff.focused is False


# ── Active window ─────────────────────────────────────────────────────────────

class TestActiveWindow:
    @pytest.mark.asyncio
    async def test_get_active(self, client: HyprctlClient) -> None:
        with patch.object(client, "_run_json", return_value=MOCK_ACTIVE_WINDOW):
            active = await client.get_active_window()
            assert active is not None
            assert active.title == "Bantz Chat"
            assert active.focused is True

    @pytest.mark.asyncio
    async def test_no_active_window(self, client: HyprctlClient) -> None:
        with patch.object(client, "_run_json", return_value={}):
            active = await client.get_active_window()
            assert active is None


# ── Dispatchers ───────────────────────────────────────────────────────────────

class TestDispatchers:
    @pytest.mark.asyncio
    async def test_move_to_workspace(self, client: HyprctlClient) -> None:
        with patch.object(client, "_dispatch", return_value=HyprctlResult(success=True, output="ok")):
            result = await client.move_to_workspace(2)
            assert result.success

    @pytest.mark.asyncio
    async def test_focus_window(self, client: HyprctlClient) -> None:
        with patch.object(client, "_dispatch", return_value=HyprctlResult(success=True)):
            result = await client.focus_window("0x1234")
            assert result.success

    @pytest.mark.asyncio
    async def test_resize_window(self, client: HyprctlClient) -> None:
        with patch.object(client, "_dispatch", return_value=HyprctlResult(success=True)):
            result = await client.resize_window("0x1234", 1920, 1080)
            assert result.success

    @pytest.mark.asyncio
    async def test_move_window(self, client: HyprctlClient) -> None:
        with patch.object(client, "_dispatch", return_value=HyprctlResult(success=True)):
            result = await client.move_window("0x1234", 100, 200)
            assert result.success

    @pytest.mark.asyncio
    async def test_toggle_floating(self, client: HyprctlClient) -> None:
        with patch.object(client, "_dispatch", return_value=HyprctlResult(success=True)):
            result = await client.toggle_floating("0x1234")
            assert result.success

    @pytest.mark.asyncio
    async def test_fullscreen(self, client: HyprctlClient) -> None:
        with patch.object(client, "_dispatch", return_value=HyprctlResult(success=True)):
            result = await client.fullscreen(1)
            assert result.success

    @pytest.mark.asyncio
    async def test_close_window(self, client: HyprctlClient) -> None:
        with patch.object(client, "_dispatch", return_value=HyprctlResult(success=True)):
            result = await client.close_window("0x1234")
            assert result.success

    @pytest.mark.asyncio
    async def test_exec_once(self, client: HyprctlClient) -> None:
        with patch.object(client, "_dispatch", return_value=HyprctlResult(success=True)):
            result = await client.exec_once("kitty")
            assert result.success


# ── Window rules ──────────────────────────────────────────────────────────────

class TestWindowRules:
    @pytest.mark.asyncio
    async def test_set_window_rule(self, client: HyprctlClient) -> None:
        with patch.object(client, "_keyword", return_value=HyprctlResult(success=True)):
            result = await client.set_window_rule("float, class:^(mpv)$")
            assert result.success


# ── Layout helper ─────────────────────────────────────────────────────────────

class TestBantzLayout:
    @pytest.mark.asyncio
    async def test_apply_layout_success(self, client: HyprctlClient) -> None:
        with patch.object(client, "get_monitors", return_value=[
            MonitorInfo(name="DP-1", width=2560, height=1440, focused=True),
        ]):
            with patch.object(client, "get_clients", return_value=[
                WindowInfo(
                    address="0x1234", title="Bantz", class_name="bantz-chat",
                    workspace_id=1, width=1000, height=1000,
                ),
            ]):
                with patch.object(client, "resize_window", return_value=HyprctlResult(success=True)):
                    with patch.object(client, "move_window", return_value=HyprctlResult(success=True)):
                        ok = await client.apply_bantz_layout()
                        assert ok is True

    @pytest.mark.asyncio
    async def test_apply_layout_no_bantz_window(self, client: HyprctlClient) -> None:
        with patch.object(client, "get_monitors", return_value=[
            MonitorInfo(name="DP-1", width=2560, height=1440),
        ]):
            with patch.object(client, "get_clients", return_value=[
                WindowInfo(address="0x5678", title="Firefox", class_name="firefox"),
            ]):
                ok = await client.apply_bantz_layout()
                assert ok is False

    @pytest.mark.asyncio
    async def test_apply_layout_no_monitors(self, client: HyprctlClient) -> None:
        with patch.object(client, "get_monitors", return_value=[]):
            ok = await client.apply_bantz_layout()
            assert ok is False


# ── Status summary ────────────────────────────────────────────────────────────

class TestStatusSummary:
    @pytest.mark.asyncio
    async def test_status_when_running(self, client: HyprctlClient) -> None:
        client._available = True
        with patch.object(client, "get_monitors", return_value=[
            MonitorInfo(name="DP-1", width=2560, height=1440),
        ]):
            with patch.object(client, "get_workspaces", return_value=[
                WorkspaceInfo(id=1, name="1"),
                WorkspaceInfo(id=2, name="2"),
            ]):
                with patch.object(client, "get_active_window", return_value=
                    WindowInfo(address="0x1234", title="Bantz", class_name="bantz-chat", focused=True)):
                    status = await client.status_summary()
                    assert status["running"] is True
                    assert status["monitors"] == 1
                    assert status["workspaces"] == 2
                    assert status["active_window"] == "Bantz"

    @pytest.mark.asyncio
    async def test_status_when_not_running(self, client: HyprctlClient) -> None:
        client._available = False
        status = await client.status_summary()
        assert status == {"running": False}


# ── Raw execution error handling ──────────────────────────────────────────────

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_run_json_timeout(self, client: HyprctlClient) -> None:
        async def timeout_proc(*args, **kwargs):
            mock = AsyncMock()
            mock.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
            return mock
        with patch("asyncio.create_subprocess_exec", side_effect=timeout_proc):
            result = await client._run_json("monitors")
            assert result == []

    @pytest.mark.asyncio
    async def test_run_json_os_error(self, client: HyprctlClient) -> None:
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await client._run_json("monitors")
            assert result == []

    @pytest.mark.asyncio
    async def test_dispatch_timeout(self, client: HyprctlClient) -> None:
        async def timeout_proc(*args, **kwargs):
            mock = AsyncMock()
            mock.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
            return mock
        with patch("asyncio.create_subprocess_exec", side_effect=timeout_proc):
            result = await client._dispatch("workspace", "1")
            assert result.success is False
            assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_dispatch_os_error(self, client: HyprctlClient) -> None:
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("no such file")):
            result = await client._dispatch("workspace", "1")
            assert result.success is False

    @pytest.mark.asyncio
    async def test_keyword_timeout(self, client: HyprctlClient) -> None:
        with patch("asyncio.create_subprocess_exec", side_effect=asyncio.TimeoutError):
            result = await client._keyword("key", "value")
            assert result.success is False
