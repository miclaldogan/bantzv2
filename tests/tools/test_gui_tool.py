"""Tests for GUITool (#292)."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from bantz.tools.gui_tool import (
    CACHE_DIR,
    GUITool,
    GUIToolError,
    gui_tool,
    _get_pyautogui,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tool():
    """Fresh GUITool with empty action log."""
    t = GUITool()
    return t


@pytest.fixture
def dry(monkeypatch):
    """Enable DRY_RUN mode."""
    monkeypatch.setenv("BANTZ_DRY_RUN", "1")


@pytest.fixture
def no_dry(monkeypatch):
    """Ensure DRY_RUN is off."""
    monkeypatch.setenv("BANTZ_DRY_RUN", "0")


@pytest.fixture
def fake_template(tmp_path):
    """Create a fake template image on disk."""
    p = tmp_path / "button.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    return str(p)


# ── Tests: _dry_run property ─────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_off_by_default(self, tool, no_dry):
        assert tool._dry_run is False

    def test_dry_run_on(self, tool, dry):
        assert tool._dry_run is True

    def test_dry_run_checked_at_call_time(self, tool, monkeypatch):
        """Verify that _dry_run is NOT cached at import time."""
        monkeypatch.setenv("BANTZ_DRY_RUN", "0")
        assert tool._dry_run is False
        monkeypatch.setenv("BANTZ_DRY_RUN", "1")
        assert tool._dry_run is True


# ── Tests: click ──────────────────────────────────────────────────────────────

class TestClick:
    @patch("bantz.tools.gui_tool._get_pyautogui")
    @patch("bantz.tools.gui_tool.time")
    def test_click_calls_pyautogui(self, mock_time, mock_get_pag, tool, no_dry):
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag
        tool.click(100, 200)
        mock_time.sleep.assert_called_once_with(0.3)
        mock_get_pag.click.assert_called_once_with(100, 200)

    @patch("bantz.tools.gui_tool._get_pyautogui")
    def test_click_dry_run(self, mock_get_pag, tool, dry):
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag
        tool.click(100, 200)
        mock_get_pag.click.assert_not_called()

    def test_click_logs_action(self, tool, dry):
        tool.click(50, 60)
        log = tool.get_action_log()
        assert len(log) == 1
        assert log[0]["action"] == "click"
        assert log[0]["x"] == 50
        assert log[0]["y"] == 60
        assert "ts" in log[0]


# ── Tests: click_image ────────────────────────────────────────────────────────

class TestClickImage:
    def test_missing_template_raises(self, tool, no_dry):
        with pytest.raises(GUIToolError, match="not found on disk"):
            tool.click_image("/nonexistent/image.png")

    def test_dry_run_returns_origin(self, tool, dry, fake_template):
        result = tool.click_image(fake_template)
        assert result == (0, 0)

    @patch("bantz.tools.gui_tool._get_pyautogui")
    def test_not_implemented_opencv(self, mock_get_pag, tool, no_dry:
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag, fake_template):
        mock_get_pag.locateOnScreen.side_effect = NotImplementedError
        with pytest.raises(GUIToolError, match="OpenCV is required"):
            tool.click_image(fake_template)

    @patch("bantz.tools.gui_tool._get_pyautogui")
    def test_image_not_found_exception(self, mock_get_pag, tool, no_dry:
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag, fake_template):
        mock_get_pag.ImageNotFoundException = type("ImageNotFoundException", (Exception,), {})
        mock_get_pag.locateOnScreen.side_effect = mock_get_pag.ImageNotFoundException
        with pytest.raises(GUIToolError, match="Image not found on screen"):
            tool.click_image(fake_template)

    @patch("bantz.tools.gui_tool._get_pyautogui")
    def test_image_not_found_none(self, mock_get_pag, tool, no_dry:
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag, fake_template):
        """Legacy pyautogui returns None instead of raising."""
        mock_get_pag.locateOnScreen.return_value = None
        mock_get_pag.ImageNotFoundException = type("ImageNotFoundException", (Exception,), {})
        with pytest.raises(GUIToolError, match="Image not found on screen"):
            tool.click_image(fake_template)

    @patch("bantz.tools.gui_tool._get_pyautogui")
    def test_click_image_success(self, mock_get_pag, tool, no_dry:
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag, fake_template):
        mock_get_pag.ImageNotFoundException = type("ImageNotFoundException", (Exception,), {})
        mock_loc = MagicMock()
        mock_get_pag.locateOnScreen.return_value = mock_loc
        mock_center = SimpleNamespace(x=150, y=250)
        mock_get_pag.center.return_value = mock_center

        result = tool.click_image(fake_template, confidence=0.8)

        mock_get_pag.locateOnScreen.assert_called_once_with(
            fake_template, confidence=0.8
        )
        mock_get_pag.center.assert_called_once_with(mock_loc)
        mock_get_pag.click.assert_called_once_with(mock_center)
        assert result == (150, 250)

    def test_click_image_logs_action(self, tool, dry, fake_template):
        tool.click_image(fake_template)
        log = tool.get_action_log()
        assert log[-1]["action"] == "click_image"
        assert log[-1]["template"] == fake_template


# ── Tests: type_text ──────────────────────────────────────────────────────────

class TestTypeText:
    @patch("bantz.tools.gui_tool._get_pyautogui")
    @patch("bantz.tools.gui_tool.time")
    def test_type_calls_typewrite(self, mock_time, mock_get_pag, tool, no_dry:
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag):
        tool.type_text("hello", interval=0.1)
        mock_time.sleep.assert_called_once_with(0.3)
        mock_get_pag.typewrite.assert_called_once_with("hello", interval=0.1)

    @patch("bantz.tools.gui_tool._get_pyautogui")
    def test_type_dry_run(self, mock_get_pag, tool, dry):
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag
        tool.type_text("secret")
        mock_get_pag.typewrite.assert_not_called()

    def test_type_logs_action(self, tool, dry):
        tool.type_text("hi")
        log = tool.get_action_log()
        assert log[-1]["action"] == "type"
        assert log[-1]["text"] == "hi"


# ── Tests: focus_window ──────────────────────────────────────────────────────

class TestFocusWindow:
    @patch("bantz.tools.gui_tool.subprocess")
    def test_focus_by_name(self, mock_sub, tool, no_dry):
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        tool.focus_window("Firefox")
        mock_sub.run.assert_called_once_with(
            ["xdotool", "search", "--name", "Firefox",
             "windowactivate", "--sync"],
            check=True, timeout=5,
        )

    @patch("bantz.tools.gui_tool.subprocess")
    def test_focus_by_class(self, mock_sub, tool, no_dry):
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        tool.focus_window("Firefox", wm_class="Navigator")
        mock_sub.run.assert_called_once_with(
            ["xdotool", "search", "--class", "Navigator",
             "windowactivate", "--sync"],
            check=True, timeout=5,
        )

    @patch("bantz.tools.gui_tool.subprocess")
    def test_focus_class_fallback_to_name(self, mock_sub, tool, no_dry):
        """When --class fails, falls back to --name."""
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        mock_sub.run.side_effect = [
            subprocess.CalledProcessError(1, "xdotool"),  # --class fails
            None,  # --name succeeds
        ]
        tool.focus_window("Firefox", wm_class="Navigator")
        assert mock_sub.run.call_count == 2

    @patch("bantz.tools.gui_tool.subprocess")
    def test_focus_class_and_name_fail(self, mock_sub, tool, no_dry):
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        mock_sub.run.side_effect = subprocess.CalledProcessError(1, "xdotool")
        with pytest.raises(GUIToolError, match="not found by class or name"):
            tool.focus_window("Firefox", wm_class="Navigator")

    @patch("bantz.tools.gui_tool.subprocess")
    def test_focus_name_only_fail(self, mock_sub, tool, no_dry):
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        mock_sub.run.side_effect = subprocess.CalledProcessError(1, "xdotool")
        with pytest.raises(GUIToolError, match="Window not found: 'Firefox'"):
            tool.focus_window("Firefox")

    @patch("bantz.tools.gui_tool.subprocess")
    def test_focus_xdotool_missing(self, mock_sub, tool, no_dry):
        """xdotool not installed → warning, no crash."""
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        mock_sub.run.side_effect = FileNotFoundError
        tool.focus_window("Firefox")  # should NOT raise

    @patch("bantz.tools.gui_tool.subprocess")
    def test_focus_timeout(self, mock_sub, tool, no_dry):
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        mock_sub.run.side_effect = subprocess.TimeoutExpired("xdotool", 5)
        with pytest.raises(GUIToolError, match="timed out"):
            tool.focus_window("Firefox")

    @patch("bantz.tools.gui_tool.subprocess")
    def test_focus_class_fail_name_timeout(self, mock_sub, tool, no_dry):
        """--class fails, --name times out."""
        mock_sub.CalledProcessError = subprocess.CalledProcessError
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        mock_sub.run.side_effect = [
            subprocess.CalledProcessError(1, "xdotool"),
            subprocess.TimeoutExpired("xdotool", 5),
        ]
        with pytest.raises(GUIToolError, match="timed out"):
            tool.focus_window("Firefox", wm_class="Navigator")

    def test_focus_dry_run(self, tool, dry):
        tool.focus_window("Firefox")  # should NOT raise or call subprocess

    def test_focus_logs_action(self, tool, dry):
        tool.focus_window("Firefox", wm_class="Nav")
        log = tool.get_action_log()
        assert log[-1]["action"] == "focus_window"
        assert log[-1]["wm_class"] == "Nav"


# ── Tests: screenshot ────────────────────────────────────────────────────────

class TestScreenshot:
    @patch("bantz.tools.gui_tool._get_pyautogui")
    def test_screenshot_full(self, mock_get_pag, tool, no_dry:
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag, tmp_path, monkeypatch):
        import bantz.tools.gui_tool as mod
        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)

        mock_img = MagicMock()
        mock_get_pag.screenshot.return_value = mock_img

        path = tool.screenshot()

        mock_get_pag.screenshot.assert_called_once_with(region=None)
        assert path.startswith(str(tmp_path))
        assert path.endswith(".png")
        mock_img.save.assert_called_once()

    @patch("bantz.tools.gui_tool._get_pyautogui")
    def test_screenshot_region(self, mock_get_pag, tool, no_dry:
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag, tmp_path, monkeypatch):
        import bantz.tools.gui_tool as mod
        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)

        mock_img = MagicMock()
        mock_get_pag.screenshot.return_value = mock_img
        region = (10, 20, 300, 400)

        path = tool.screenshot(region=region)

        mock_get_pag.screenshot.assert_called_once_with(region=region)

    def test_screenshot_dry_run(self, tool, dry):
        path = tool.screenshot()
        assert "dry_run_screenshot" in path

    def test_screenshot_logs_action(self, tool, dry):
        tool.screenshot(region=(1, 2, 3, 4))
        log = tool.get_action_log()
        assert log[-1]["action"] == "screenshot"
        assert log[-1]["region"] == (1, 2, 3, 4)


# ── Tests: scroll ────────────────────────────────────────────────────────────

class TestScroll:
    @patch("bantz.tools.gui_tool._get_pyautogui")
    @patch("bantz.tools.gui_tool.time")
    def test_scroll_calls_pyautogui(self, mock_time, mock_get_pag, tool, no_dry:
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag):
        tool.scroll(100, 200, 5)
        mock_time.sleep.assert_called_once_with(0.3)
        mock_get_pag.moveTo.assert_called_once_with(100, 200)
        mock_get_pag.scroll.assert_called_once_with(5)

    @patch("bantz.tools.gui_tool._get_pyautogui")
    def test_scroll_dry_run(self, mock_get_pag, tool, dry):
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag
        tool.scroll(100, 200, 5)
        mock_get_pag.moveTo.assert_not_called()
        mock_get_pag.scroll.assert_not_called()

    def test_scroll_logs_action(self, tool, dry):
        tool.scroll(10, 20, -3)
        log = tool.get_action_log()
        assert log[-1]["action"] == "scroll"
        assert log[-1]["clicks"] == -3


# ── Tests: action log ────────────────────────────────────────────────────────

class TestActionLog:
    def test_empty_log(self, tool):
        assert tool.get_action_log() == []

    def test_log_accumulates(self, tool, dry):
        tool.click(1, 2)
        tool.type_text("a")
        tool.scroll(3, 4, 1)
        assert len(tool.get_action_log()) == 3

    def test_clear_log(self, tool, dry):
        tool.click(1, 2)
        tool.click(3, 4)
        n = tool.clear_action_log()
        assert n == 2
        assert tool.get_action_log() == []

    def test_get_log_returns_copy(self, tool, dry):
        tool.click(1, 2)
        log1 = tool.get_action_log()
        log1.clear()
        assert len(tool.get_action_log()) == 1  # internal not affected


# ── Tests: execute (BaseTool interface) ───────────────────────────────────────

class TestExecute:
    @pytest.mark.asyncio
    @patch("bantz.tools.gui_tool._get_pyautogui")
    async def test_execute_click(self, mock_get_pag, tool, no_dry):
        mock_get_pag = MagicMock()
        mock_get_pag.return_value = mock_get_pag
        r = await tool.execute(action="click", x=10, y=20)
        assert r.success is True
        assert "10" in r.output and "20" in r.output

    @pytest.mark.asyncio
    async def test_execute_click_missing_param(self, tool, no_dry):
        r = await tool.execute(action="click")  # no x, y
        assert r.success is False
        assert "Missing required parameter" in r.error

    @pytest.mark.asyncio
    async def test_execute_type(self, tool, dry):
        r = await tool.execute(action="type", text="hello")
        assert r.success is True
        assert "5" in r.output  # 5 characters

    @pytest.mark.asyncio
    async def test_execute_focus_window(self, tool, dry):
        r = await tool.execute(action="focus_window", title="Firefox")
        assert r.success is True
        assert "Firefox" in r.output

    @pytest.mark.asyncio
    async def test_execute_screenshot_dry(self, tool, dry):
        r = await tool.execute(action="screenshot")
        assert r.success is True
        assert "path" in r.data

    @pytest.mark.asyncio
    async def test_execute_scroll(self, tool, dry):
        r = await tool.execute(action="scroll", x=50, y=60, clicks=3)
        assert r.success is True

    @pytest.mark.asyncio
    async def test_execute_action_log(self, tool, dry):
        await tool.execute(action="click", x=1, y=2)
        r = await tool.execute(action="action_log")
        assert r.success is True
        assert len(r.data["log"]) == 1

    @pytest.mark.asyncio
    async def test_execute_clear_log(self, tool, dry):
        await tool.execute(action="click", x=1, y=2)
        r = await tool.execute(action="clear_log")
        assert r.success is True
        assert "Cleared 1" in r.output

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, tool):
        r = await tool.execute(action="dance")
        assert r.success is False
        assert "Unknown action" in r.error

    @pytest.mark.asyncio
    async def test_execute_click_image_missing_file(self, tool, no_dry):
        r = await tool.execute(action="click_image", template="/nope.png")
        assert r.success is False
        assert "not found on disk" in r.error

    @pytest.mark.asyncio
    async def test_execute_screenshot_with_region(self, tool, dry):
        r = await tool.execute(action="screenshot", region=[0, 0, 100, 100])
        assert r.success is True


# ── Tests: registration ──────────────────────────────────────────────────────

class TestRegistration:
    def test_registered_in_global_registry(self):
        from bantz.tools import registry
        assert registry.get("gui") is not None

    def test_schema(self):
        s = gui_tool.schema()
        assert s["name"] == "gui"
        assert s["risk_level"] == "destructive"


# ── Tests: pyautogui global config ───────────────────────────────────────────

class TestPyautoguiConfig:
    def test_failsafe_disabled(self):
        pg = _get_pyautogui()
        if pg:
            assert pg.FAILSAFE is False

    def test_pause_zero(self):
        pg = _get_pyautogui()
        if pg:
            assert pg.PAUSE == 0
