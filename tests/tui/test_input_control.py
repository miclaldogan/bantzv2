"""
Tests for Bantz v3 — Input Control Tool (#122)

Coverage:
  - Backend detection
  - Safety classification (safe/moderate/destructive)
  - Action logging
  - Click, double_click, right_click, move_to, scroll, drag, hotkey, type_text
  - Tool class execute dispatch
  - Config gating (input_control_enabled)
  - Brain quick_route patterns
  - Destructive hotkey confirmation
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from bantz.tools.input_control import (
    classify_action,
    _log_action,
    _action_log,
    get_action_log,
    InputControlTool,
    DESTRUCTIVE_HOTKEYS,
    MODERATE_HOTKEYS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_action_log():
    """Clear action log before each test."""
    _action_log.clear()
    yield
    _action_log.clear()


@pytest.fixture
def tool():
    return InputControlTool()


# ── Safety Classification ─────────────────────────────────────────────────

class TestClassifyAction:
    def test_click_is_safe(self):
        assert classify_action("click") == "safe"

    def test_move_to_is_safe(self):
        assert classify_action("move_to") == "safe"

    def test_scroll_is_safe(self):
        assert classify_action("scroll") == "safe"

    def test_get_position_is_safe(self):
        assert classify_action("get_position") == "safe"

    def test_double_click_moderate(self):
        assert classify_action("double_click") == "moderate"

    def test_right_click_moderate(self):
        assert classify_action("right_click") == "moderate"

    def test_drag_moderate(self):
        assert classify_action("drag") == "moderate"

    def test_type_text_moderate(self):
        assert classify_action("type_text") == "moderate"

    def test_hotkey_ctrl_s_moderate(self):
        assert classify_action("hotkey", keys=("ctrl", "s")) == "moderate"

    def test_hotkey_ctrl_w_destructive(self):
        assert classify_action("hotkey", keys=("ctrl", "w")) == "destructive"

    def test_hotkey_alt_f4_destructive(self):
        assert classify_action("hotkey", keys=("alt", "f4")) == "destructive"

    def test_hotkey_ctrl_q_destructive(self):
        assert classify_action("hotkey", keys=("ctrl", "q")) == "destructive"

    def test_hotkey_string_parsing(self):
        assert classify_action("hotkey", keys="ctrl+w") == "destructive"

    def test_hotkey_enter_moderate(self):
        assert classify_action("hotkey", keys=("enter",)) == "moderate"

    def test_unknown_hotkey_moderate(self):
        assert classify_action("hotkey", keys=("ctrl", "shift", "f7")) == "moderate"

    def test_unknown_action_moderate(self):
        assert classify_action("unknown_action") == "moderate"


class TestDestructiveHotkeySet:
    def test_ctrl_w_in_set(self):
        assert ("ctrl", "w") in DESTRUCTIVE_HOTKEYS

    def test_alt_f4_in_set(self):
        assert ("alt", "f4") in DESTRUCTIVE_HOTKEYS

    def test_ctrl_s_not_destructive(self):
        assert ("ctrl", "s") not in DESTRUCTIVE_HOTKEYS

    def test_ctrl_s_in_moderate(self):
        assert ("ctrl", "s") in MODERATE_HOTKEYS


# ── Action Logging ────────────────────────────────────────────────────────

class TestActionLog:
    def test_log_adds_entry(self):
        with patch("bantz.tools.input_control.log"):
            _log_action("click", {"x": 100, "y": 200}, "safe")
        entries = get_action_log()
        assert len(entries) == 1
        assert entries[0]["action"] == "click"
        assert entries[0]["risk"] == "safe"
        assert "timestamp" in entries[0]

    def test_log_caps_at_500(self):
        with patch("bantz.tools.input_control.log"):
            for i in range(510):
                _log_action("click", {"x": i}, "safe")
        assert len(get_action_log()) == 500

    def test_log_returns_copy(self):
        with patch("bantz.tools.input_control.log"):
            _log_action("click", {"x": 0}, "safe")
        log1 = get_action_log()
        log1.clear()
        assert len(get_action_log()) == 1  # original not affected


# ── Backend Detection ─────────────────────────────────────────────────────

class TestBackendDetection:
    @patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=False)
    @patch("bantz.tools.input_control._backend", None)
    def test_x11_prefers_pyautogui(self):
        from bantz.tools.input_control import _detect_backend
        with patch.dict("sys.modules", {"pyautogui": MagicMock()}):
            import bantz.tools.input_control as ic
            ic._backend = None
            backend = _detect_backend()
            assert backend == "pyautogui"

    @patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"}, clear=False)
    @patch("bantz.tools.input_control._backend", None)
    def test_wayland_prefers_pynput(self):
        from bantz.tools.input_control import _detect_backend
        with patch.dict("sys.modules", {
            "pynput": MagicMock(),
            "pynput.mouse": MagicMock(),
            "pynput.keyboard": MagicMock(),
        }):
            import bantz.tools.input_control as ic
            ic._backend = None
            backend = _detect_backend()
            assert backend == "pynput"


# ── Tool Execute (mocked backends) ───────────────────────────────────────

class TestToolExecute:
    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="none")
    async def test_no_backend(self, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="click", x=100, y=200)
        assert not result.success
        assert "No input backend" in result.error

    @patch("bantz.tools.input_control.config")
    async def test_disabled(self, mock_cfg, tool):
        mock_cfg.input_control_enabled = False
        result = await tool.execute(action="click", x=100, y=200)
        assert not result.success
        assert "disabled" in result.error

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    @patch("bantz.tools.input_control.click", new_callable=AsyncMock, return_value={"x": 100, "y": 200, "button": "left"})
    async def test_click(self, mock_click, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="click", x=100, y=200)
        assert result.success
        assert "Clicked" in result.output
        mock_click.assert_called_once_with(100, 200)

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    @patch("bantz.tools.input_control.double_click", new_callable=AsyncMock, return_value={"x": 100, "y": 200})
    async def test_double_click(self, mock_dc, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="double_click", x=100, y=200)
        assert result.success
        assert "Double-clicked" in result.output

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    @patch("bantz.tools.input_control.right_click", new_callable=AsyncMock, return_value={"x": 100, "y": 200})
    async def test_right_click(self, mock_rc, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="right_click", x=100, y=200)
        assert result.success
        assert "Right-clicked" in result.output

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    @patch("bantz.tools.input_control.move_to", new_callable=AsyncMock, return_value={"x": 300, "y": 400})
    async def test_move_to(self, mock_move, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="move_to", x=300, y=400)
        assert result.success
        assert "Moved" in result.output

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    @patch("bantz.tools.input_control.type_text", new_callable=AsyncMock, return_value={"text": "hello", "length": 5, "interval_ms": 50})
    async def test_type_text(self, mock_type, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        mock_cfg.input_confirm_destructive = False
        result = await tool.execute(action="type_text", text="hello")
        assert result.success
        assert "Typed" in result.output

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    @patch("bantz.tools.input_control.scroll", new_callable=AsyncMock, return_value={"direction": "down", "amount": 3})
    async def test_scroll(self, mock_scroll, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="scroll", direction="down", amount=3)
        assert result.success
        assert "Scrolled" in result.output

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    @patch("bantz.tools.input_control.drag", new_callable=AsyncMock, return_value={"from": (0, 0), "to": (100, 100)})
    async def test_drag(self, mock_drag, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="drag", from_x=0, from_y=0, to_x=100, to_y=100)
        assert result.success
        assert "Dragged" in result.output

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    @patch("bantz.tools.input_control.hotkey", new_callable=AsyncMock, return_value={"keys": ["ctrl", "s"], "combo": "ctrl+s"})
    async def test_hotkey_safe(self, mock_hk, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        mock_cfg.input_confirm_destructive = True
        result = await tool.execute(action="hotkey", keys="ctrl+s")
        assert result.success
        assert "Pressed" in result.output

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    async def test_hotkey_destructive_confirm(self, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        mock_cfg.input_confirm_destructive = True
        result = await tool.execute(action="hotkey", keys="ctrl+w")
        assert result.success
        assert "_needs_confirm" in result.data
        assert "Destructive" in result.output

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    @patch("bantz.tools.input_control.get_position", new_callable=AsyncMock, return_value={"x": 500, "y": 300})
    async def test_get_position(self, mock_pos, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="get_position")
        assert result.success
        assert "500" in result.output

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    async def test_action_log_result(self, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="action_log")
        assert result.success
        assert "No input actions logged" in result.output

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    async def test_unknown_action(self, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="nonexistent")
        assert not result.success
        assert "Unknown action" in result.error

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    async def test_click_no_coords(self, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="click")
        assert not result.success
        assert "coordinates" in result.error

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    async def test_type_text_empty(self, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="type_text", text="")
        assert not result.success

    @patch("bantz.tools.input_control.config")
    @patch("bantz.tools.input_control._detect_backend", return_value="pyautogui")
    async def test_hotkey_empty(self, mock_detect, mock_cfg, tool):
        mock_cfg.input_control_enabled = True
        result = await tool.execute(action="hotkey", keys="")
        assert not result.success


# ── Tool Properties ───────────────────────────────────────────────────────

class TestToolProperties:
    def test_name(self, tool):
        assert tool.name == "input_control"

    def test_risk_level(self, tool):
        assert tool.risk_level == "destructive"

    def test_description(self, tool):
        assert "mouse" in tool.description.lower()
        assert "keyboard" in tool.description.lower()

    def test_schema(self, tool):
        s = tool.schema()
        assert s["name"] == "input_control"
        assert s["risk_level"] == "destructive"


# ── Brain Quick Route ─────────────────────────────────────────────────────

class TestQuickRoute:
    """Test brain._quick_route patterns for input control."""

    @staticmethod
    def _route(text: str):
        from bantz.core.brain import Brain
        return Brain._quick_route(text, text)

    def test_scroll_down(self):
        r = self._route("scroll down")
        assert r is None

    def test_scroll_up_5(self):
        r = self._route("scroll up 5")
        assert r is None

    def test_type_quoted_text(self):
        r = self._route('type "hello world"')
        assert r is None

    def test_hotkey_ctrl_s(self):
        r = self._route("press ctrl+s")
        assert r is None

    def test_press_enter(self):
        r = self._route("press enter")
        assert r is None

    def test_press_escape(self):
        r = self._route("press escape")
        assert r is None

    def test_double_click_coords(self):
        r = self._route("double click at 500, 300")
        assert r is None

    def test_right_click_coords(self):
        r = self._route("right click at 200, 100")
        assert r is None

    def test_drag_coords(self):
        r = self._route("drag from 100, 200 to 300, 400")
        assert r is None

    def test_mouse_position(self):
        r = self._route("mouse position")
        assert r is None

    def test_move_mouse(self):
        r = self._route("move mouse to 500, 300")
        assert r is None

    def test_click_the_still_goes_to_a11y(self):
        """'click the Send button' should go to gui_action (unified pipeline), not input_control."""
        r = self._route("click the Send button in Firefox")
        assert r is None
        # #123: gui_action is the unified pipeline that wraps AT-SPI

    def test_scroll_doesnt_match_random(self):
        r = self._route("what is a scrollbar?")
        # Should NOT match input_control
        assert r is None or r.get("tool") != "input_control"