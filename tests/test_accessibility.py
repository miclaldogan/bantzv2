"""
Tests for AT-SPI2 accessibility tool (#119).

Covers:
  - Display server detection
  - Fuzzy matching algorithm
  - Tool actions (with mocked AT-SPI)
  - Quick route patterns in brain
  - Graceful fallback when AT-SPI unavailable
  - Element tree formatting
  - Window focus logic
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


# ── Fuzzy matching ─────────────────────────────────────────────────────────

class TestFuzzyMatch:
    def test_exact_match(self):
        from bantz.tools.accessibility import _fuzzy_match
        assert _fuzzy_match("Send", "Send") == 1.0

    def test_case_insensitive(self):
        from bantz.tools.accessibility import _fuzzy_match
        assert _fuzzy_match("send", "Send") == 1.0

    def test_substring(self):
        from bantz.tools.accessibility import _fuzzy_match
        score = _fuzzy_match("Send", "Send Message")
        assert score >= 0.8

    def test_reverse_substring(self):
        from bantz.tools.accessibility import _fuzzy_match
        score = _fuzzy_match("Send Message Button", "Send")
        assert score >= 0.7

    def test_word_overlap(self):
        from bantz.tools.accessibility import _fuzzy_match
        score = _fuzzy_match("URL bar", "URL Address Bar")
        assert score > 0.5

    def test_no_match(self):
        from bantz.tools.accessibility import _fuzzy_match
        score = _fuzzy_match("Send", "Cancel")
        assert score < 0.5

    def test_empty_strings(self):
        from bantz.tools.accessibility import _fuzzy_match
        assert _fuzzy_match("", "Send") == 0.0
        assert _fuzzy_match("Send", "") == 0.0
        assert _fuzzy_match("", "") == 0.0

    def test_bigram_similarity(self):
        from bantz.tools.accessibility import _fuzzy_match
        # Similar but not exact or substring
        score = _fuzzy_match("Settings", "Setting")
        assert score > 0.3

    def test_whitespace_handling(self):
        from bantz.tools.accessibility import _fuzzy_match
        assert _fuzzy_match("  Send  ", "Send") == 1.0


# ── Display server detection ──────────────────────────────────────────────

class TestDisplayServer:
    def test_detect_x11(self):
        from bantz.tools.accessibility import detect_display_server
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}, clear=False):
            assert detect_display_server() == "x11"

    def test_detect_wayland(self):
        from bantz.tools.accessibility import detect_display_server
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=False):
            assert detect_display_server() == "wayland"

    def test_detect_wayland_display(self):
        from bantz.tools.accessibility import detect_display_server
        env = {"XDG_SESSION_TYPE": "", "WAYLAND_DISPLAY": "wayland-0"}
        with patch.dict(os.environ, env, clear=False):
            result = detect_display_server()
            assert result in ("wayland", "x11")  # may have DISPLAY too

    def test_detect_x11_display(self):
        from bantz.tools.accessibility import detect_display_server
        env = {"XDG_SESSION_TYPE": "", "DISPLAY": ":0"}
        with patch.dict(os.environ, env, clear=False):
            result = detect_display_server()
            assert result in ("x11", "wayland")  # may have WAYLAND_DISPLAY too


# ── Tool actions with mocked AT-SPI ───────────────────────────────────────

def _mock_atspi_module():
    """Create a mock AT-SPI module."""
    mock = MagicMock()
    mock.CoordType.SCREEN = 0
    mock.StateType.VISIBLE = 1
    mock.StateType.ENABLED = 2
    mock.StateType.FOCUSED = 3
    return mock


class TestAccessibilityToolActions:
    """Test tool execute() with AT-SPI mocked at module level."""

    def _make_tool(self):
        from bantz.tools.accessibility import AccessibilityTool
        return AccessibilityTool()

    @pytest.mark.asyncio
    async def test_info_action(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = True
            mod._atspi = _mock_atspi_module()

            # Mock desktop with 0 children
            mock_desktop = MagicMock()
            mock_desktop.get_child_count.return_value = 0
            mod._atspi.get_desktop.return_value = mock_desktop

            tool = self._make_tool()
            result = await tool.execute(action="info")
            assert result.success
            assert "AT-SPI2: available" in result.output
            assert result.data["atspi_available"] is True
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi

    @pytest.mark.asyncio
    async def test_list_apps_empty(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = True
            mod._atspi = _mock_atspi_module()

            mock_desktop = MagicMock()
            mock_desktop.get_child_count.return_value = 0
            mod._atspi.get_desktop.return_value = mock_desktop

            tool = self._make_tool()
            result = await tool.execute(action="list_apps")
            assert result.success
            assert result.data["apps"] == []
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi

    @pytest.mark.asyncio
    async def test_list_apps_with_apps(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = True
            mod._atspi = _mock_atspi_module()

            # Mock desktop with 2 apps
            app1 = MagicMock()
            app1.get_name.return_value = "Firefox"
            app2 = MagicMock()
            app2.get_name.return_value = "Terminal"

            mock_desktop = MagicMock()
            mock_desktop.get_child_count.return_value = 2
            mock_desktop.get_child_at_index.side_effect = [app1, app2]
            mod._atspi.get_desktop.return_value = mock_desktop

            tool = self._make_tool()
            result = await tool.execute(action="list_apps")
            assert result.success
            assert "Firefox" in result.data["apps"]
            assert "Terminal" in result.data["apps"]
            assert "2 accessible" in result.output
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi

    @pytest.mark.asyncio
    async def test_find_missing_args(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = True
            mod._atspi = _mock_atspi_module()

            tool = self._make_tool()
            result = await tool.execute(action="find", app="", label="")
            assert not result.success
            assert "Specify" in result.error
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi

    @pytest.mark.asyncio
    async def test_tree_missing_app(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = True
            mod._atspi = _mock_atspi_module()

            tool = self._make_tool()
            result = await tool.execute(action="tree", app="")
            assert not result.success
            assert "Specify" in result.error
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi

    @pytest.mark.asyncio
    async def test_focus_missing_app(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = True
            mod._atspi = _mock_atspi_module()

            tool = self._make_tool()
            result = await tool.execute(action="focus", app="")
            assert not result.success
            assert "Specify" in result.error
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = True
            mod._atspi = _mock_atspi_module()

            tool = self._make_tool()
            result = await tool.execute(action="unknown_action")
            assert not result.success
            assert "Unknown action" in result.error
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi


# ── Graceful fallback ─────────────────────────────────────────────────────

class TestATSPIUnavailable:
    @pytest.mark.asyncio
    async def test_graceful_error_when_unavailable(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = False
            mod._atspi = None

            tool = mod.AccessibilityTool()
            result = await tool.execute(action="info")
            assert not result.success
            assert "AT-SPI2 is not available" in result.error
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi

    def test_list_applications_when_unavailable(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = False
            mod._atspi = None
            assert mod.list_applications() == []
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi

    def test_get_element_tree_when_unavailable(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = False
            mod._atspi = None
            assert mod.get_element_tree("firefox") is None
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi

    def test_find_element_when_unavailable(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = False
            mod._atspi = None
            assert mod.find_element("firefox", "Send") is None
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi


# ── Tree formatting ───────────────────────────────────────────────────────

class TestTreeFormatting:
    def test_format_tree_lines(self):
        from bantz.tools.accessibility import AccessibilityTool
        tool = AccessibilityTool()

        tree = {
            "name": "Firefox",
            "role": "application",
            "bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
            "center": (960, 540),
            "children": [
                {
                    "name": "Navigation Toolbar",
                    "role": "tool bar",
                    "bounds": {"x": 0, "y": 0, "width": 1920, "height": 40},
                    "center": (960, 20),
                },
                {
                    "name": "URL bar",
                    "role": "entry",
                    "bounds": {"x": 200, "y": 5, "width": 800, "height": 30},
                    "center": (600, 20),
                },
            ],
        }

        lines: list[str] = []
        tool._format_tree_lines(tree, lines)
        assert len(lines) == 3
        assert "[application]" in lines[0]
        assert "Firefox" in lines[0]
        assert "[tool bar]" in lines[1]
        assert "[entry]" in lines[2]

    def test_format_tree_truncation(self):
        from bantz.tools.accessibility import AccessibilityTool
        tool = AccessibilityTool()

        tree = {
            "name": "App",
            "role": "application",
            "children": [
                {"name": f"Item {i}", "role": "button"}
                for i in range(100)
            ],
        }

        lines: list[str] = []
        tool._format_tree_lines(tree, lines, max_lines=10)
        assert len(lines) <= 12  # 10 + truncation message + some buffer
        assert any("truncated" in line for line in lines)


# ── Quick route patterns ──────────────────────────────────────────────────

class TestQuickRouteAccessibility:
    @staticmethod
    def _quick(orig: str, en: str = "") -> dict | None:
        from bantz.core.brain import Brain
        return Brain._quick_route(orig, en or orig)

    def test_click_button_route(self):
        result = self._quick("click the Send button in Firefox")
        assert result is not None
        assert result["tool"] == "accessibility"

    def test_list_apps_route(self):
        result = self._quick("list open apps")
        assert result is not None
        assert result["tool"] == "accessibility"
        assert result["args"]["action"] == "list_apps"

    def test_focus_window_route(self):
        result = self._quick("focus window firefox")
        assert result is not None
        assert result["tool"] == "accessibility"
        assert result["args"]["action"] == "focus"

    def test_element_tree_route(self):
        result = self._quick("element tree of firefox")
        assert result is not None
        assert result["tool"] == "accessibility"
        assert result["args"]["action"] == "tree"

    def test_find_ui_element_route(self):
        result = self._quick("show the ui element tree for firefox")
        assert result is not None
        assert result["tool"] == "accessibility"

    def test_switch_to_app_route(self):
        result = self._quick("switch to terminal")
        assert result is not None
        assert result["tool"] == "accessibility"
        assert result["args"]["action"] == "focus"

    def test_accessibility_info_fallback(self):
        result = self._quick("check accessibility info")
        assert result is not None
        assert result["tool"] == "accessibility"
        assert result["args"]["action"] == "info"

    def test_unrelated_not_matched(self):
        """Normal chat should not trigger accessibility."""
        result = self._quick("what's the weather today?")
        assert result is None or result["tool"] != "accessibility"


# ── Tool schema ───────────────────────────────────────────────────────────

class TestToolSchema:
    def test_schema(self):
        from bantz.tools.accessibility import AccessibilityTool
        tool = AccessibilityTool()
        schema = tool.schema()
        assert schema["name"] == "accessibility"
        assert "safe" == schema["risk_level"]
        assert "accessibility" in schema["description"].lower()


# ── Find element with mocked tree ─────────────────────────────────────────

class TestFindElement:
    def test_find_element_in_mocked_tree(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mock_atspi = _mock_atspi_module()
            mod._atspi_available = True
            mod._atspi = mock_atspi

            # Build mock tree: desktop → app → button
            mock_button = MagicMock()
            mock_button.get_name.return_value = "Send"
            mock_button.get_role_name.return_value = "push button"
            mock_button.get_child_count.return_value = 0
            # Component with bounds
            mock_comp = MagicMock()
            mock_ext = MagicMock()
            mock_ext.x = 100
            mock_ext.y = 200
            mock_ext.width = 80
            mock_ext.height = 30
            mock_comp.get_extents.return_value = mock_ext
            mock_button.get_component_iface.return_value = mock_comp
            # States
            mock_states = MagicMock()
            mock_states.contains.return_value = True
            mock_button.get_state_set.return_value = mock_states

            mock_app = MagicMock()
            mock_app.get_name.return_value = "Firefox"
            mock_app.get_child_count.return_value = 1
            mock_app.get_child_at_index.return_value = mock_button
            mock_app.get_role_name.return_value = "application"

            mock_desktop = MagicMock()
            mock_desktop.get_child_count.return_value = 1
            mock_desktop.get_child_at_index.return_value = mock_app
            mock_atspi.get_desktop.return_value = mock_desktop

            result = mod.find_element("Firefox", "Send")
            assert result is not None
            assert result["name"] == "Send"
            assert result["role"] == "push button"
            assert result["center"] == (140, 215)  # 100+80//2, 200+30//2
            assert result["score"] >= 0.9
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi

    def test_find_element_no_match(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mock_atspi = _mock_atspi_module()
            mod._atspi_available = True
            mod._atspi = mock_atspi

            mock_button = MagicMock()
            mock_button.get_name.return_value = "Cancel"
            mock_button.get_role_name.return_value = "push button"
            mock_button.get_child_count.return_value = 0
            mock_comp = MagicMock()
            mock_ext = MagicMock()
            mock_ext.x = 10
            mock_ext.y = 10
            mock_ext.width = 50
            mock_ext.height = 20
            mock_comp.get_extents.return_value = mock_ext
            mock_button.get_component_iface.return_value = mock_comp

            mock_app = MagicMock()
            mock_app.get_name.return_value = "Firefox"
            mock_app.get_child_count.return_value = 1
            mock_app.get_child_at_index.return_value = mock_button

            mock_desktop = MagicMock()
            mock_desktop.get_child_count.return_value = 1
            mock_desktop.get_child_at_index.return_value = mock_app
            mock_atspi.get_desktop.return_value = mock_desktop

            result = mod.find_element("Firefox", "Send")
            assert result is None  # "Cancel" doesn't match "Send" well enough
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi


# ── Focus window ──────────────────────────────────────────────────────────

class TestFocusWindow:
    def test_focus_via_wmctrl(self):
        from bantz.tools.accessibility import focus_window
        with patch("bantz.tools.accessibility.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert focus_window("Firefox") is True
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "wmctrl" in args

    def test_focus_wmctrl_missing_falls_to_xdotool(self):
        from bantz.tools.accessibility import focus_window
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise FileNotFoundError("wmctrl not found")
            return MagicMock(returncode=0)

        with patch("bantz.tools.accessibility.subprocess.run", side_effect=side_effect), \
             patch("bantz.tools.accessibility.detect_display_server", return_value="x11"), \
             patch("bantz.tools.accessibility._find_app", return_value=None):
            result = focus_window("Firefox")
            assert result is True
            assert call_count == 2  # wmctrl failed, xdotool succeeded

    def test_focus_all_fail(self):
        import bantz.tools.accessibility as mod
        old_avail = mod._atspi_available
        old_atspi = mod._atspi
        try:
            mod._atspi_available = False
            mod._atspi = None

            with patch("bantz.tools.accessibility.subprocess.run", side_effect=FileNotFoundError), \
                 patch("bantz.tools.accessibility.detect_display_server", return_value="x11"):
                result = mod.focus_window("NonExistent")
                assert result is False
        finally:
            mod._atspi_available = old_avail
            mod._atspi = old_atspi
