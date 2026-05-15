"""Tests for DesktopTool (#322 — Computer Use & Desktop Automation).

Covers:
  1. parse_locator — locator string parsing (name:X, role:Y, combined)
  2. _resolve_app_command — app name → command resolution
  3. _format_tree_compact — tree formatting
  4. _extract_interactive_elements — element extraction from AT-SPI tree
  5. DesktopTool.get_ui_tree — tree retrieval + interactive element listing
  6. DesktopTool.interact — find + act (click/type/double_click/right_click)
  7. DesktopTool.click — shorthand click via locator
  8. DesktopTool.type_text — text input with optional locator
  9. DesktopTool.press_key — keyboard shortcuts
  10. DesktopTool.open_app — app launching
  11. DesktopTool.close_app — app closing
  12. DesktopTool.list_windows — accessible window listing
  13. DesktopTool.active_window — active window info
  14. Error handling & edge cases
  15. Tool registration
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from bantz.tools.desktop import (
    DesktopTool,
    parse_locator,
    _resolve_app_command,
    _format_tree_compact,
    _extract_interactive_elements,
    _APP_COMMANDS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tool():
    return DesktopTool()


@pytest.fixture
def sample_tree():
    """A realistic AT-SPI UI tree for testing."""
    return {
        "name": "Calculator",
        "role": "frame",
        "visible": True,
        "enabled": True,
        "bounds": {"x": 100, "y": 100, "width": 400, "height": 500},
        "center": (300, 350),
        "children": [
            {
                "name": "",
                "role": "panel",
                "visible": True,
                "enabled": True,
                "children": [
                    {
                        "name": "5",
                        "role": "push button",
                        "visible": True,
                        "enabled": True,
                        "bounds": {"x": 150, "y": 300, "width": 60, "height": 60},
                        "center": (180, 330),
                    },
                    {
                        "name": "9",
                        "role": "push button",
                        "visible": True,
                        "enabled": True,
                        "bounds": {"x": 270, "y": 300, "width": 60, "height": 60},
                        "center": (300, 330),
                    },
                    {
                        "name": "×",
                        "role": "push button",
                        "visible": True,
                        "enabled": True,
                        "bounds": {"x": 330, "y": 300, "width": 60, "height": 60},
                        "center": (360, 330),
                    },
                    {
                        "name": "=",
                        "role": "push button",
                        "visible": True,
                        "enabled": True,
                        "bounds": {"x": 330, "y": 420, "width": 60, "height": 60},
                        "center": (360, 450),
                    },
                    {
                        "name": "Result",
                        "role": "entry",
                        "visible": True,
                        "enabled": True,
                        "bounds": {"x": 120, "y": 120, "width": 350, "height": 50},
                        "center": (295, 145),
                    },
                    {
                        "name": "Hidden",
                        "role": "push button",
                        "visible": False,
                        "enabled": True,
                        "bounds": {"x": 0, "y": 0, "width": 0, "height": 0},
                        "center": (0, 0),
                    },
                    {
                        "name": "Disabled",
                        "role": "push button",
                        "visible": True,
                        "enabled": False,
                        "bounds": {"x": 400, "y": 400, "width": 60, "height": 60},
                        "center": (430, 430),
                    },
                ],
            },
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  1. parse_locator
# ══════════════════════════════════════════════════════════════════════════════

class TestParseLocator:
    def test_name_only(self):
        result = parse_locator("name:Submit")
        assert result == {"name": "Submit"}

    def test_role_only(self):
        result = parse_locator("role:push button")
        assert result == {"role": "push button"}

    def test_name_and_role(self):
        result = parse_locator("name:OK,role:push button")
        assert result == {"name": "OK", "role": "push button"}

    def test_case_insensitive_keys(self):
        result = parse_locator("Name:Send,Role:entry")
        assert result == {"name": "Send", "role": "entry"}

    def test_bare_string_becomes_name(self):
        result = parse_locator("Send button")
        assert result == {"name": "Send button"}

    def test_empty_string(self):
        result = parse_locator("")
        assert result == {}

    def test_whitespace_trimmed(self):
        result = parse_locator("  name:  OK  , role:  push button  ")
        assert result == {"name": "OK", "role": "push button"}

    def test_id_locator(self):
        result = parse_locator("id:btn_submit")
        assert result == {"id": "btn_submit"}


# ══════════════════════════════════════════════════════════════════════════════
#  2. _resolve_app_command
# ══════════════════════════════════════════════════════════════════════════════

class TestResolveAppCommand:
    def test_known_app(self):
        assert _resolve_app_command("calculator") == "gnome-calculator"

    def test_known_app_case_insensitive(self):
        assert _resolve_app_command("Calculator") == "gnome-calculator"

    def test_known_app_turkish(self):
        assert _resolve_app_command("hesap makinesi") == "gnome-calculator"

    def test_unknown_returns_as_is(self):
        assert _resolve_app_command("some-weird-app") == "some-weird-app"

    def test_which_found(self):
        with patch("bantz.tools.desktop.shutil.which", return_value="/usr/bin/ls"):
            result = _resolve_app_command("ls")
            assert result == "ls"

    def test_firefox(self):
        assert _resolve_app_command("firefox") == "firefox"


# ══════════════════════════════════════════════════════════════════════════════
#  3. _format_tree_compact
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatTreeCompact:
    def test_basic_formatting(self, sample_tree):
        lines: list[str] = []
        _format_tree_compact(sample_tree, lines)
        text = "\n".join(lines)
        assert "[frame]" in text
        assert '"Calculator"' in text
        assert '"5"' in text
        assert '"×"' in text

    def test_hidden_elements_skipped(self, sample_tree):
        lines: list[str] = []
        _format_tree_compact(sample_tree, lines)
        text = "\n".join(lines)
        assert '"Hidden"' not in text

    def test_disabled_shown_with_marker(self, sample_tree):
        lines: list[str] = []
        _format_tree_compact(sample_tree, lines)
        text = "\n".join(lines)
        assert "(disabled)" in text

    def test_truncation(self):
        tree = {
            "name": "root", "role": "frame", "visible": True,
            "enabled": True,
            "children": [
                {"name": f"item{i}", "role": "push button", "visible": True,
                 "enabled": True, "center": (i, i)}
                for i in range(200)
            ],
        }
        lines: list[str] = []
        _format_tree_compact(tree, lines, max_lines=10)
        assert len(lines) <= 11  # 10 + truncation message
        assert "truncated" in lines[-1]


# ══════════════════════════════════════════════════════════════════════════════
#  4. _extract_interactive_elements
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractInteractiveElements:
    def test_extracts_buttons_and_entry(self, sample_tree):
        elements: list[dict] = []
        _extract_interactive_elements(sample_tree, elements)
        names = [e["name"] for e in elements]
        assert "5" in names
        assert "9" in names
        assert "×" in names
        assert "=" in names
        assert "Result" in names

    def test_hidden_excluded(self, sample_tree):
        elements: list[dict] = []
        _extract_interactive_elements(sample_tree, elements)
        names = [e["name"] for e in elements]
        assert "Hidden" not in names

    def test_disabled_excluded(self, sample_tree):
        elements: list[dict] = []
        _extract_interactive_elements(sample_tree, elements)
        names = [e["name"] for e in elements]
        assert "Disabled" not in names

    def test_max_elements_limit(self, sample_tree):
        elements: list[dict] = []
        _extract_interactive_elements(sample_tree, elements, max_elements=2)
        assert len(elements) == 2

    def test_all_have_center(self, sample_tree):
        elements: list[dict] = []
        _extract_interactive_elements(sample_tree, elements)
        for elem in elements:
            assert "center" in elem
            assert len(elem["center"]) == 2


# ══════════════════════════════════════════════════════════════════════════════
#  5. get_ui_tree
# ══════════════════════════════════════════════════════════════════════════════

class TestGetUiTree:
    @pytest.mark.asyncio
    async def test_returns_interactive_elements(self, tool, sample_tree):
        with patch("bantz.tools.desktop._get_active_app_name", return_value="gnome-calculator"), \
             patch("bantz.tools.accessibility.get_element_tree", return_value=sample_tree):
            result = await tool.execute(action="get_ui_tree")
        assert result.success is True
        assert "interactive elements" in result.output
        assert "5" in result.output
        assert result.data["count"] == 5  # 5, 9, ×, =, Result

    @pytest.mark.asyncio
    async def test_with_explicit_app(self, tool, sample_tree):
        with patch("bantz.tools.accessibility.get_element_tree", return_value=sample_tree):
            result = await tool.execute(action="get_ui_tree", app="Calculator")
        assert result.success is True
        assert "Calculator" in result.output

    @pytest.mark.asyncio
    async def test_no_active_window(self, tool):
        with patch("bantz.tools.desktop._get_active_app_name", return_value=""), \
             patch("bantz.tools.desktop._get_active_window_name", return_value=""):
            result = await tool.execute(action="get_ui_tree")
        assert result.success is False
        assert "could not determine" in result.error.lower()

    @pytest.mark.asyncio
    async def test_atspi_unavailable(self, tool):
        with patch("bantz.tools.desktop._get_active_app_name", return_value="firefox"), \
             patch.dict("sys.modules", {"bantz.tools.accessibility": None}):
            result = await tool.execute(action="get_ui_tree")
        assert result.success is False
        assert "AT-SPI2" in result.error

    @pytest.mark.asyncio
    async def test_tree_not_found(self, tool):
        with patch("bantz.tools.desktop._get_active_app_name", return_value="ghost"), \
             patch("bantz.tools.accessibility.get_element_tree", return_value=None):
            result = await tool.execute(action="get_ui_tree")
        assert result.success is False
        assert "no accessibility tree" in result.error.lower()


# ══════════════════════════════════════════════════════════════════════════════
#  6. interact — find + act
# ══════════════════════════════════════════════════════════════════════════════

class TestInteract:
    @pytest.mark.asyncio
    async def test_click_by_name(self, tool):
        fake_element = {
            "name": "Submit", "role": "push button",
            "center": (200, 300), "bounds": {"x": 180, "y": 280, "width": 40, "height": 40},
            "score": 0.95,
        }
        mock_click = AsyncMock()
        with patch("bantz.tools.desktop._get_active_app_name", return_value="firefox"), \
             patch("bantz.tools.accessibility.find_element", return_value=fake_element), \
             patch("bantz.tools.input_control.click", mock_click):
            result = await tool.execute(action="interact", locator="name:Submit")
        assert result.success is True
        assert "Clicked" in result.output
        assert "Submit" in result.output
        mock_click.assert_called_once_with(200, 300)

    @pytest.mark.asyncio
    async def test_double_click(self, tool):
        fake_element = {
            "name": "file.txt", "role": "list item",
            "center": (150, 200), "bounds": {"x": 100, "y": 180, "width": 100, "height": 40},
            "score": 0.90,
        }
        mock_dbl = AsyncMock()
        with patch("bantz.tools.desktop._get_active_app_name", return_value="nautilus"), \
             patch("bantz.tools.accessibility.find_element", return_value=fake_element), \
             patch("bantz.tools.input_control.double_click", mock_dbl):
            result = await tool.execute(action="interact", locator="name:file.txt", interact_action="double_click")
        assert result.success is True
        assert "Double-clicked" in result.output
        mock_dbl.assert_called_once_with(150, 200)

    @pytest.mark.asyncio
    async def test_right_click(self, tool):
        fake_element = {
            "name": "Desktop", "role": "panel",
            "center": (500, 400), "bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
            "score": 0.80,
        }
        mock_rc = AsyncMock()
        with patch("bantz.tools.desktop._get_active_app_name", return_value="desktop"), \
             patch("bantz.tools.accessibility.find_element", return_value=fake_element), \
             patch("bantz.tools.input_control.right_click", mock_rc):
            result = await tool.execute(action="interact", locator="name:Desktop", interact_action="right_click")
        assert result.success is True
        assert "Right-clicked" in result.output
        mock_rc.assert_called_once_with(500, 400)

    @pytest.mark.asyncio
    async def test_type_into_element(self, tool):
        fake_element = {
            "name": "Search", "role": "entry",
            "center": (400, 50), "bounds": {"x": 300, "y": 30, "width": 200, "height": 40},
            "score": 0.92,
        }
        mock_click = AsyncMock()
        mock_type = AsyncMock()
        with patch("bantz.tools.desktop._get_active_app_name", return_value="firefox"), \
             patch("bantz.tools.accessibility.find_element", return_value=fake_element), \
             patch("bantz.tools.input_control.click", mock_click), \
             patch("bantz.tools.input_control.type_text", mock_type):
            result = await tool.execute(
                action="interact", locator="name:Search",
                interact_action="type", text="hello world",
            )
        assert result.success is True
        assert "Typed" in result.output
        mock_click.assert_called_once_with(400, 50)
        mock_type.assert_called_once_with("hello world")

    @pytest.mark.asyncio
    async def test_missing_locator(self, tool):
        result = await tool.execute(action="interact", locator="")
        assert result.success is False
        assert "locator is required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_element_not_found(self, tool):
        with patch("bantz.tools.desktop._get_active_app_name", return_value="firefox"), \
             patch("bantz.tools.accessibility.find_element", return_value=None):
            result = await tool.execute(action="interact", locator="name:Phantom")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_interact_action(self, tool):
        fake_element = {
            "name": "OK", "role": "push button",
            "center": (100, 100), "bounds": {"x": 80, "y": 80, "width": 40, "height": 40},
            "score": 1.0,
        }
        with patch("bantz.tools.desktop._get_active_app_name", return_value="dialog"), \
             patch("bantz.tools.accessibility.find_element", return_value=fake_element):
            result = await tool.execute(action="interact", locator="name:OK", interact_action="smash")
        assert result.success is False
        assert "unknown interact action" in result.error.lower()

    @pytest.mark.asyncio
    async def test_click_failure_caught(self, tool):
        fake_element = {
            "name": "OK", "role": "push button",
            "center": (100, 100), "bounds": {"x": 80, "y": 80, "width": 40, "height": 40},
            "score": 1.0,
        }
        with patch("bantz.tools.desktop._get_active_app_name", return_value="dialog"), \
             patch("bantz.tools.accessibility.find_element", return_value=fake_element), \
             patch("bantz.tools.input_control.click", AsyncMock(side_effect=OSError("Display error"))):
            result = await tool.execute(action="interact", locator="name:OK")
        assert result.success is False
        assert "click failed" in result.error.lower()


# ══════════════════════════════════════════════════════════════════════════════
#  7. click (shorthand)
# ══════════════════════════════════════════════════════════════════════════════

class TestClickShorthand:
    @pytest.mark.asyncio
    async def test_click_with_locator(self, tool):
        fake_element = {
            "name": "OK", "role": "push button",
            "center": (300, 200), "bounds": {"x": 280, "y": 180, "width": 40, "height": 40},
            "score": 1.0,
        }
        mock_click = AsyncMock()
        with patch("bantz.tools.desktop._get_active_app_name", return_value="dialog"), \
             patch("bantz.tools.accessibility.find_element", return_value=fake_element), \
             patch("bantz.tools.input_control.click", mock_click):
            result = await tool.execute(action="click", locator="name:OK")
        assert result.success is True
        mock_click.assert_called_once_with(300, 200)

    @pytest.mark.asyncio
    async def test_click_with_target(self, tool):
        fake_element = {
            "name": "Cancel", "role": "push button",
            "center": (400, 200), "bounds": {"x": 380, "y": 180, "width": 40, "height": 40},
            "score": 1.0,
        }
        mock_click = AsyncMock()
        with patch("bantz.tools.desktop._get_active_app_name", return_value="dialog"), \
             patch("bantz.tools.accessibility.find_element", return_value=fake_element), \
             patch("bantz.tools.input_control.click", mock_click):
            result = await tool.execute(action="click", target="Cancel")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_click_no_locator(self, tool):
        result = await tool.execute(action="click")
        assert result.success is False
        assert "locator" in result.error.lower() or "target" in result.error.lower()


# ══════════════════════════════════════════════════════════════════════════════
#  8. type_text
# ══════════════════════════════════════════════════════════════════════════════

class TestTypeText:
    @pytest.mark.asyncio
    async def test_type_without_locator(self, tool):
        mock_type = AsyncMock()
        with patch("bantz.tools.input_control.type_text", mock_type):
            result = await tool.execute(action="type_text", text="hello world")
        assert result.success is True
        assert "Typed" in result.output
        assert "11 chars" in result.output
        mock_type.assert_called_once_with("hello world")

    @pytest.mark.asyncio
    async def test_type_with_locator(self, tool):
        fake_element = {
            "name": "Search", "role": "entry",
            "center": (200, 50), "bounds": {"x": 100, "y": 30, "width": 200, "height": 40},
            "score": 0.95,
        }
        mock_click = AsyncMock()
        mock_type = AsyncMock()
        with patch("bantz.tools.desktop._get_active_app_name", return_value="firefox"), \
             patch("bantz.tools.accessibility.find_element", return_value=fake_element), \
             patch("bantz.tools.input_control.click", mock_click), \
             patch("bantz.tools.input_control.type_text", mock_type):
            result = await tool.execute(action="type_text", text="query", locator="name:Search")
        assert result.success is True
        mock_click.assert_called_once_with(200, 50)  # Focus first
        mock_type.assert_called_once_with("query")

    @pytest.mark.asyncio
    async def test_type_empty_text(self, tool):
        result = await tool.execute(action="type_text", text="")
        assert result.success is False
        assert "text" in result.error.lower()

    @pytest.mark.asyncio
    async def test_type_long_text_truncated_in_output(self, tool):
        mock_type = AsyncMock()
        long_text = "x" * 200
        with patch("bantz.tools.input_control.type_text", mock_type):
            result = await tool.execute(action="type_text", text=long_text)
        assert result.success is True
        assert "…" in result.output
        assert "200 chars" in result.output


# ══════════════════════════════════════════════════════════════════════════════
#  9. press_key
# ══════════════════════════════════════════════════════════════════════════════

class TestPressKey:
    @pytest.mark.asyncio
    async def test_single_key(self, tool):
        mock_hk = AsyncMock()
        with patch("bantz.tools.input_control.hotkey", mock_hk):
            result = await tool.execute(action="press_key", keys="Return")
        assert result.success is True
        assert "Return" in result.output
        mock_hk.assert_called_once_with("Return")

    @pytest.mark.asyncio
    async def test_combo_key(self, tool):
        mock_hk = AsyncMock()
        with patch("bantz.tools.input_control.hotkey", mock_hk):
            result = await tool.execute(action="press_key", keys="ctrl+s")
        assert result.success is True
        assert "ctrl+s" in result.output
        mock_hk.assert_called_once_with("ctrl", "s")

    @pytest.mark.asyncio
    async def test_key_alias(self, tool):
        """Action 'hotkey' should also work."""
        mock_hk = AsyncMock()
        with patch("bantz.tools.input_control.hotkey", mock_hk):
            result = await tool.execute(action="hotkey", keys="alt+tab")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_missing_keys(self, tool):
        result = await tool.execute(action="press_key")
        assert result.success is False
        assert "keys" in result.error.lower()

    @pytest.mark.asyncio
    async def test_key_error_caught(self, tool):
        with patch("bantz.tools.input_control.hotkey", AsyncMock(side_effect=RuntimeError("No backend"))):
            result = await tool.execute(action="press_key", keys="ctrl+z")
        assert result.success is False
        assert "failed to press" in result.error.lower()


# ══════════════════════════════════════════════════════════════════════════════
#  10. open_app
# ══════════════════════════════════════════════════════════════════════════════

class TestOpenApp:
    @pytest.mark.asyncio
    async def test_open_known_app(self, tool):
        mock_proc = AsyncMock()
        mock_proc.returncode = None  # still running
        mock_proc.pid = 12345
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await tool.execute(action="open_app", app="calculator")
        assert result.success is True
        assert "Launched" in result.output
        assert result.data["command"] == "gnome-calculator"
        assert result.data["pid"] == 12345

    @pytest.mark.asyncio
    async def test_open_app_not_found(self, tool):
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await tool.execute(action="open_app", app="nonexistent-app-xyz")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_open_app_crashes_immediately(self, tool):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.pid = 99999
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await tool.execute(action="open_app", app="calculator")
        assert result.success is False
        assert "exited" in result.error.lower()

    @pytest.mark.asyncio
    async def test_open_app_missing_name(self, tool):
        result = await tool.execute(action="open_app")
        assert result.success is False
        assert "app" in result.error.lower()

    @pytest.mark.asyncio
    async def test_open_alias(self, tool):
        """'open' action should work as alias."""
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.pid = 42
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await tool.execute(action="open", app="firefox")
        assert result.success is True


# ══════════════════════════════════════════════════════════════════════════════
#  11. close_app
# ══════════════════════════════════════════════════════════════════════════════

class TestCloseApp:
    @pytest.mark.asyncio
    async def test_close_via_wmctrl(self, tool):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("asyncio.wait_for", return_value=None):
            result = await tool.execute(action="close_app", app="Calculator")
        assert result.success is True
        assert "Closed" in result.output

    @pytest.mark.asyncio
    async def test_close_via_alt_f4(self, tool):
        mock_hk = AsyncMock()
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError), \
             patch("bantz.tools.input_control.hotkey", mock_hk):
            result = await tool.execute(action="close_app", app="SomeApp")
        assert result.success is True
        assert "close signal" in result.output.lower()
        mock_hk.assert_called_once_with("alt", "F4")

    @pytest.mark.asyncio
    async def test_close_active_window(self, tool):
        mock_hk = AsyncMock()
        with patch("bantz.tools.input_control.hotkey", mock_hk):
            result = await tool.execute(action="close_app")
        assert result.success is True
        assert "active window" in result.output.lower()


# ══════════════════════════════════════════════════════════════════════════════
#  12. list_windows
# ══════════════════════════════════════════════════════════════════════════════

class TestListWindows:
    @pytest.mark.asyncio
    async def test_list_windows_success(self, tool):
        with patch("bantz.tools.accessibility.list_applications", return_value=["Firefox", "Terminal", "VS Code"]):
            result = await tool.execute(action="list_windows")
        assert result.success is True
        assert "3" in result.output
        assert "Firefox" in result.output
        assert result.data["count"] == 3

    @pytest.mark.asyncio
    async def test_list_windows_empty(self, tool):
        with patch("bantz.tools.accessibility.list_applications", return_value=[]):
            result = await tool.execute(action="list_windows")
        assert result.success is True
        assert "no accessible" in result.output.lower()

    @pytest.mark.asyncio
    async def test_list_windows_atspi_unavailable(self, tool):
        with patch.dict("sys.modules", {"bantz.tools.accessibility": None}):
            result = await tool.execute(action="list_windows")
        assert result.success is False
        assert "AT-SPI2" in result.error


# ══════════════════════════════════════════════════════════════════════════════
#  13. active_window
# ══════════════════════════════════════════════════════════════════════════════

class TestActiveWindow:
    @pytest.mark.asyncio
    async def test_active_window_info(self, tool):
        with patch("bantz.tools.desktop._get_active_window_name", return_value="main.py — VS Code"), \
             patch("bantz.tools.desktop._get_active_app_name", return_value="code"), \
             patch("bantz.tools.desktop._get_active_window_pid", return_value=5678):
            result = await tool.execute(action="active_window")
        assert result.success is True
        assert "VS Code" in result.output
        assert "code" in result.output
        assert result.data["pid"] == 5678

    @pytest.mark.asyncio
    async def test_active_window_no_window(self, tool):
        with patch("bantz.tools.desktop._get_active_window_name", return_value=""), \
             patch("bantz.tools.desktop._get_active_app_name", return_value=""), \
             patch("bantz.tools.desktop._get_active_window_pid", return_value=0):
            result = await tool.execute(action="active_window")
        assert result.success is False
        assert "could not determine" in result.error.lower()


# ══════════════════════════════════════════════════════════════════════════════
#  14. Error handling & edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        result = await tool.execute(action="dance")
        assert result.success is False
        assert "unknown action" in result.error.lower()

    @pytest.mark.asyncio
    async def test_action_aliases(self, tool):
        """All alias actions should dispatch correctly."""
        # tree aliases
        for alias in ("get_ui_tree", "ui_tree", "tree"):
            with patch("bantz.tools.desktop._get_active_app_name", return_value=""), \
                 patch("bantz.tools.desktop._get_active_window_name", return_value=""):
                result = await tool.execute(action=alias)
                # Should fail because no window, but should dispatch correctly
                assert "could not determine" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handler_exception_caught(self, tool):
        """Unexpected exception in handler should be caught."""
        with patch.object(tool, "_get_ui_tree", side_effect=RuntimeError("boom")):
            result = await tool.execute(action="get_ui_tree")
        assert result.success is False
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_default_action_is_get_ui_tree(self, tool):
        """No action specified should default to get_ui_tree."""
        with patch("bantz.tools.desktop._get_active_app_name", return_value=""), \
             patch("bantz.tools.desktop._get_active_window_name", return_value=""):
            result = await tool.execute()
        assert "could not determine" in result.error.lower()


# ══════════════════════════════════════════════════════════════════════════════
#  15. Tool registration
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistration:
    def test_registered_in_global_registry(self):
        from bantz.tools import registry
        tool = registry.get("desktop")
        assert tool is not None
        assert tool.name == "desktop"

    def test_risk_level(self):
        from bantz.tools import registry
        tool = registry.get("desktop")
        assert tool.risk_level == "moderate"

    def test_schema_has_description(self):
        from bantz.tools import registry
        tool = registry.get("desktop")
        schema = tool.schema()
        assert "desktop" in schema["description"].lower() or "automation" in schema["description"].lower()
        assert "get_ui_tree" in schema["description"]
        assert "interact" in schema["description"]
        assert "open_app" in schema["description"]
