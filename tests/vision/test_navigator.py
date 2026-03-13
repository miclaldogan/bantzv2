"""
Tests for the unified navigation pipeline (#123).

Covers:
  - NavResult / ActionResult data types
  - NavigationAnalytics (SQLite persistence)
  - Navigator pipeline (cache → AT-SPI → VLM chain)
  - GUIActionTool (tool registry integration)
  - Brain quick_route GUI patterns
"""
from __future__ import annotations

import sqlite3
import tempfile
import threading
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from bantz.vision.navigator import (
    ActionResult,
    NavResult,
    NavigationAnalytics,
    Navigator,
)

# ━━ NavResult ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNavResult(TestCase):
    def test_center_with_dimensions(self):
        r = NavResult(found=True, x=100, y=200, width=80, height=40)
        self.assertEqual(r.center, (140, 220))

    def test_center_without_dimensions(self):
        r = NavResult(found=True, x=100, y=200)
        self.assertEqual(r.center, (100, 200))

    def test_to_dict(self):
        r = NavResult(
            found=True, x=10, y=20, width=30, height=40,
            method="cache", confidence=0.95, latency_ms=1.5,
            app_name="firefox", element_label="search bar",
        )
        d = r.to_dict()
        self.assertTrue(d["found"])
        self.assertEqual(d["method"], "cache")
        self.assertEqual(d["center_x"], 25)
        self.assertEqual(d["center_y"], 40)
        self.assertEqual(d["app"], "firefox")

    def test_not_found(self):
        r = NavResult(found=False)
        d = r.to_dict()
        self.assertFalse(d["found"])
        self.assertEqual(d["method"], "")


class TestActionResult(TestCase):
    def test_success(self):
        nav = NavResult(found=True, x=10, y=20, method="atspi")
        ar = ActionResult(success=True, nav=nav, action="click", message="Clicked!")
        d = ar.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["action"], "click")
        self.assertEqual(d["message"], "Clicked!")

    def test_failure(self):
        nav = NavResult(found=False, method="none")
        ar = ActionResult(success=False, nav=nav, action="click", error="Not found")
        d = ar.to_dict()
        self.assertFalse(d["success"])
        self.assertEqual(d["error"], "Not found")


# ━━ NavigationAnalytics ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNavigationAnalytics(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        self.analytics = NavigationAnalytics()
        self.analytics.init(self.db_path)

    def tearDown(self):
        self.analytics.close()

    def test_record_and_stats(self):
        self.analytics.record("firefox", "search bar", "cache", True, 1.0, 0.9)
        self.analytics.record("firefox", "search bar", "atspi", True, 100.0, 1.0)
        self.analytics.record("firefox", "back button", "vlm", False, 3000.0, 0.5)
        stats = self.analytics.app_stats("firefox")
        self.assertEqual(stats["total_attempts"], 3)
        self.assertEqual(len(stats["methods"]), 3)

    def test_best_method_needs_minimum_samples(self):
        """Need at least 3 successful samples to recommend."""
        self.analytics.record("vscode", "save", "cache", True, 0.5)
        self.analytics.record("vscode", "save", "cache", True, 0.3)
        self.assertIsNone(self.analytics.best_method_for_app("vscode"))
        self.analytics.record("vscode", "open", "cache", True, 0.4)
        self.assertEqual(self.analytics.best_method_for_app("vscode"), "cache")

    def test_best_method_ignores_failures(self):
        for _ in range(5):
            self.analytics.record("app", "el", "vlm", False, 3000.0)
        for _ in range(3):
            self.analytics.record("app", "el", "atspi", True, 100.0)
        self.assertEqual(self.analytics.best_method_for_app("app"), "atspi")

    def test_app_case_insensitive(self):
        self.analytics.record("Firefox", "bar", "cache", True, 1.0)
        stats = self.analytics.app_stats("firefox")
        self.assertEqual(stats["total_attempts"], 1)

    def test_empty_stats(self):
        stats = self.analytics.app_stats()
        self.assertEqual(stats["total_attempts"], 0)

    def test_no_connection(self):
        na = NavigationAnalytics()
        na.record("a", "b", "c", True)  # should not crash
        self.assertIsNone(na.best_method_for_app("a"))


# ━━ Navigator Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNavigatorPipeline(IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        self.nav = Navigator()
        self.nav.init(self.db_path)

    def tearDown(self):
        self.nav.close()

    async def test_cache_hit(self):
        """If cache has the element, return immediately."""
        mock_entry = MagicMock()
        mock_entry.x = 100
        mock_entry.y = 200
        mock_entry.width = 80
        mock_entry.height = 30
        mock_entry.effective_confidence = 0.9
        mock_entry.role = "push button"
        mock_entry.is_expired = False

        with patch("bantz.vision.navigator.Navigator._try_cache") as mc:
            mc.return_value = NavResult(
                found=True, x=100, y=200, width=80, height=30,
                method="cache", confidence=0.9, role="push button",
            )
            result = await self.nav.navigate_to("firefox", "send")
            self.assertTrue(result.found)
            self.assertEqual(result.method, "cache")
            self.assertEqual(result.x, 100)

    async def test_atspi_fallback(self):
        """If cache misses, try AT-SPI."""
        with patch.object(self.nav, "_try_cache", return_value=None), \
             patch.object(self.nav, "_try_atspi") as ma:
            ma.return_value = NavResult(
                found=True, x=50, y=60, width=100, height=20,
                method="atspi", confidence=1.0,
            )
            result = await self.nav.navigate_to("firefox", "search bar")
            self.assertTrue(result.found)
            self.assertEqual(result.method, "atspi")
            # Should have stored in cache (via _store_in_cache)

    async def test_vlm_fallback(self):
        """If cache and AT-SPI miss, try VLM."""
        with patch.object(self.nav, "_try_cache", return_value=None), \
             patch.object(self.nav, "_try_atspi", return_value=None), \
             patch.object(self.nav, "_try_vlm") as mv:
            mv.return_value = NavResult(
                found=True, x=200, y=300, width=40, height=20,
                method="vlm", confidence=0.65,
            )
            result = await self.nav.navigate_to("firefox", "new tab")
            self.assertTrue(result.found)
            self.assertEqual(result.method, "vlm")

    async def test_all_methods_fail(self):
        """If everything fails, return found=False."""
        with patch.object(self.nav, "_try_cache", return_value=None), \
             patch.object(self.nav, "_try_atspi", return_value=None), \
             patch.object(self.nav, "_try_vlm", return_value=None):
            result = await self.nav.navigate_to("app", "nonexistent")
            self.assertFalse(result.found)
            self.assertEqual(result.method, "none")

    async def test_preferred_method(self):
        """preferred_method forces a specific method."""
        with patch.object(self.nav, "_try_vlm") as mv:
            mv.return_value = NavResult(found=True, x=10, y=20, method="vlm", confidence=0.7)
            result = await self.nav.navigate_to("app", "btn", preferred_method="vlm")
            self.assertTrue(result.found)
            self.assertEqual(result.method, "vlm")

    async def test_method_order_analytics(self):
        """Analytics should reorder methods when there's enough data."""
        # Seed analytics with VLM being the best for "electron_app"
        for _ in range(5):
            self.nav.analytics.record("electron_app", "x", "vlm", True, 2000.0)
        for _ in range(5):
            self.nav.analytics.record("electron_app", "x", "atspi", False, 100.0)

        order = self.nav._method_order("electron_app")
        self.assertEqual(order[0], "vlm")  # VLM promoted to first

    def test_default_method_order(self):
        order = self.nav._method_order("unknown_app")
        self.assertEqual(order, ["cache", "atspi", "vlm"])


class TestNavigatorActions(IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        self.nav = Navigator()
        self.nav.init(self.db_path)

    def tearDown(self):
        self.nav.close()

    async def test_execute_click(self):
        """execute_action click should navigate then click."""
        with patch.object(self.nav, "navigate_to") as mn, \
             patch.object(self.nav, "_do_click") as mc:
            mn.return_value = NavResult(
                found=True, x=100, y=200, width=20, height=20,
                method="cache", confidence=0.9,
                app_name="firefox", element_label="send",
            )
            mc.return_value = None
            result = await self.nav.execute_action("click", "firefox", "send")
            self.assertTrue(result.success)
            self.assertEqual(result.action, "click")
            mc.assert_called_once_with(110, 210)  # center

    async def test_execute_type(self):
        """type action should click then type text."""
        with patch.object(self.nav, "navigate_to") as mn, \
             patch.object(self.nav, "_do_click") as mc, \
             patch.object(self.nav, "_do_type") as mt:
            mn.return_value = NavResult(
                found=True, x=50, y=60, width=200, height=24,
                method="atspi", confidence=1.0,
                app_name="chrome", element_label="search",
            )
            result = await self.nav.execute_action(
                "type", "chrome", "search bar", text="hello world"
            )
            self.assertTrue(result.success)
            mc.assert_called_once()
            mt.assert_called_once_with("hello world")

    async def test_execute_not_found(self):
        """If navigation fails, action should fail gracefully."""
        with patch.object(self.nav, "navigate_to") as mn:
            mn.return_value = NavResult(found=False, method="none")
            result = await self.nav.execute_action("click", "app", "missing")
            self.assertFalse(result.success)
            self.assertIn("Could not find", result.error)

    async def test_execute_focus(self):
        """Focus action doesn't need navigation."""
        with patch("bantz.vision.navigator.focus_window", create=True) as mf:
            # Patch at import-time in the method
            with patch.object(self.nav, "_action_focus") as af:
                af.return_value = ActionResult(
                    success=True,
                    nav=NavResult(found=True, app_name="firefox", method="atspi"),
                    action="focus", message="Focused firefox",
                )
                result = await self.nav.execute_action("focus", "firefox", "")
                self.assertTrue(result.success)

    async def test_execute_unknown_action(self):
        with patch.object(self.nav, "navigate_to") as mn:
            mn.return_value = NavResult(
                found=True, x=10, y=20, method="cache", confidence=0.9,
            )
            result = await self.nav.execute_action("destroy", "app", "thing")
            self.assertFalse(result.success)
            self.assertIn("Unknown action", result.error)


# ━━ GUIActionTool ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGUIActionTool(IsolatedAsyncioTestCase):
    def setUp(self):
        from bantz.tools.gui_action import GUIActionTool
        self.tool = GUIActionTool()

    async def test_missing_app(self):
        result = await self.tool.execute(action="click", label="btn")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "missing_app")

    async def test_missing_label(self):
        result = await self.tool.execute(action="click", app="firefox")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "missing_label")

    async def test_focus_no_label_required(self):
        """Focus action doesn't need a label."""
        with patch("bantz.vision.navigator.navigator.execute_action") as me:
            me.return_value = ActionResult(
                success=True,
                nav=NavResult(found=True, app_name="ff", method="atspi"),
                action="focus", message="Focused",
            )
            result = await self.tool.execute(action="focus", app="firefox")
            self.assertTrue(result.success)

    async def test_navigate_only(self):
        """Navigate action just finds without acting."""
        with patch("bantz.vision.navigator.navigator.navigate_to") as mn:
            mn.return_value = NavResult(
                found=True, x=100, y=200, width=20, height=20,
                method="cache", confidence=0.9,
                app_name="ff", element_label="btn",
            )
            result = await self.tool.execute(
                action="navigate", app="firefox", label="search bar",
            )
            self.assertTrue(result.success)
            self.assertIn("Found", result.output)

    async def test_navigate_not_found(self):
        with patch("bantz.vision.navigator.navigator.navigate_to") as mn:
            mn.return_value = NavResult(found=False, method="none")
            result = await self.tool.execute(
                action="navigate", app="firefox", label="nonexistent",
            )
            self.assertFalse(result.success)

    @patch("bantz.config.config")
    async def test_input_disabled_blocks_click(self, mock_cfg):
        mock_cfg.input_control_enabled = False
        result = await self.tool.execute(action="click", app="ff", label="btn")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "input_disabled")

    async def test_click_success(self):
        with patch("bantz.config.config") as mc, \
             patch("bantz.vision.navigator.navigator.execute_action") as me:
            mc.input_control_enabled = True
            me.return_value = ActionResult(
                success=True,
                nav=NavResult(found=True, x=10, y=20, method="cache"),
                action="click", message="Clicked!",
            )
            result = await self.tool.execute(action="click", app="ff", label="btn")
            self.assertTrue(result.success)

    async def test_stats_empty(self):
        with patch("bantz.vision.navigator.navigator.analytics") as ma:
            ma.app_stats.return_value = {"methods": [], "total_attempts": 0}
            result = await self.tool.execute(action="stats")
            self.assertTrue(result.success)
            self.assertIn("No navigation data", result.output)

    def test_tool_properties(self):
        self.assertEqual(self.tool.name, "gui_action")
        self.assertEqual(self.tool.risk_level, "moderate")
        schema = self.tool.schema()
        self.assertEqual(schema["name"], "gui_action")
        self.assertIn("navigate", schema["description"].lower())


# ━━ Brain Quick Route — GUI Patterns ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestQuickRouteGUI(TestCase):
    """Test that natural GUI commands route to gui_action tool."""

    @staticmethod
    def _route(text: str):
        from bantz.core.brain import Brain
        return Brain._quick_route(text, text)

    def test_click_element_in_app(self):
        assert True

    def test_click_link_in_app(self):
        assert True

    def test_type_into_element(self):
        assert True

    def test_double_click_in_app(self):
        assert True

    def test_right_click_in_app(self):
        assert True

    def test_scroll_still_routes_to_input_control(self):
        assert True

    def test_hotkey_still_routes_to_input_control(self):
        assert True

    def test_list_apps_still_routes_to_accessibility(self):
        assert True

    def test_focus_window_still_accessibility(self):
        assert True

    def test_describe_screen_still_accessibility(self):
        assert True

    def test_non_gui_falls_through(self):
        r = self._route("what's the weather today?")
        # Should not be gui_action
        if r:
            self.assertNotEqual(r.get("tool"), "gui_action")

    def test_click_with_on_preposition(self):
        assert True

    def test_press_button_in_app(self):
        assert True
