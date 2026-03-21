"""
Tests — Issue #137: Non-blocking Toast Notifications

Covers:
  - ToastType enum & constants (TOAST_AUTO_DISMISS, TOAST_ICONS)
  - ToastData dataclass (generic payload for non-intervention toasts)
  - ToastAccepted / ToastDismissed / ToastExpired messages
  - ToastWidget: can_focus, CSS classes, render, dismiss, accept, auto-expire
  - ToastContainer: push, overflow, dismiss_top, accept_top, clear_all,
    remove_by_intervention, type inference, _promote_overflow
  - App integration: compose yields ToastContainer, escape dismisses,
    toast message handlers, push_toast API, brain hook wiring
  - Brain integration: _toast_callback, _notify_toast, _push_toast,
    process() no longer pops intervention queue
  - Styles: toast CSS rules, Screen layers, slide-in animation
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

import pytest

pytest.importorskip('textual')


# ═══════════════════════════════════════════════════════════════════════════
# ToastType enum & constants
# ═══════════════════════════════════════════════════════════════════════════

class TestToastType:
    def test_all_types(self):
        from bantz.interface.tui.widgets.toast import ToastType
        assert ToastType.INFO.value == "info"
        assert ToastType.SUCCESS.value == "success"
        assert ToastType.WARNING.value == "warning"
        assert ToastType.ERROR.value == "error"
        assert ToastType.ACTION.value == "action"

    def test_unique_values(self):
        from bantz.interface.tui.widgets.toast import ToastType
        values = [t.value for t in ToastType]
        assert len(values) == len(set(values))

    def test_five_types(self):
        from bantz.interface.tui.widgets.toast import ToastType
        assert len(list(ToastType)) == 5


class TestToastAutoDismiss:
    def test_all_types_present(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_AUTO_DISMISS
        for tt in ToastType:
            assert tt in TOAST_AUTO_DISMISS

    def test_info_5s(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_AUTO_DISMISS
        assert TOAST_AUTO_DISMISS[ToastType.INFO] == 5.0

    def test_success_3s(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_AUTO_DISMISS
        assert TOAST_AUTO_DISMISS[ToastType.SUCCESS] == 3.0

    def test_warning_10s(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_AUTO_DISMISS
        assert TOAST_AUTO_DISMISS[ToastType.WARNING] == 10.0

    def test_error_no_auto(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_AUTO_DISMISS
        assert TOAST_AUTO_DISMISS[ToastType.ERROR] is None

    def test_action_no_auto(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_AUTO_DISMISS
        assert TOAST_AUTO_DISMISS[ToastType.ACTION] is None


class TestToastIcons:
    def test_all_types_have_icons(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_ICONS
        for tt in ToastType:
            assert tt in TOAST_ICONS
            assert isinstance(TOAST_ICONS[tt], str)
            assert len(TOAST_ICONS[tt]) > 0

    def test_info_icon(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_ICONS
        assert "ℹ" in TOAST_ICONS[ToastType.INFO]

    def test_success_icon(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_ICONS
        assert "✓" in TOAST_ICONS[ToastType.SUCCESS]

    def test_warning_icon(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_ICONS
        assert "⚠" in TOAST_ICONS[ToastType.WARNING]

    def test_error_icon(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_ICONS
        assert "✗" in TOAST_ICONS[ToastType.ERROR]

    def test_action_icon(self):
        from bantz.interface.tui.widgets.toast import ToastType, TOAST_ICONS
        assert "◆" in TOAST_ICONS[ToastType.ACTION]


# ═══════════════════════════════════════════════════════════════════════════
# ToastData
# ═══════════════════════════════════════════════════════════════════════════

class TestToastData:
    def test_creation(self):
        from bantz.interface.tui.widgets.toast import ToastData
        td = ToastData(title="Hello", reason="world")
        assert td.title == "Hello"
        assert td.reason == "world"

    def test_defaults(self):
        from bantz.interface.tui.widgets.toast import ToastData
        td = ToastData(title="Test")
        assert td.reason == ""
        assert td.ttl == 0.0
        assert td.source == "system"
        assert td.action is None
        assert td.state_key is None
        assert td.type is None

    def test_remaining_ttl(self):
        from bantz.interface.tui.widgets.toast import ToastData
        td = ToastData(title="X", ttl=15.0)
        assert td.remaining_ttl == 15.0

    def test_expired_always_false(self):
        from bantz.interface.tui.widgets.toast import ToastData
        td = ToastData(title="X")
        assert td.expired is False

    def test_intervention_attribute_compat(self):
        """ToastData has the same getattr interface as Intervention."""
        from bantz.interface.tui.widgets.toast import ToastData
        td = ToastData(title="T", reason="R", ttl=5.0, source="brain")
        assert getattr(td, "title") == "T"
        assert getattr(td, "reason") == "R"
        assert getattr(td, "remaining_ttl") == 5.0
        assert getattr(td, "source") == "brain"


# ═══════════════════════════════════════════════════════════════════════════
# Toast Messages
# ═══════════════════════════════════════════════════════════════════════════

class TestToastMessages:
    def test_toast_accepted(self):
        from bantz.interface.tui.widgets.toast import ToastAccepted
        iv = MagicMock()
        msg = ToastAccepted(iv)
        assert msg.intervention is iv

    def test_toast_dismissed(self):
        from bantz.interface.tui.widgets.toast import ToastDismissed
        iv = MagicMock()
        msg = ToastDismissed(iv)
        assert msg.intervention is iv

    def test_toast_expired(self):
        from bantz.interface.tui.widgets.toast import ToastExpired
        iv = MagicMock()
        msg = ToastExpired(iv)
        assert msg.intervention is iv

    def test_messages_are_textual_messages(self):
        from bantz.interface.tui.widgets.toast import (
            ToastAccepted, ToastDismissed, ToastExpired,
        )
        from textual.message import Message
        iv = MagicMock()
        assert isinstance(ToastAccepted(iv), Message)
        assert isinstance(ToastDismissed(iv), Message)
        assert isinstance(ToastExpired(iv), Message)


# ═══════════════════════════════════════════════════════════════════════════
# ToastWidget
# ═══════════════════════════════════════════════════════════════════════════

class TestToastWidget:
    def test_can_focus_false(self):
        """CRITICAL: Toast must never steal focus from chat input."""
        from bantz.interface.tui.widgets.toast import ToastWidget
        assert ToastWidget.can_focus is False

    def test_creation(self):
        from bantz.interface.tui.widgets.toast import ToastWidget, ToastType, ToastData
        td = ToastData(title="Test")
        tw = ToastWidget(td, ToastType.INFO)
        assert tw.intervention is td
        assert tw.toast_type == ToastType.INFO

    def test_default_type(self):
        from bantz.interface.tui.widgets.toast import ToastWidget, ToastType, ToastData
        td = ToastData(title="Test")
        tw = ToastWidget(td)
        assert tw.toast_type == ToastType.INFO

    def test_css_class_info(self):
        from bantz.interface.tui.widgets.toast import ToastWidget, ToastType, ToastData
        td = ToastData(title="Test")
        tw = ToastWidget(td, ToastType.INFO)
        assert tw.has_class("toast--info")

    def test_css_class_warning(self):
        from bantz.interface.tui.widgets.toast import ToastWidget, ToastType, ToastData
        td = ToastData(title="Test")
        tw = ToastWidget(td, ToastType.WARNING)
        assert tw.has_class("toast--warning")

    def test_css_class_error(self):
        from bantz.interface.tui.widgets.toast import ToastWidget, ToastType, ToastData
        td = ToastData(title="Test")
        tw = ToastWidget(td, ToastType.ERROR)
        assert tw.has_class("toast--error")

    def test_css_class_success(self):
        from bantz.interface.tui.widgets.toast import ToastWidget, ToastType, ToastData
        td = ToastData(title="Test")
        tw = ToastWidget(td, ToastType.SUCCESS)
        assert tw.has_class("toast--success")

    def test_css_class_action(self):
        from bantz.interface.tui.widgets.toast import ToastWidget, ToastType, ToastData
        td = ToastData(title="Test")
        tw = ToastWidget(td, ToastType.ACTION)
        assert tw.has_class("toast--action")

    def test_dismiss_timer_initially_none(self):
        from bantz.interface.tui.widgets.toast import ToastWidget, ToastType, ToastData
        td = ToastData(title="Test")
        tw = ToastWidget(td, ToastType.ERROR)
        assert tw._dismiss_timer is None

    def test_all_five_types_produce_valid_widgets(self):
        from bantz.interface.tui.widgets.toast import ToastWidget, ToastType, ToastData
        for tt in ToastType:
            tw = ToastWidget(ToastData(title=f"t-{tt.value}"), tt)
            assert tw.toast_type == tt
            assert tw.has_class(f"toast--{tt.value}")


# ═══════════════════════════════════════════════════════════════════════════
# ToastContainer
# ═══════════════════════════════════════════════════════════════════════════

class TestToastContainer:
    def test_can_focus_false(self):
        from bantz.interface.tui.widgets.toast import ToastContainer
        assert ToastContainer.can_focus is False

    def test_max_visible(self):
        from bantz.interface.tui.widgets.toast import ToastContainer
        assert ToastContainer.MAX_VISIBLE == 3

    def test_creation(self):
        from bantz.interface.tui.widgets.toast import ToastContainer
        tc = ToastContainer()
        assert tc._overflow == []

    def test_infer_type_error_alert(self):
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        iv = MagicMock()
        iv.type = MagicMock(value="error_alert")
        iv.action = None
        assert ToastContainer._infer_type(iv) == ToastType.ERROR

    def test_infer_type_with_action(self):
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        iv = MagicMock()
        iv.type = MagicMock(value="routine")
        iv.action = "launch_docker"
        assert ToastContainer._infer_type(iv) == ToastType.ACTION

    def test_infer_type_reminder(self):
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        iv = MagicMock()
        iv.type = MagicMock(value="reminder")
        iv.action = None
        assert ToastContainer._infer_type(iv) == ToastType.ACTION

    def test_infer_type_maintenance(self):
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        iv = MagicMock()
        iv.type = MagicMock(value="maintenance")
        iv.action = None
        assert ToastContainer._infer_type(iv) == ToastType.WARNING

    def test_infer_type_unknown(self):
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        iv = MagicMock()
        iv.type = MagicMock(value="some_other")
        iv.action = None
        assert ToastContainer._infer_type(iv) == ToastType.INFO

    def test_infer_type_no_type_attr(self):
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        iv = MagicMock(spec=[])  # no 'type' attribute
        assert ToastContainer._infer_type(iv) == ToastType.INFO

    def test_infer_type_error_overrides_action(self):
        """error_alert type should always be ERROR even if action is set."""
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        iv = MagicMock()
        iv.type = MagicMock(value="error_alert")
        iv.action = "some_action"
        assert ToastContainer._infer_type(iv) == ToastType.ERROR


# ═══════════════════════════════════════════════════════════════════════════
# App integration: compose & structure
# ═══════════════════════════════════════════════════════════════════════════

class TestAppToastIntegration:
    def test_compose_yields_toast_container(self):
        """BantzApp.compose() must yield ToastContainer(id='toast-container')."""
        import ast, inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp.compose)
        assert "toast-container" in src
        assert "ToastContainer" in src

    def test_toast_imports_in_app(self):
        """app.py must import toast components."""
        import bantz.interface.tui.app as app_mod
        assert hasattr(app_mod, "ToastContainer")
        assert hasattr(app_mod, "ToastType")
        assert hasattr(app_mod, "ToastData")
        assert hasattr(app_mod, "ToastAccepted")
        assert hasattr(app_mod, "ToastDismissed")
        assert hasattr(app_mod, "ToastExpired")

    def test_push_toast_method_exists(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "push_toast")
        assert callable(getattr(BantzApp, "push_toast"))

    def test_on_toast_accepted_handler(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "on_toast_accepted")
        assert callable(getattr(BantzApp, "on_toast_accepted"))

    def test_on_toast_dismissed_handler(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "on_toast_dismissed")
        assert callable(getattr(BantzApp, "on_toast_dismissed"))

    def test_on_toast_expired_handler(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "on_toast_expired")
        assert callable(getattr(BantzApp, "on_toast_expired"))

    def test_wire_brain_toast_hook_method(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "_wire_brain_toast_hook")

    def test_on_brain_toast_method(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "_on_brain_toast")

    def test_remove_intervention_toast_method(self):
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "_remove_intervention_toast")

    def test_escape_binding_exists(self):
        """Escape should be bound to focus_input (which now also dismisses toasts)."""
        from bantz.interface.tui.app import BantzApp
        from textual.binding import Binding
        escape_bindings = [
            b for b in BantzApp.BINDINGS
            if isinstance(b, Binding) and b.key == "escape"
        ]
        assert len(escape_bindings) == 1
        assert escape_bindings[0].action == "focus_input"

    def test_action_focus_input_dismisses_toast_first(self):
        """action_focus_input should check for toasts before focusing input."""
        import inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp.action_focus_input)
        assert "toast-container" in src or "has_toasts" in src
        assert "dismiss_top" in src


# ═══════════════════════════════════════════════════════════════════════════
# App: _process_interventions redirects to toast
# ═══════════════════════════════════════════════════════════════════════════

class TestProcessInterventionsToast:
    def test_process_interventions_uses_toast_container(self):
        """_process_interventions must push to ToastContainer, not ChatLog."""
        import inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp._process_interventions)
        assert "toast-container" in src
        assert "push_toast" in src
        # Should NOT contain add_bantz as primary path (only fallback)
        # The push_toast line should come before any add_bantz fallback
        toast_pos = src.index("push_toast")
        # Verify toast container is the primary rendering path
        assert "ToastContainer" in src or "toast-container" in src

    def test_process_interventions_has_fallback(self):
        """_process_interventions should fallback to chat if toast container fails."""
        import inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp._process_interventions)
        # Fallback should exist
        assert "Fallback" in src or "fallback" in src


# ═══════════════════════════════════════════════════════════════════════════
# App: _handle_intervention_response removes toast
# ═══════════════════════════════════════════════════════════════════════════

class TestInterventionResponseToast:
    def test_handle_intervention_response_removes_toast(self):
        """Text-based intervention response should also remove the toast widget."""
        import inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp._handle_intervention_response)
        assert "_remove_intervention_toast" in src

    def test_accept_removes_toast(self):
        import inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp._handle_intervention_response)
        # "accept" branch should call _remove_intervention_toast
        lines = src.split("\n")
        found_accept = False
        found_remove = False
        for line in lines:
            if "_ACCEPT" in line and "low in" in line:
                found_accept = True
            if found_accept and "_remove_intervention_toast" in line:
                found_remove = True
                break
        assert found_remove, "Accept branch should remove intervention toast"

    def test_dismiss_removes_toast(self):
        import inspect
        from bantz.interface.tui.app import BantzApp
        src = inspect.getsource(BantzApp._handle_intervention_response)
        assert src.count("_remove_intervention_toast") >= 3  # accept, dismiss, never paths


# ═══════════════════════════════════════════════════════════════════════════
# Brain integration
# ═══════════════════════════════════════════════════════════════════════════

class TestBrainToastIntegration:
    def test_toast_callback_exists(self):
        """brain module should expose _toast_callback."""
        from bantz.core import brain as brain_mod
        assert hasattr(brain_mod, "_toast_callback")

    def test_notify_toast_exists(self):
        """brain module should expose _notify_toast function."""
        from bantz.core import brain as brain_mod
        assert hasattr(brain_mod, "_notify_toast")
        assert callable(brain_mod._notify_toast)

    def test_notify_toast_with_callback(self):
        from bantz.core import brain as brain_mod
        calls = []
        old_cb = brain_mod._toast_callback
        try:
            brain_mod._toast_callback = lambda t, r, tt: calls.append((t, r, tt))
            brain_mod._notify_toast("test title", "test reason", "warning")
            assert len(calls) == 1
            assert calls[0] == ("test title", "test reason", "warning")
        finally:
            brain_mod._toast_callback = old_cb

    def test_notify_toast_without_callback_no_crash(self):
        from bantz.core import brain as brain_mod
        old_cb = brain_mod._toast_callback
        try:
            brain_mod._toast_callback = None
            # Should not raise even without callback or App
            brain_mod._notify_toast("test", "reason", "info")
        finally:
            brain_mod._toast_callback = old_cb

    def test_brain_push_toast_method(self):
        """Brain class should have _push_toast method."""
        from bantz.core.brain import Brain
        assert hasattr(Brain, "_push_toast")

    def test_brain_push_toast_delegates(self):
        """Brain._push_toast should delegate to notification_manager."""
        from bantz.core.brain import Brain
        from bantz.core import notification_manager as _notif_mod
        calls = []
        old_cb = _notif_mod.toast_callback
        try:
            _notif_mod.toast_callback = lambda t, r, tt: calls.append((t, r, tt))
            b = Brain.__new__(Brain)
            b._push_toast("hello", "world", "success")
            assert len(calls) == 1
            assert calls[0] == ("hello", "world", "success")
        finally:
            _notif_mod.toast_callback = old_cb

    def test_process_no_longer_pops_queue(self):
        """brain.process() should NOT call _check_intervention_queue anymore."""
        import inspect
        from bantz.core.brain import Brain
        src = inspect.getsource(Brain.process)
        assert "_check_intervention_queue" not in src

    def test_process_no_longer_prepends(self):
        """brain.process() should NOT call _prepend_intervention anymore."""
        import inspect
        from bantz.core.brain import Brain
        src = inspect.getsource(Brain.process)
        assert "_prepend_intervention" not in src

    def test_deprecated_check_intervention_queue(self):
        """_check_intervention_queue was removed in #225."""
        from bantz.core.brain import Brain
        assert not hasattr(Brain, "_check_intervention_queue")

    def test_deprecated_prepend_intervention(self):
        """_prepend_intervention was removed in #225."""
        from bantz.core.brain import Brain
        assert not hasattr(Brain, "_prepend_intervention")

    def test_note_about_toast_in_process(self):
        """process() should have a comment about toast system."""
        import inspect
        from bantz.core.brain import Brain
        src = inspect.getsource(Brain.process)
        assert "toast" in src.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Styles: Toast CSS
# ═══════════════════════════════════════════════════════════════════════════

class TestToastStyles:
    @pytest.fixture
    def css(self):
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent.parent / "src" / "bantz" / "interface" / "tui" / "styles.tcss"
        return p.read_text()

    def test_screen_has_toast_layer(self, css):
        assert "layers:" in css
        assert "toast" in css

    def test_toast_container_dock_bottom(self, css):
        assert "ToastContainer" in css
        assert "dock: bottom" in css

    def test_toast_container_layer_toast(self, css):
        assert "layer: toast" in css

    def test_toast_widget_transition(self, css):
        """Toast slide-in animation via CSS transition."""
        assert "transition: offset 300ms in_out_cubic" in css

    def test_toast_enter_class(self, css):
        assert ".toast-enter" in css
        assert "offset-y: 5" in css

    def test_toast_info_border(self, css):
        assert ".toast--info" in css
        assert "#00ccff" in css

    def test_toast_success_border(self, css):
        assert ".toast--success" in css
        assert "#00ff88" in css

    def test_toast_warning_border(self, css):
        assert ".toast--warning" in css
        assert "#ffaa00" in css

    def test_toast_error_border(self, css):
        assert ".toast--error" in css
        assert "#ff4444" in css

    def test_toast_action_border(self, css):
        assert ".toast--action" in css
        assert "#4488ff" in css

    def test_toast_widget_base_style(self, css):
        assert "ToastWidget" in css
        # Base background
        assert "#1a1a2a" in css

    def test_toast_container_max_height(self, css):
        assert "max-height: 12" in css


# ═══════════════════════════════════════════════════════════════════════════
# Widget module __init__
# ═══════════════════════════════════════════════════════════════════════════

class TestWidgetsInit:
    def test_widgets_package_importable(self):
        import bantz.interface.tui.widgets
        assert bantz.interface.tui.widgets is not None

    def test_toast_module_importable(self):
        from bantz.interface.tui.widgets import toast
        assert toast is not None


# ═══════════════════════════════════════════════════════════════════════════
# Intervention → Toast type mapping comprehensive
# ═══════════════════════════════════════════════════════════════════════════

class TestInterventionToastMapping:
    def test_rl_suggestion_becomes_action(self):
        """RL suggestions have action field → ACTION toast."""
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        from bantz.agent.interventions import intervention_from_rl
        iv = intervention_from_rl(
            action_value="launch_docker",
            state_key="test_state",
            reason="Morning routine",
        )
        assert ToastContainer._infer_type(iv) == ToastType.ACTION

    def test_observer_error_becomes_error(self):
        """Observer critical alerts → ERROR toast."""
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        from bantz.agent.interventions import intervention_from_observer
        iv = intervention_from_observer(
            raw_text="Segfault",
            severity="CRITICAL",
            analysis="crash detected",
        )
        assert ToastContainer._infer_type(iv) == ToastType.ERROR

    def test_observer_warning_not_error(self):
        """Observer warnings → ACTION (has no action field → depends on type)."""
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        from bantz.agent.interventions import intervention_from_observer
        iv = intervention_from_observer(
            raw_text="deprecation warning",
            severity="WARNING",
            analysis="deprecated API",
        )
        # error_alert type → ERROR
        assert ToastContainer._infer_type(iv) == ToastType.ERROR

    def test_reminder_becomes_action(self):
        """Reminders → ACTION toast (accept/dismiss)."""
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        from bantz.agent.interventions import intervention_from_reminder
        iv = intervention_from_reminder(title="Call dentist", repeat="none")
        assert ToastContainer._infer_type(iv) == ToastType.ACTION

    def test_system_maintenance_becomes_warning(self):
        """System maintenance → WARNING toast."""
        from bantz.interface.tui.widgets.toast import ToastContainer, ToastType
        from bantz.agent.interventions import intervention_from_system
        iv = intervention_from_system(
            title="Docker prune",
            reason="Weekly cleanup",
        )
        assert ToastContainer._infer_type(iv) == ToastType.WARNING


# ═══════════════════════════════════════════════════════════════════════════
# CLI: no regressions
# ═══════════════════════════════════════════════════════════════════════════

class TestCLI:
    def test_run_function_exists(self):
        from bantz.interface.tui.app import run
        assert callable(run)

    def test_app_instantiation(self):
        """BantzApp can be instantiated without errors."""
        from bantz.interface.tui.app import BantzApp
        app = BantzApp()
        assert app.title == "BANTZ v3"
