"""Tests for bantz.core.notification_manager (#225)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip('textual')

from bantz.core import notification_manager


class TestNotifyToast:
    """Unit tests for the notify_toast routing logic."""

    def test_callback_takes_priority(self):
        calls: list[tuple] = []
        old = notification_manager.toast_callback
        try:
            notification_manager.toast_callback = lambda t, r, tt: calls.append((t, r, tt))
            notification_manager.notify_toast("title", "reason", "warning")
            assert calls == [("title", "reason", "warning")]
        finally:
            notification_manager.toast_callback = old

    def test_no_crash_without_callback(self):
        old = notification_manager.toast_callback
        try:
            notification_manager.toast_callback = None
            # Should not raise even without callback or App
            notification_manager.notify_toast("test", "reason", "info")
        finally:
            notification_manager.toast_callback = old

    def test_fallback_to_notifier(self):
        old = notification_manager.toast_callback
        notification_manager.toast_callback = None
        mock_notifier = MagicMock()
        mock_notifier.enabled = True
        try:
            with patch("bantz.agent.notifier.notifier", mock_notifier):
                # Suppress App.current failure
                with patch("textual.app.App") as mock_app_cls:
                    type(mock_app_cls).current = property(lambda self: None)
                    notification_manager.notify_toast("Title", "Body", "info")
            mock_notifier.send.assert_called_once()
            assert "Title" in mock_notifier.send.call_args[0][0]
        finally:
            notification_manager.toast_callback = old

    def test_push_toast_alias(self):
        calls: list[tuple] = []
        old = notification_manager.toast_callback
        try:
            notification_manager.toast_callback = lambda t, r, tt: calls.append((t, r, tt))
            notification_manager.push_toast("hello", "world", "success")
            assert calls == [("hello", "world", "success")]
        finally:
            notification_manager.toast_callback = old


class TestBrainBackwardCompat:
    """brain._toast_callback and brain._notify_toast must still work."""

    def test_brain_module_exposes_toast_callback(self):
        from bantz.core import brain as brain_mod
        assert hasattr(brain_mod, "_toast_callback")

    def test_brain_module_exposes_notify_toast(self):
        from bantz.core import brain as brain_mod
        assert hasattr(brain_mod, "_notify_toast")
        assert callable(brain_mod._notify_toast)

    def test_brain_callback_write_flows_to_notification_manager(self):
        """Setting brain_mod._toast_callback then calling _notify_toast
        should route through notification_manager with that callback."""
        from bantz.core import brain as brain_mod
        calls: list[tuple] = []
        old = brain_mod._toast_callback
        try:
            brain_mod._toast_callback = lambda t, r, tt: calls.append((t, r, tt))
            brain_mod._notify_toast("via brain", "compat", "info")
            assert len(calls) == 1
            assert calls[0] == ("via brain", "compat", "info")
        finally:
            brain_mod._toast_callback = old

    def test_brain_push_toast_method(self):
        from bantz.core.brain import Brain
        assert hasattr(Brain, "_push_toast")

    def test_brain_push_toast_delegates(self):
        from bantz.core.brain import Brain
        from bantz.core import brain as brain_mod
        calls: list[tuple] = []
        old = brain_mod._toast_callback
        try:
            brain_mod._toast_callback = lambda t, r, tt: calls.append((t, r, tt))
            b = Brain.__new__(Brain)
            b._push_toast("hello", "world", "success")
            # push_toast goes through notification_manager directly,
            # not through brain_mod._notify_toast, so set the canonical one too
            notification_manager.toast_callback = brain_mod._toast_callback
            b._push_toast("hello2", "world2", "success")
            assert ("hello2", "world2", "success") in calls
        finally:
            brain_mod._toast_callback = old
            notification_manager.toast_callback = None

    def test_deprecated_methods_removed(self):
        """_check_intervention_queue and _prepend_intervention are removed."""
        from bantz.core.brain import Brain
        assert not hasattr(Brain, "_check_intervention_queue")
        assert not hasattr(Brain, "_prepend_intervention")
