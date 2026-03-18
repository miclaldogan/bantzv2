"""
Tests — #220 Sprint 3 Part 3: EventBus → TUI Bridge

Covers:
  1. BantzEventMessage — Textual Message wrapping an Event
  2. _subscribe_event_bus / _unsubscribe_event_bus lifecycle
  3. _relay_bus_event — call_from_thread + post_message bridge
  4. on_bantz_event_message — event name dispatching
  5. _on_bus_wake_word — chat message + input focus
  6. _on_bus_ambient_change — panel update (with/without update_ambient)
  7. _on_bus_health_alert — push_toast with warning
  8. _start_wake_word_listener — no longer passes _on_wake callback
  9. action_quit — unsubscribes + shuts down bus
 10. Source audit — import and structural checks
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call

import pytest

from bantz.core.event_bus import Event, EventBus


# ═══════════════════════════════════════════════════════════════════════════
# 1. BantzEventMessage
# ═══════════════════════════════════════════════════════════════════════════

class TestBantzEventMessage:
    def test_import(self):
        from bantz.interface.tui.app import BantzEventMessage
        assert BantzEventMessage is not None

    def test_is_textual_message(self):
        from bantz.interface.tui.app import BantzEventMessage
        from textual.message import Message
        assert issubclass(BantzEventMessage, Message)

    def test_wraps_event(self):
        from bantz.interface.tui.app import BantzEventMessage
        ev = Event(name="test_event", data={"k": 42})
        msg = BantzEventMessage(ev)
        assert msg.event is ev
        assert msg.event.name == "test_event"
        assert msg.event.data["k"] == 42

    def test_different_events(self):
        from bantz.interface.tui.app import BantzEventMessage
        ev1 = Event(name="a")
        ev2 = Event(name="b", data={"x": 1})
        m1 = BantzEventMessage(ev1)
        m2 = BantzEventMessage(ev2)
        assert m1.event.name != m2.event.name

    def test_event_data_preserved(self):
        from bantz.interface.tui.app import BantzEventMessage
        data = {"label": "loud", "rms": 420.5, "zcr": 0.12}
        ev = Event(name="ambient_change", data=data)
        msg = BantzEventMessage(ev)
        assert msg.event.data == data


# ═══════════════════════════════════════════════════════════════════════════
# 2. WakeWordDetected (removed — legacy dead code cleaned up)
# ═══════════════════════════════════════════════════════════════════════════

class TestWakeWordDetectedLegacy:
    def test_legacy_class_removed(self):
        """WakeWordDetected was dead code — verify it's been removed."""
        from bantz.interface.tui import app as tui_app
        assert not hasattr(tui_app, "WakeWordDetected")

    def test_bus_wake_handler_exists(self):
        """The canonical wake word handler is via EventBus bridge."""
        from bantz.interface.tui.app import BantzApp
        assert hasattr(BantzApp, "_on_bus_wake_word")


# ═══════════════════════════════════════════════════════════════════════════
# 3. _subscribe_event_bus
# ═══════════════════════════════════════════════════════════════════════════

class TestSubscribeEventBus:
    def _make_app(self):
        """Create a BantzApp with mocked-out Textual internals."""
        from bantz.interface.tui.app import BantzApp
        app = object.__new__(BantzApp)
        # Minimal mocks so call_from_thread / post_message don't crash
        app.call_from_thread = MagicMock()
        app.post_message = MagicMock()
        return app

    def test_subscribes_three_events(self):
        app = self._make_app()
        test_bus = EventBus()
        with patch("bantz.interface.tui.app.bus", test_bus):
            test_bus.bind_loop = MagicMock()
            app._subscribe_event_bus()
            assert test_bus.subscriber_count("wake_word_detected") == 1
            assert test_bus.subscriber_count("ambient_change") == 1
            assert test_bus.subscriber_count("health_alert") == 1
            assert test_bus.subscriber_count("voice_input") == 1
            assert test_bus.subscriber_count("ghost_loop_listening") == 1
            assert test_bus.subscriber_count("ghost_loop_transcribing") == 1
            assert test_bus.subscriber_count("ghost_loop_idle") == 1
            assert test_bus.subscriber_count("voice_no_speech") == 1
            assert test_bus.subscriber_count("stt_model_loading") == 1
            assert test_bus.subscriber_count("stt_model_ready") == 1
            assert test_bus.subscriber_count("stt_model_failed") == 1

    def test_calls_bind_loop(self):
        app = self._make_app()
        test_bus = EventBus()
        test_bus.bind_loop = MagicMock()
        with patch("bantz.interface.tui.app.bus", test_bus):
            app._subscribe_event_bus()
            test_bus.bind_loop.assert_called_once()

    def test_stores_relay_ref(self):
        app = self._make_app()
        test_bus = EventBus()
        test_bus.bind_loop = MagicMock()
        with patch("bantz.interface.tui.app.bus", test_bus):
            app._subscribe_event_bus()
            assert hasattr(app, "_bus_relay")
            assert callable(app._bus_relay)


# ═══════════════════════════════════════════════════════════════════════════
# 4. _unsubscribe_event_bus
# ═══════════════════════════════════════════════════════════════════════════

class TestUnsubscribeEventBus:
    def _make_subscribed_app(self):
        from bantz.interface.tui.app import BantzApp
        app = object.__new__(BantzApp)
        app.call_from_thread = MagicMock()
        app.post_message = MagicMock()
        return app

    def test_removes_subscriptions(self):
        app = self._make_subscribed_app()
        test_bus = EventBus()
        test_bus.bind_loop = MagicMock()
        with patch("bantz.interface.tui.app.bus", test_bus):
            app._subscribe_event_bus()
            assert test_bus.subscriber_count() == 16
            app._unsubscribe_event_bus()
            assert test_bus.subscriber_count("wake_word_detected") == 0
            assert test_bus.subscriber_count("ambient_change") == 0
            assert test_bus.subscriber_count("health_alert") == 0
            assert test_bus.subscriber_count("voice_input") == 0
            assert test_bus.subscriber_count("ghost_loop_listening") == 0
            assert test_bus.subscriber_count("ghost_loop_transcribing") == 0
            assert test_bus.subscriber_count("ghost_loop_idle") == 0
            assert test_bus.subscriber_count("voice_no_speech") == 0
            assert test_bus.subscriber_count("stt_model_loading") == 0
            assert test_bus.subscriber_count("stt_model_ready") == 0
            assert test_bus.subscriber_count("stt_model_failed") == 0
            assert test_bus.subscriber_count("thinking_start") == 0
            assert test_bus.subscriber_count("thinking_token") == 0
            assert test_bus.subscriber_count("thinking_done") == 0
            assert test_bus.subscriber_count("planner_step") == 0

    def test_safe_when_not_subscribed(self):
        """No crash if _unsubscribe called before _subscribe."""
        app = self._make_subscribed_app()
        test_bus = EventBus()
        with patch("bantz.interface.tui.app.bus", test_bus):
            app._unsubscribe_event_bus()  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# 5. _relay_bus_event
# ═══════════════════════════════════════════════════════════════════════════

class TestRelayBusEvent:
    def _make_app(self):
        from bantz.interface.tui.app import BantzApp, BantzEventMessage
        app = object.__new__(BantzApp)
        app.call_from_thread = MagicMock()
        app.post_message = MagicMock()
        return app

    def test_calls_call_from_thread(self):
        from bantz.interface.tui.app import BantzEventMessage
        app = self._make_app()
        ev = Event(name="test", data={"x": 1})
        app._relay_bus_event(ev)
        # _relay_bus_event calls post_message directly when on the main thread
        # (Textual v8: call_from_thread raises RuntimeError on main thread).
        app.post_message.assert_called_once()
        args = app.post_message.call_args
        msg = args[0][0]
        assert isinstance(msg, BantzEventMessage)
        assert msg.event is ev

    def test_swallows_exception(self):
        """If call_from_thread raises (app shutting down), no crash."""
        app = self._make_app()
        app.call_from_thread.side_effect = RuntimeError("app closed")
        ev = Event(name="test")
        app._relay_bus_event(ev)  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# 6. on_bantz_event_message — dispatching
# ═══════════════════════════════════════════════════════════════════════════

class TestOnBantzEventMessage:
    def _make_app(self):
        from bantz.interface.tui.app import BantzApp
        app = object.__new__(BantzApp)
        app._on_bus_wake_word = MagicMock()
        app._on_bus_ambient_change = MagicMock()
        app._on_bus_health_alert = MagicMock()
        return app

    def test_dispatches_wake_word(self):
        from bantz.interface.tui.app import BantzEventMessage
        app = self._make_app()
        ev = Event(name="wake_word_detected", data={"count": 5})
        msg = BantzEventMessage(ev)
        app.on_bantz_event_message(msg)
        app._on_bus_wake_word.assert_called_once_with(ev)
        app._on_bus_ambient_change.assert_not_called()
        app._on_bus_health_alert.assert_not_called()

    def test_dispatches_ambient(self):
        from bantz.interface.tui.app import BantzEventMessage
        app = self._make_app()
        ev = Event(name="ambient_change", data={"label": "loud", "rms": 300.0})
        msg = BantzEventMessage(ev)
        app.on_bantz_event_message(msg)
        app._on_bus_ambient_change.assert_called_once_with(ev)
        app._on_bus_wake_word.assert_not_called()

    def test_dispatches_health_alert(self):
        from bantz.interface.tui.app import BantzEventMessage
        app = self._make_app()
        ev = Event(name="health_alert", data={"title": "CPU Hot", "reason": "95°C"})
        msg = BantzEventMessage(ev)
        app.on_bantz_event_message(msg)
        app._on_bus_health_alert.assert_called_once_with(ev)

    def test_unknown_event_no_crash(self):
        from bantz.interface.tui.app import BantzEventMessage
        app = self._make_app()
        ev = Event(name="unknown_event")
        msg = BantzEventMessage(ev)
        app.on_bantz_event_message(msg)  # no dispatch, no crash
        app._on_bus_wake_word.assert_not_called()
        app._on_bus_ambient_change.assert_not_called()
        app._on_bus_health_alert.assert_not_called()

    def test_handler_error_swallowed(self):
        from bantz.interface.tui.app import BantzEventMessage
        app = self._make_app()
        app._on_bus_wake_word.side_effect = RuntimeError("boom")
        ev = Event(name="wake_word_detected")
        msg = BantzEventMessage(ev)
        app.on_bantz_event_message(msg)  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# 7. _on_bus_wake_word
# ═══════════════════════════════════════════════════════════════════════════

class TestOnBusWakeWord:
    def _make_app(self):
        from bantz.interface.tui.app import BantzApp
        app = object.__new__(BantzApp)
        return app

    def test_sets_voice_status_and_focuses(self):
        # Wake word no longer writes to chat — it updates the header voice status
        # bar and focuses the input.
        app = self._make_app()
        inp = MagicMock()
        app._set_voice_status = MagicMock()
        app.query_one = MagicMock(side_effect=lambda sel, cls=None: inp)
        ev = Event(name="wake_word_detected", data={"count": 3})
        app._on_bus_wake_word(ev)
        app._set_voice_status.assert_called_once_with("[bold green]🎤 Wake word...[/]")
        inp.focus.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# 8. _on_bus_ambient_change
# ═══════════════════════════════════════════════════════════════════════════

class TestOnBusAmbientChange:
    def _make_app(self):
        from bantz.interface.tui.app import BantzApp
        app = object.__new__(BantzApp)
        return app

    def test_calls_update_ambient_if_available(self):
        app = self._make_app()
        panel = MagicMock()
        panel.update_ambient = MagicMock()
        app.query_one = MagicMock(return_value=panel)
        ev = Event(name="ambient_change", data={"label": "loud", "rms": 420.5})
        app._on_bus_ambient_change(ev)
        panel.update_ambient.assert_called_once_with("loud", 420.5)

    def test_no_crash_without_update_ambient(self):
        """If SystemStatus has no update_ambient method, no crash."""
        app = self._make_app()
        panel = MagicMock(spec=[])  # no attributes
        app.query_one = MagicMock(return_value=panel)
        ev = Event(name="ambient_change", data={"label": "quiet"})
        app._on_bus_ambient_change(ev)  # should not raise

    def test_no_crash_on_query_failure(self):
        app = self._make_app()
        app.query_one = MagicMock(side_effect=Exception("no widget"))
        ev = Event(name="ambient_change", data={"label": "x"})
        app._on_bus_ambient_change(ev)  # should not raise

    def test_label_extraction(self):
        app = self._make_app()
        panel = MagicMock()
        panel.update_ambient = MagicMock()
        app.query_one = MagicMock(return_value=panel)
        ev = Event(name="ambient_change", data={"label": "speech", "rms": 123.4, "zcr": 0.3})
        app._on_bus_ambient_change(ev)
        label_arg = panel.update_ambient.call_args[0][0]
        rms_arg = panel.update_ambient.call_args[0][1]
        assert label_arg == "speech"
        assert rms_arg == 123.4


# ═══════════════════════════════════════════════════════════════════════════
# 9. _on_bus_health_alert
# ═══════════════════════════════════════════════════════════════════════════

class TestOnBusHealthAlert:
    def _make_app(self):
        from bantz.interface.tui.app import BantzApp
        app = object.__new__(BantzApp)
        app.push_toast = MagicMock()
        return app

    def test_pushes_warning_toast(self):
        app = self._make_app()
        ev = Event(name="health_alert", data={
            "title": "CPU Overheating",
            "reason": "Temperature 95°C",
            "rule_id": "cpu_temp",
        })
        app._on_bus_health_alert(ev)
        app.push_toast.assert_called_once_with("CPU Overheating", "Temperature 95°C", "warning")

    def test_defaults_when_missing(self):
        app = self._make_app()
        ev = Event(name="health_alert", data={})
        app._on_bus_health_alert(ev)
        app.push_toast.assert_called_once_with("Health Alert", "", "warning")

    def test_title_from_event_data(self):
        app = self._make_app()
        ev = Event(name="health_alert", data={"title": "RAM Full", "reason": "96% used"})
        app._on_bus_health_alert(ev)
        args = app.push_toast.call_args[0]
        assert args[0] == "RAM Full"
        assert args[1] == "96% used"
        assert args[2] == "warning"


# ═══════════════════════════════════════════════════════════════════════════
# 10. _start_wake_word_listener — no legacy callback
# ═══════════════════════════════════════════════════════════════════════════

class TestStartWakeWordListenerRefactored:
    def test_no_on_wake_callback_passed(self):
        """After Part 3, wake_listener.start() is called without on_wake."""
        from bantz.interface.tui.app import BantzApp
        app = object.__new__(BantzApp)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        mock_listener = MagicMock()
        mock_listener.start = MagicMock(return_value=True)

        with patch("bantz.interface.tui.app.config") as mock_cfg:
            mock_cfg.wake_word_enabled = True
            mock_cfg.picovoice_access_key = "test-key"
            with patch.dict("sys.modules", {"bantz.agent.wake_word": MagicMock(wake_listener=mock_listener)}):
                app._start_wake_word_listener()
                mock_listener.start.assert_called_once()
                # Must NOT pass on_wake keyword argument
                _, kwargs = mock_listener.start.call_args
                assert "on_wake" not in kwargs

    def test_skips_when_disabled(self):
        from bantz.interface.tui.app import BantzApp
        app = object.__new__(BantzApp)

        with patch("bantz.interface.tui.app.config") as mock_cfg:
            mock_cfg.wake_word_enabled = False
            mock_listener = MagicMock()
            with patch.dict("sys.modules", {"bantz.agent.wake_word": MagicMock(wake_listener=mock_listener)}):
                app._start_wake_word_listener()
                mock_listener.start.assert_not_called()

    def test_skips_when_no_api_key(self):
        from bantz.interface.tui.app import BantzApp
        app = object.__new__(BantzApp)

        with patch("bantz.interface.tui.app.config") as mock_cfg:
            mock_cfg.wake_word_enabled = True
            mock_cfg.picovoice_access_key = ""
            mock_listener = MagicMock()
            with patch.dict("sys.modules", {"bantz.agent.wake_word": MagicMock(wake_listener=mock_listener)}):
                app._start_wake_word_listener()
                mock_listener.start.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# 11. action_quit — bus teardown
# ═══════════════════════════════════════════════════════════════════════════

class TestActionQuitBusTeardown:
    @pytest.mark.asyncio
    async def test_unsubscribes_and_shuts_down(self):
        from bantz.interface.tui.app import BantzApp
        app = object.__new__(BantzApp)
        app._unsubscribe_event_bus = MagicMock()
        app.exit = MagicMock()

        with patch("bantz.interface.tui.app.bus") as mock_bus:
            mock_bus.shutdown = AsyncMock()
            # Mock out other quit steps
            with patch.dict("sys.modules", {
                "bantz.core.gps_server": MagicMock(gps_server=MagicMock(stop=AsyncMock())),
                "bantz.agent.observer": MagicMock(observer=MagicMock()),
                "bantz.agent.wake_word": MagicMock(wake_listener=MagicMock()),
            }):
                await app.action_quit()
                app._unsubscribe_event_bus.assert_called_once()
                mock_bus.shutdown.assert_awaited_once()
                app.exit.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# 12. Source-level audit
# ═══════════════════════════════════════════════════════════════════════════

class TestSourceAudit:
    @pytest.fixture(autouse=True)
    def _read_source(self):
        import pathlib
        self.src = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "bantz" / "interface" / "tui" / "app.py"
        ).read_text()

    def test_imports_event_bus(self):
        assert "from bantz.core.event_bus import bus, Event" in self.src

    def test_bantz_event_message_class(self):
        assert "class BantzEventMessage(Message):" in self.src

    def test_subscribe_event_bus_method(self):
        assert "def _subscribe_event_bus(self)" in self.src

    def test_unsubscribe_event_bus_method(self):
        assert "def _unsubscribe_event_bus(self)" in self.src

    def test_relay_bus_event_method(self):
        assert "def _relay_bus_event(self, event: Event)" in self.src

    def test_on_bantz_event_message_handler(self):
        assert "def on_bantz_event_message(self, msg: BantzEventMessage)" in self.src

    def test_call_from_thread_in_relay(self):
        assert "self.call_from_thread(self.post_message, BantzEventMessage(event))" in self.src

    def test_subscribe_called_in_on_mount(self):
        assert "self._subscribe_event_bus()" in self.src

    def test_unsubscribe_called_in_action_quit(self):
        assert "self._unsubscribe_event_bus()" in self.src

    def test_bus_shutdown_in_action_quit(self):
        assert "await bus.shutdown()" in self.src

    def test_no_on_wake_closure_in_start_listener(self):
        """The legacy _on_wake closure must be gone from _start_wake_word_listener."""
        import inspect
        from bantz.interface.tui.app import BantzApp
        method_body = inspect.getsource(BantzApp._start_wake_word_listener)
        assert "def _on_wake()" not in method_body
        assert "on_wake=_on_wake" not in method_body

    def test_wake_listener_start_no_callback(self):
        """wake_listener.start() should be called without on_wake arg."""
        import inspect
        from bantz.interface.tui.app import BantzApp
        method_body = inspect.getsource(BantzApp._start_wake_word_listener)
        assert "wake_listener.start()" in method_body

    def test_three_bus_subscriptions(self):
        assert 'bus.on("wake_word_detected"' in self.src
        assert 'bus.on("ambient_change"' in self.src
        assert 'bus.on("health_alert"' in self.src

    def test_three_per_event_handlers(self):
        assert "def _on_bus_wake_word(self" in self.src
        assert "def _on_bus_ambient_change(self" in self.src
        assert "def _on_bus_health_alert(self" in self.src

    def test_bind_loop_called(self):
        assert "bus.bind_loop()" in self.src
