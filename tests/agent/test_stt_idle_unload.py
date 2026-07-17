"""STT idle unload (#560)."""
from unittest.mock import patch

from bantz.agent.stt import STTEngine
from bantz.config import config


def test_idle_unload_disabled_by_default():
    eng = STTEngine()
    eng._model = object()
    with patch.object(config, "stt_idle_unload_min", 0):
        eng._schedule_idle_unload()
    assert eng._unload_timer is None
    assert eng._model is not None


def test_idle_unload_timer_armed_and_fires():
    eng = STTEngine()
    eng._model = object()
    with patch.object(config, "stt_idle_unload_min", 1):
        eng._schedule_idle_unload()
        assert eng._unload_timer is not None
        eng._unload_timer.cancel()  # don't wait 60s in tests
        # Simulate the timer firing after a genuinely idle period.
        eng._last_used -= 61
        eng._unload_timer.function()
    assert eng._model is None
    assert eng._available is None  # reload allowed on next use


def test_recent_use_prevents_unload():
    eng = STTEngine()
    eng._model = object()
    with patch.object(config, "stt_idle_unload_min", 1):
        eng._schedule_idle_unload()
        eng._unload_timer.cancel()
        # _last_used is fresh — the timer body must keep the model.
        eng._unload_timer.function()
    assert eng._model is not None
